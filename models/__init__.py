from typing import TYPE_CHECKING
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

if TYPE_CHECKING:
    from typing import List, Union
    from torch import BatchEncoding
    from data_module.bAbI import bAbIItem

NER_MODEL_PATH = "dslim/bert-base-NER"
PERSON_ENTITY = "PER"
SUBWORD_TOKEN = "##"
CONFIDENCE_THRESHOLD = 0.5


class NERModel():
    def __init__(self):
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_PATH)

    def annotate_entities(self, text):
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        annotations = pipeline("ner", model=self.model, tokenizer=self.tokenizer, device=device)(text)
        entities = []

        for i, token in enumerate(annotations):
            if not token["entity"].endswith(PERSON_ENTITY):
                continue

            if token["score"] < CONFIDENCE_THRESHOLD:
                continue

            if SUBWORD_TOKEN not in token["word"]:
                entities.append(token["word"])
            elif (
                SUBWORD_TOKEN in token["word"] and
                len(entities) and
                len(annotations) and
                annotations[i]["start"] == annotations[i-1]["end"]
            ):
                entities[-1] += token["word"].replace(SUBWORD_TOKEN, "")
            else:
                pass

        return set(entities)


class QuestionAnswerModel(LightningModule):
    def __init__(self, data_module, max_new_tokens=64):
        super().__init__()
        self.tokenizer = data_module.tokenizer
        self.model_name = self.tokenizer.name_or_path
        if "flan-t5" in self.model_name:
            from transformers import AutoModelForSeq2SeqLM
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        elif "gpt" in self.model_name:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.max_new_tokens = max_new_tokens

    def forward(self, inputs: "BatchEncoding") -> "torch.Tensor":
        """
        Generates tokens for question answer

        Args:
            inputs (BatchEncoding):
                Output from transformer tokenizer
        """
        # generate tokens
        gen_tokens = self.model.generate(**inputs,
                                         max_new_tokens=self.max_new_tokens,
                                         early_stopping=True)

        return gen_tokens

    def validation_step(self,
                        batch: "dict[str, Union[List[bAbIItem], BatchEncoding]]",
                        batch_index: int,
                        dataset_index: int = 0):
        """
        One iteration of validation loop

        Args:
            batch (dict[str, Union[List[bAbIItem], BatchEncoding]]):
                Output of bAbIDataset.collate_fn
            batch_index (int):
                Index of subset
        """
        # generate tokens
        gen_tokens = self.forward(batch["batch_encoding"])

        # decode tokens
        gen_text = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        # update bAbIItem prediction attribute
        qa_items = batch["batch"]
        demonstrations = batch["demonstrations"]
        for idx, item in enumerate(qa_items):
            item.prediction = gen_text[idx]

        prompts_ppl, predictions_ppl, answers_ppl = self.calculate_perplexity(demonstrations, qa_items)
        for idx, item in enumerate(qa_items):
            item.prompt_perplexity = prompts_ppl[idx].item()
            item.prediction_perplexity = predictions_ppl[idx].item()
            item.answer_perplexity = answers_ppl[idx].item()

        return qa_items

    def calculate_perplexity(self, demonstrations, qa_items):
        batch_size = len(qa_items)
        prompts_start, prompts_end = 0 * batch_size, 1 * batch_size,
        predictions_start, predictions_end = 1 * batch_size, 2 * batch_size
        answers_start, answers_end = 2 * batch_size, 3 * batch_size

        prompts = [item.format(demonstrations) for item in qa_items]
        predictions = [item.format(demonstrations, include_prediction=True) for item in qa_items]
        answers = [item.format(demonstrations, include_answer=True) for item in qa_items]

        batch = prompts + predictions + answers
        batch_encodings = self.tokenizer(batch,
                                         padding="longest",
                                         max_length=self.tokenizer.model_max_length,
                                         truncation=True,
                                         return_tensors="pt")

        batch_encodings["input_ids"] = batch_encodings["input_ids"].to(self.device)
        batch_encodings["attention_mask"] = batch_encodings["attention_mask"].to(self.device)

        with torch.no_grad():
            self.model = self.model.to(self.device)
            logits = self.model(**batch_encodings, labels=batch_encodings["input_ids"]).logits
            logits_flatten = logits.reshape(-1, logits.shape[-1])

        target_ids = batch_encodings["input_ids"].clone()
        prediction_ids = target_ids[predictions_start: predictions_end]
        answer_ids = target_ids[answers_start: answers_end]

        if "flan-t5" in self.model_name:
            self._mask_target_ids(prompts, prediction_ids)
            self._mask_target_ids(prompts, answer_ids)

        elif "gpt" in self.model_name:
            self._mask_target_ids([item.prediction for item in qa_items], prediction_ids, inverse=True)
            self._mask_target_ids([item.answer for item in qa_items], answer_ids, inverse=True)

        target_ids_flatten = target_ids.reshape(-1)

        negative_log_likelihood_loss = F.cross_entropy(
            logits_flatten,
            target_ids_flatten,
            reduction="none",
            ignore_index=self.tokenizer.pad_token_id
        )

        negative_log_likelihood_loss = negative_log_likelihood_loss.reshape(target_ids.shape)
        perplexities = torch.exp(negative_log_likelihood_loss.mean(1))

        prompts_ppl = perplexities[prompts_start: prompts_end]
        predictions_ppl = perplexities[predictions_start: predictions_end]
        answers_ppl = perplexities[answers_start: answers_end]

        return prompts_ppl, predictions_ppl, answers_ppl

    def _mask_target_ids(self, text, target_ids, inverse=False):
        sequence_length = target_ids.shape[1]
        ignore_index = self.tokenizer.pad_token_id
        attention_mask = self.tokenizer(text,
                                        padding="max_length",
                                        max_length=sequence_length,
                                        add_special_tokens=False,
                                        truncation=True,
                                        return_tensors="pt")["attention_mask"]
        attention_mask = attention_mask.to(self.device)

        # TODO target_ids.masked_fill(~attention_mask, ignore_index) doesn't work
        if not inverse:
            target_ids.masked_fill_(attention_mask == 1, ignore_index)
        else:
            target_ids.masked_fill_(attention_mask == 0, ignore_index)
        return target_ids
