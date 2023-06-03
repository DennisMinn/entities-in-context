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

    def forward(self, batch_encodings: "BatchEncoding") -> "torch.Tensor":
        """
        Generates tokens for question answer and perplexity associated with
        prompt

        Args:
            batch_encodings (BatchEncoding):
                Output from transformer tokenizer
        """
        # generate tokens
        predictions = self.model.generate(**batch_encodings,
                                          max_new_tokens=self.max_new_tokens,
                                          early_stopping=True)

        perplexities = self.calculate_perplexity(batch_encodings,
                                                 reduction="unnormalized")

        return predictions, perplexities

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
        predictions, perplexities = self.forward(batch["batch_encoding"])
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # update bAbIItem prediction attribute
        qa_items = batch["batch"]
        for idx, item in enumerate(qa_items):
            item.prediction = predictions[idx]
            item.prompt_perplexity = perplexities[idx]

        return qa_items

    def calculate_perplexity(self, batch_encodings, reduction="unnormalized"):
        self.model = self.model.to(self.device)
        logits = self.model(**batch_encodings, labels=batch_encodings["input_ids"]).logits
        logits_flatten = logits.reshape(-1, logits.shape[-1])

        target_ids = batch_encodings["input_ids"].clone()
        target_ids_flatten = target_ids.reshape(-1)

        negative_log_likelihood_loss = F.cross_entropy(
            logits_flatten,
            target_ids_flatten,
            reduction="none",
            ignore_index=self.tokenizer.pad_token_id
        )

        negative_log_likelihood_loss = negative_log_likelihood_loss.reshape(target_ids.shape)
        negative_log_likelihood_loss = negative_log_likelihood_loss.sum(1)
        if reduction == "unnormalized":
            return negative_log_likelihood_loss.tolist()
        else:
            pass
