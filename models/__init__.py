from typing import TYPE_CHECKING
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

if TYPE_CHECKING:
    from typing import List, Union
    from torch import BatchEncoding
    from data_module.bAbI import bAbIItem

IGNORE_TOKEN = -100


class QuestionAnswerModel(LightningModule):
    def __init__(self, model_name: str, max_new_tokens=128):
        super().__init__()
        if "flan-t5" in model_name:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        elif "gpt-j" in model_name or "gpt-neo-1.3B" in model_name:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        gen_tokens = self.forward(batch["BatchEncoding"])

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
        prompts = [item.format(demonstrations, include_answer=False) for item in qa_items]
        predictions = [item.format(demonstrations, include_answer=False, include_prediction=True) for item in qa_items]
        answers = [item.format(demonstrations, include_answer=True) for item in qa_items]

        batch = prompts + predictions + answers
        batch_encodings = self.tokenizer(batch,
                                         padding="longest",
                                         max_length=self.tokenizer.model_max_length,
                                         truncation=True,
                                         return_tensors="pt")

        batch_encodings["input_ids"].to(self.device)
        batch_encodings["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(**batch_encodings, labels=batch_encodings["input_ids"]).logits
            logits_flatten = logits.reshape(-1, logits.shape[-1])

        target_ids = batch_encodings["input_ids"].clone()
        target_ids_flatten = target_ids.reshape(-1)

        negative_log_likelihood_loss = F.cross_entropy(logits_flatten,
                                                       target_ids_flatten,
                                                       reduction="none",
                                                       ignore_index=self.tokenizer.pad_token_id)

        negative_log_likelihood_loss = negative_log_likelihood_loss.reshape(batch_size * 3, -1)
        perplexities = torch.exp(negative_log_likelihood_loss.mean(1))

        prompts_ppl = perplexities[0 * batch_size: 1 * batch_size]
        predictions_ppl = perplexities[1 * batch_size: 2 * batch_size]
        answers_ppl = perplexities[2 * batch_size: 3 * batch_size]

        return prompts_ppl, predictions_ppl, answers_ppl
