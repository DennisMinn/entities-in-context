from typing import TYPE_CHECKING
import torch
from pytorch_lightning import LightningModule

if TYPE_CHECKING:
    from typing import List, Union
    from torch import BatchEncoding
    from data_module.bAbI import bAbIItem


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

        # calculate accuracy
        qa_items = batch["batch"]

        # update bAbIItem prediction attribute
        for idx, item in enumerate(qa_items):
            item.prediction = gen_text[idx]

        return qa_items

    def calculate_perplexity(self, prompt, target, mask_prompt):
        prompt_length = len(self.tokenizer.encode(prompt))
        text = prompt + target
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        target_ids = input_ids.clone()
        if mask_prompt:
            target_ids[:, :prompt_length] = -100

        with torch.no_grad():
            negative_log_likelihood = self.model(input_ids, labels=target_ids).loss
            perplexity = torch.exp(negative_log_likelihood)

        return perplexity.item()
