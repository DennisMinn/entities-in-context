from typing import TYPE_CHECKING
import torch
from pytorch_lightning import LightningModule
from model.metrics import accuracy

if TYPE_CHECKING:
    from typing import List, Union
    from torch import BatchEncoding
    from data_module.bAbI import bAbIItem


class QuestionAnswerModel(LightningModule):
    def __init__(self, model_name: str):
        super().__init__()
        if "flan-t5" in model_name:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif "gpt-j" in model_name:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, inputs: BatchEncoding) -> torch.Tensor:
        """
        Generates tokens for question answer

        Args:
            inputs (BatchEncoding):
                Output from transformer tokenizer
        """
        # generate tokens
        gen_tokens = self.model.generate(inputs.input_ids)

        return gen_tokens

    def training_step(
        self, batch: dict[str, Union[List[bAbIItem], BatchEncoding]], batch_index: int
    ):
        """
        One iteration of training loop

        Args:
            batch (dict[str, Union[List[bAbIItem], BatchEncoding]]):
                Output of bAbIDataset.collate_fn
            batch_index (int):
                Index of subset
        """
        with torch.no_grad():
            # generate tokens
            gen_tokens = self.forward(batch["BatchEncoding"])

            # decode tokens
            gen_text = self.tokenizer.batch_decode(gen_tokens)

            # calculate accuracy
            babi_items = batch["batch"]
            labels = [item.answer for item in babi_items]
            metrics = {
                "accuracy": accuracy(labels, gen_text),
            }

            # TODO: calculate f1 score
            # TODO: calculate recall
            # update bAbIItem prediction attribute
            for idx, item in enumerate(babi_items):
                item.prediction = gen_text[idx]

            return metrics

    def validation_step(
        self, batch: dict[str, Union[List[bAbIItem], BatchEncoding]], batch_index: int
    ):
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
        gen_text = self.tokenizer.batch_decode(gen_tokens)

        # calculate accuracy
        babi_items = batch["batch"]
        labels = [item.answer for item in babi_items]
        metrics = {
            "accuracy": accuracy(labels, gen_text),
        }

        # TODO: calculate f1 score
        # TODO: calculate recall
        # update bAbIItem prediction attribute
        for idx, item in enumerate(babi_items):
            item.prediction = gen_text[idx]

        return metrics
