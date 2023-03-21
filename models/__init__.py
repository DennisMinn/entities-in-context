from typing import TYPE_CHECKING
import torch
from transformers import AutoModel, AutoTokenizer
from pytorch_lightning import LightningModule
from model.metrics import accuracy

if TYPE_CHECKING:
    from typing import List, Union
    from torch import BatchEncoding
    from data_module.bAbI import bAbIItem


class QuestionAnswerModel(LightningModule):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, inputs: BatchEncoding) -> torch.Tensor:
        '''
        Generates tokens for question answer

        Args:
            inputs (BatchEncoding):
                Output from transformer tokenizer
        '''
        # generate tokens
        input_ids = inputs.input_ids

        gen_tokens = self.model.generate(
            input_ids,
            do_sample=True,
            temperature=0.9,
            max_length=100,
        )

        return gen_tokens

    def training_step(self, batch: dict[str, Union[List[bAbIItem], BatchEncoding]], batch_index: int):
        '''
        One iteration of training loop

        Args:
            batch (dict[str, Union[List[bAbIItem], BatchEncoding]]):
                Output of bAbIDataset.collate_fn
            batch_index (int):
                Index of subset
        '''
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

    def validation_step(self, batch: dict[str, Union[List[bAbIItem], BatchEncoding]], batch_index: int):
        '''
        One iteration of validation loop

        Args:
            batch (dict[str, Union[List[bAbIItem], BatchEncoding]]):
                Output of bAbIDataset.collate_fn
            batch_index (int):
                Index of subset
        '''
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
