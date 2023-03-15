from typing import TYPE_CHECKING
import torch
from transformers import AutoModel
from pytorch_lightning import LightningModule

if TYPE_CHECKING:
    from typing import List, Union
    from torch import BatchEncoding
    from data_module.bAbI import bAbIItem


class QuestionAnswerModel(LightningModule):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, inputs: BatchEncoding) -> torch.Tensor:
        '''
        Generates tokens for question answer

        Args:
            inputs (BatchEncoding):
                Output from transformer tokenizer
        '''
        # TODO: generate tokens
        pass

    def training_step(self, batch: dict[str, Union[List[bAbIItem], BatchEncoding]], batch_index: int):
        '''
        One iteration of training loop

        Args:
            batch (dict[str, Union[List[bAbIItem], BatchEncoding]]):
                Output of bAbIDataset.collate_fn
            batch_index (int):
                Index of subset
        '''
        # TODO: generate tokens
        # TODO: decode tokens
        # TODO: calculate accuracy
        # TODO: calculate f1 score
        # TODO: calculate recall
        # TODO: update bAbIItem prediction attribute
        pass

    def validation_step(self, batch: dict[str, Union[List[bAbIItem], BatchEncoding]], batch_index: int):
        '''
        One iteration of validation loop

        Args:
            batch (dict[str, Union[List[bAbIItem], BatchEncoding]]):
                Output of bAbIDataset.collate_fn
            batch_index (int):
                Index of subset
        '''
        # TODO: generate tokens
        # TODO: decode tokens
        # TODO: calculate accuracy
        # TODO: calculate f1 score
        # TODO: calculate recall
        # TODO: update bAbIItem prediction attribute
        pass
