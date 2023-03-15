from typing import TYPE_CHECKING
from transformers import AutoTokenizer
from data_modules import QuestionAnswerItem, QuestionAnswerDataset, QuestionAnswerDataModule

if TYPE_CHECKING:
    from typing import List, Union
    from pandas import DataFrame
    from torch import BatchEncoding
    from transformers import PreTrainedTokenizerFast


class bAbIItem(QuestionAnswerItem):
    '''
    Sample/Datum from bAbI dataset. Subclassed from QuestionAnswerItem

    Args:
        context (str):
            Text/Passage associated with question.

        question (str):
            Knowledge tested by datum. Question asked by datum.

        answer (str):
            Answer to asked question.

        prediction (str):
            Predicted answer to asked question.

        context_entities (List[Entity]):
            List of entities in context.

        question_entities (List[Entity]):
            List of entities in question.

        answer_entities (List[Entity]):
            List of entities in answer.

    '''
    def __init__(self, input):
        '''
        Parses in a single sample/datum from text file

        Args:
            input (List[str] or str):
                All information neccessary for question-answering task.

        '''
        # TODO decide typing of input
        # TODO change variable name "input" to something more descriptive
        # Don't worry about entities... yet
        self.context = None
        self.question = None
        self.answer = None


class bAbIDataset(QuestionAnswerDataset):
    def __init__(self,
                 demonstrations: str,
                 bAbI_items: List[bAbIItem],
                 tokenizer: PreTrainedTokenizerFast,
                 entity_dataframe: DataFrame,
                 entity_augmentation: str,
                 prompt_augmentation: str):

        self.demonstrations = demonstrations
        self.bAbI_items = bAbI_items
        self.tokenizer = tokenizer
        self.entity_dataframe = entity_dataframe
        self.entity_augmentation = entity_augmentation
        self.prompt_augmentation = prompt_augmentation

    def __getitem__(self, index: int) -> bAbIItem:
        return self.bAbI_items[index]

    def __len__(self) -> int:
        return len(self.bAbI_items)

    def collate_fn(self, batch: List[bAbIItem]) -> dict[str, Union[List[bAbIItem], BatchEncoding]]:
        '''
        Coalesces and formats list of bAbI items for model prediction

        Args:
            batch (List[bAbIItem]):
                Subset of bAbI dataset
        '''
        # TODO call QuestionAnswerItem.format() to convert bAbIItems to strings
        # TODO call self.tokenizer to convert string to input for model
        # return {
        #   "batch": batch,
        #   "BatchEncoding": batch_encoding
        # }
        pass

    @staticmethod
    def entity_statistics(self, fpath):
        pass


class bAbIDataModule(QuestionAnswerDataModule):
    def __init__(self,
                 model_name: str,
                 batch_size: int,
                 num_demonstrations: int = 2,
                 num_workers: int = 0,
                 prompt_augmentation: str = None,
                 entity_augmentation: str = None,
                 train_path: str = None,
                 validation_path: str = None,
                 test_path: str = None):

        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_demonstrations = num_demonstrations
        self.num_workers = num_workers
        self.prompt_augmentation = prompt_augmentation
        self.entity_augmentation = entity_augmentation
        self.train_path = train_path
        self.validation_path = validation_path
        self.test_path = test_path

        self.demonstrations = None

    def parse(self, fpath) -> bAbIDataset:
        # TODO read data from text file
        # TODO create list of bAbIItem from read data
        # TODO select demonstrations from list
        # TODO format and assign self.demonstrations to string using
        # QuestionAnswerDataModule.initialize_demonstrations
        # TODO create bAbIDataset using demonstrations and self.tokenizer and other
        # initial arguments passed during initialization.
        pass

    def setup(self, stage=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.datasets = {}
        if stage in (None, "fit"):
            self.datasets["train"] = self.parse(self.train_path)
            self.datasets["validation"] = self.parse(self.validation_path)

        if stage in (None, "test"):
            self.datasets["test"] = self.parse(self.test_path)
