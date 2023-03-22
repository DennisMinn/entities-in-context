from typing import TYPE_CHECKING
from transformers import AutoTokenizer
from data_modules import QuestionAnswerItem, QuestionAnswerDataset, QuestionAnswerDataModule
from functools import reduce

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


class bAbIDataset(QuestionAnswerDataset):
    def __init__(self,
                 bAbI_items: "List[bAbIItem]",
                 tokenizer: "PreTrainedTokenizerFast",
                 entity_dataframe: "DataFrame" = None,
                 entity_augmentation: str = None,
                 prompt_augmentation: str = None,
                 num_demonstrations: int = 5):

        self.num_demonstrations = num_demonstrations
        self.demonstrations = self.initialize_demonstrations(bAbI_items)
        self.bAbI_items = bAbI_items
        self.tokenizer = tokenizer
        self.entity_dataframe = entity_dataframe
        self.entity_augmentation = entity_augmentation
        self.prompt_augmentation = prompt_augmentation

    def __getitem__(self, index: int) -> bAbIItem:
        return self.bAbI_items[index]

    def __len__(self) -> int:
        return len(self.bAbI_items)

    def collate_fn(self, batch: "List[bAbIItem]") -> "dict[str, Union[List[bAbIItem], BatchEncoding]]":
        '''
        Coalesces and formats list of bAbI items for model prediction

        Args:
            batch (List[bAbIItem]):
                Subset of bAbI dataset
        '''
        # call QuestionAnswerItem.format() to convert bAbIItems to strings
        formatted_batch = [item.format(item, self.demonstrations) for item in batch]

        # call self.tokenizer to convert string to input for model
        batch_encoding = self.tokenizer(formatted_batch, padding='max_length', return_tensors='pt')
        return {
          "batch": batch,
          "BatchEncoding": batch_encoding
        }

    @staticmethod
    def entity_statistics(self, fpath):
        pass


class bAbIDataModule(QuestionAnswerDataModule):
    # TODO remove default values for some arguments
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

    def parse(self, fpath) -> "List[bAbIDataset]":
        with open(fpath) as f:
            lines = f.readlines()

        data = []
        story = []
        for line in lines:
            nid, line = line.split(' ', 1)
            if nid == "1":
                # reset story when line ID=1 (start of new story)
                story.clear()
            if '\t' in line:
                # this line is tab separated Q, A & support fact ID
                q, a, supporting = line.split('\t')
                # Provide all the sub-stories till this question
                substory = [x for x in story if story]
                # A story ends and is appended to global story data-set
                data.append((substory, q, a))
                story.append('')
            else:
                # this line is a sentence of story
                story.append(line)
        # lambda func to flatten the list of sentences into one list
        flatten = lambda data: reduce(lambda x, y: x + y, data)
        # creating list of dataclasses for each task
        data = [bAbIItem(flatten(story).replace('\n', ' '), q, answer) for story, q, answer in data]
        return data

    def setup(self, stage=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.data = {}
        self.data["train"] = self.parse(self.train_path)
        self.data["validation"] = self.parse(self.validation_path)
        self.data["test"] = self.parse(self.test_path)

        self.datasets = {}
        self.datasets["train"] = bAbIDataset(
            bAbI_items=self.data["train"],
            tokenizer=self.tokenizer,
            entity_dataframe=None,
            entity_augmentation=self.entity_augmentation,
            prompt_augmentation=self.prompt_augmentation,
            num_demonstrations=self.num_demonstrations
        )

        self.datasets["validation"] = bAbIDataset(
            bAbI_items=self.data["validation"],
            tokenizer=self.tokenizer,
            entity_dataframe=None,
            entity_augmentation=self.entity_augmentation,
            prompt_augmentation=self.prompt_augmentation,
            num_demonstrations=0
        )
        self.datasets["validation"].demonstrations = self.datasets["train"].demonstrations

        self.datasets["test"] = bAbIDataset(
            bAbI_items=self.data["test"],
            tokenizer=self.tokenizer,
            entity_dataframe=None,
            entity_augmentation=self.entity_augmentation,
            prompt_augmentation=self.prompt_augmentation,
            num_demonstrations=0
        )
        self.datasets["test"].demonstrations = self.datasets["train"].demonstrations