from typing import List, TYPE_CHECKING
from abc import abstractmethod
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from dataclasses import dataclass, field
from data_modules.entities import Entity
import pandas as pd
import random

from data_modules.constants import DEMONSTRATIONS, QUERY, BOTH, CONTEXT, QUESTION, ANSWER, NEXT_LINE, NUM_OF_DEMONSTRATIONS_TRIES


if TYPE_CHECKING:
    from pandas import DataFrame
    from transformers import PreTrainedTokenizerFast


@dataclass
class QuestionAnswerItem():
    '''
    Sample/Datum from a question-answer dataset (bAbI,
    SuperNaturalInstructions, etc).

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
            List of entities in context

        question_entities (List[Entity]):
            List of entities in question

        answer_entities (List[Entity]):
            List of entities in answer
  '''
    context: str
    question: str
    answer: str
    prediction: str = field(default=None)
    context_entities: List[Entity] = field(default=None)
    question_entities: List[Entity] = field(default=None)
    answer_entities: List[Entity] = field(default=None)

    def format(self, demonstrations: str = "", include_answer: bool = True) -> str:
        '''
        Concatenates `QuestionAnswerItem.context` and
        `QuestionAnswerItem.question` with demonstrations for in-context
        learning. To limit memory overhead, this function is static.

        Args:
            qa_item (QuestionAnswerItem):
                Sample from question-answer dataset.

            demonstrations (str):
                Demonstrations/Examples for the model to use as a template.
        '''
        if include_answer:
            query = CONTEXT + self.context + NEXT_LINE + QUESTION + self.question + NEXT_LINE + ANSWER + self.answer + NEXT_LINE
        else:
            query = CONTEXT + self.context + NEXT_LINE + QUESTION + self.question + NEXT_LINE + ANSWER

        return demonstrations + query

    def replace_entity(self, replacement_entity):
        if len(self.question_entities) or len(self.answer_entities):
            entities = self.question_entities + self.answer_entities
            entity = entities[0]

            context = self.context.replace(entity.text, replacement_entity.text)
            question = self.question.replace(entity.text, replacement_entity.text)
            answer = self.answer.replace(entity.text, replacement_entity.text)

            return QuestionAnswerItem(context, question, answer)
        else:
            return QuestionAnswerItem(self.context, self.question, self.answer)

    def logging(self):
        return [self.context, self.question, self.answer, self.prediction]


@dataclass
class QuestionAnswerDataset(Dataset):
    qa_items: "List[QuestionAnswerItem]" = field(repr=False)
    tokenizer: "PreTrainedTokenizerFast" = field(repr=False)
    entities_dataframe: "DataFrame" = field(repr=False, default=None)
    entity_augmentation: str = None
    prompt_augmentation: str = None
    num_demonstrations: int = -1
    max_demonstrations_token_length: int = 400
    demonstration_indices: "List[int]" = None

    def __post_init__(self):
        self.replacement_entity = self.initialize_replacement_entity()
        self.demonstrations = self.initialize_demonstrations(self.qa_items)

    def __getitem__(self, index: int) -> QuestionAnswerItem:
        if self.prompt_augmentation in [QUERY, BOTH]:
            return self.qa_items[index].replace_entity(self.replacement_entity)
        else:
            return self.qa_items[index]

    def __len__(self) -> int:
        return len(self.qa_items)

    def initialize_demonstrations(self, qa_items: List[QuestionAnswerItem]):
        if self.demonstration_indices is None and self.num_demonstrations == -1:
            return None

        # Apply prompt augmentation
        if self.prompt_augmentation in [DEMONSTRATIONS, BOTH]:
            qa_items = [item.replace_entity(self.replacement_entity) for item in qa_items]

        # Initialize using specified demonstration indices
        if self.demonstration_indices is not None:
            k = self.num_demonstrations if self.num_demonstrations != -1 else len(self.demonstration_indices)
            demonstration_indices = self.demonstration_indices[:k]
            demonstrations = [qa_items[index] for index in demonstration_indices]
            demonstrations = "".join([item.format() for item in demonstrations])
            return demonstrations

        # Initialize by randomly sampling dataset
        for _ in range(NUM_OF_DEMONSTRATIONS_TRIES):
            demonstrations = random.sample(qa_items, self.num_demonstrations)
            demonstrations = "".join([item.format() for item in demonstrations])

            num_tokens = len(self.tokenize.encode(demonstrations))
            if num_tokens <= self.max_demonstrations_token_length:
                return demonstrations

        raise Exception("Could not initialize the demonstrations within the specified token length")

    def initialize_replacement_entity(self):
        if self.entity_augmentation is None:
            return None

        if "demographics" in self.entity_augmentation:
            entity_text = self.entity_augmentation.split("_")[-1]
            entities_dataframe = self.entities_dataframe.entity.filter(dataset="demographics")
            replacement_entity = entities_dataframe.entity.entities[entity_text]
            return replacement_entity

        raise Exception("Unlisted entity augmentation")

    @abstractmethod
    def collate_fn(self, batch: List[QuestionAnswerItem]) -> dict:
        '''
        Coalesces and formats list of question-answer items for model prediction

        Args:
            batch (List[QuestionAnswerItem]):
                Subset from an arbitrary Question-Answering dataset
        '''

        raise NotImplementedError


@dataclass
class QuestionAnswerDataModule(LightningDataModule):
    model_name: str
    batch_size: int
    data_directory: str
    entities_metadata_fpath: str
    num_demonstrations: int = -1
    max_demonstrations_token_length: int = 400
    demonstration_indices: List[int] = None
    num_workers: int = 0
    prompt_augmentation: str = None
    entity_augmentation: str = None

    def __post_init__(self):
        self.entities_dataframe = pd.read_csv(self.entities_metadata_fpath)
        self.datasets = {}

    @abstractmethod
    def setup(self, stage: str = None):
        raise NotImplementedError

    @abstractmethod
    def parse(self, fpath: str) -> QuestionAnswerDataset:
        raise NotImplementedError

    def train_dataloader(self):
        if isinstance(self.datasets["train"], List):
            dataloaders = [
                DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    shuffle=False,
                    pin_memory=True,
                    collate_fn=dataset.collate_fn
                )
                for dataset in self.datasets["train"]
            ]
            return dataloaders

        else:
            dataloader = DataLoader(
                self.datasets["train"],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
                collate_fn=self.datasets["train"].collate_fn
            )
            return dataloader

    def val_dataloader(self):
        if isinstance(self.datasets["validation"], List):
            dataloaders = [
                DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    shuffle=False,
                    pin_memory=True,
                    collate_fn=dataset.collate_fn
                )
                for dataset in self.datasets["validation"]
            ]
            return dataloaders

        else:
            dataloader = DataLoader(
                self.datasets["validation"],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
                collate_fn=self.datasets["validation"].collate_fn
            )
            return dataloader

    def test_dataloader(self):
        if isinstance(self.datasets["test"], List):
            dataloaders = [
                DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    shuffle=False,
                    pin_memory=True,
                    collate_fn=dataset.collate_fn
                )
                for dataset in self.datasets["test"]
            ]
            return dataloaders

        else:
            dataloader = DataLoader(
                self.datasets["test"],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
                collate_fn=self.datasets["test"].collate_fn
            )
            return dataloader
