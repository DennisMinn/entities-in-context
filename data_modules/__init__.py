from abc import abstractmethod
from typing import List
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from dataclasses import dataclass, field
from data_modules.entities import Entity
import random

CONTEXT = 'context: '
QUESTION = 'question: '
ANSWER = 'answer: '
NEXT_LINE = '\n'
K = 5
NUM_OF_DEMONSTRATIONS_TRIES = 50


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
    # TODO find method to parse entities list
    context: str
    question: str
    answer: str
    prediction: str = field(default=None)
    context_entities: List[Entity] = field(default=None)
    question_entities: List[Entity] = field(default=None)
    answer_entities: List[Entity] = field(default=None)

    @staticmethod
    def format(question_answer_item: "QuestionAnswerItem", demonstrations: str = "") -> str:
        '''
        Concatenates `QuestionAnswerItem.context` and
        `QuestionAnswerItem.question` with demonstrations for in-context
        learning. To limit memory overhead, this function is static.

        Args:
            question_answer_item (QuestionAnswerItem):
                Sample from question-answer dataset.

            demonstrations (str):
                Demonstrations/Examples for the model to use as a template.
        '''
        # format question_answer_item to a string
        qa_item_string = CONTEXT + question_answer_item.context + NEXT_LINE + QUESTION + question_answer_item.question + NEXT_LINE + ANSWER

        # append QA string to demonstrations
        return demonstrations + qa_item_string

    @staticmethod
    def replace_entity(question_answer_item, entity):
        pass

    def logging(self):
        return [self.context, self.question, self.answer, self.prediction]


class QuestionAnswerDataset(Dataset):

    def initialize_demonstrations(self, question_answer_items: List[QuestionAnswerItem]):
        # repeatedly call QuestionAnswerItem demonstrations until
        # demonstation string is formed
        demonstrations = ""
        initialization_successful = False
        for _ in range(NUM_OF_DEMONSTRATIONS_TRIES):
            for _ in range(self.num_demonstrations):
                random_index = random.randrange(len(question_answer_items))
                question_answer_item = question_answer_items[random_index]
                demonstrations = demonstrations + CONTEXT + question_answer_item.context + NEXT_LINE + QUESTION + question_answer_item.question + NEXT_LINE + ANSWER + question_answer_item.answer + NEXT_LINE
            tokenized_demonstrations = self.tokenizer(demonstrations, return_tensors='pt')
            tokenized_demonstrations_len = len(tokenized_demonstrations['input_ids'][0])
            if tokenized_demonstrations_len <= self.max_demonstrations_token_length:
                initialization_successful = True
                break
        if not initialization_successful:
            raise Exception("Could not initialize the demonstrations within the specified token length")
        return demonstrations

    @abstractmethod
    def collate_fn(self, batch: List[QuestionAnswerItem]) -> dict:
        '''
        Coalesces and formats list of question-answer items for model prediction

        Args:
            batch (List[QuestionAnswerItem]):
                Subset from an arbitrary Question-Answering dataset
        '''

        raise NotImplementedError


class QuestionAnswerDataModule(LightningDataModule):
    @abstractmethod
    def setup(self, stage: str = None):
        raise NotImplementedError

    @abstractmethod
    def parse(self, fpath: str) -> QuestionAnswerDataset:
        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.datasets["train"].collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.datasets["validation"].collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.datasets["test"].collate_fn
        )
