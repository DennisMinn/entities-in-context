from typing import List, TYPE_CHECKING
import string
import re
from abc import abstractmethod
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from dataclasses import dataclass, field
from data_modules.entities import Entity
from functools import reduce
import pandas as pd
import random

from data_modules.constants import DEMONSTRATIONS, QUERY, BOTH, CONTEXT, QUESTION, ANSWER, NEXT_LINE, NUM_OF_DEMONSTRATIONS_TRIES


if TYPE_CHECKING:
    from pandas import DataFrame
    from transformers import PreTrainedTokenizerFast


# These functions are directly taken from:
# https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#:~:text=F1%20score%20is%20a%20common,those%20in%20the%20True%20Answer.
def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


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
    prediction: str = None
    context_entities: List[Entity] = None
    question_entities: List[Entity] = None
    answer_entities: List[Entity] = None
    prompt_perplexity: float = field(default=0.0, repr=False)
    answer_perplexity: float = field(default=0.0, repr=False)
    prediction_perplexity: float = field(default=0.0, repr=False)

    def format(self, demonstrations: str = "", include_answer: bool = True,
               include_prediction: bool = False) -> str:
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
        if include_answer and include_prediction:
            raise Exception("include_answer = True and include_prediciton = True")

        if include_answer:
            query = CONTEXT + self.context + NEXT_LINE + QUESTION + self.question + NEXT_LINE + ANSWER + self.answer + NEXT_LINE
        elif include_prediction:
            query = CONTEXT + self.context + NEXT_LINE + QUESTION + self.question + NEXT_LINE + ANSWER + self.prediction + NEXT_LINE
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

    def calculate_f1(self):
        pred_tokens = normalize_text(self.prediction).split()
        truth_tokens = normalize_text(self.answer).split()

        # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)

        common_tokens = set(pred_tokens) & set(truth_tokens)

        # if there are no common tokens then f1 = 0
        if len(common_tokens) == 0:
            return 0

        prec = len(common_tokens) / len(pred_tokens)
        rec = len(common_tokens) / len(truth_tokens)

        return 2 * (prec * rec) / (prec + rec)

    def calculate_accuracy(self):
        return self.prediction == self.answer

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

    def calculate_accuracy(self):
        accuracy = reduce(lambda accuracy, item: accuracy + item.calculate_accuracy(), self.qa_items, 0)
        accuracy /= len(self.qa_items)
        return accuracy

    def calculate_f1(self):
        f1 = reduce(lambda f1, item: f1 + item.calculate_f1(), self.qa_items, 0)
        f1 /= len(self.qa_items)
        return f1

    def calculate_perplexities(self):
        prompt_perplexity = reduce(lambda prompt_perplexity, item: prompt_perplexity + item.prompt_perplexity, self.qa_items, 0)
        prompt_perplexity /= len(self.qa_items)

        answer_perplexity = reduce(lambda answer_perplexity, item: answer_perplexity + item.answer_perplexity, self.qa_items, 0)
        answer_perplexity /= len(self.qa_items)

        prediction_perplexity = reduce(lambda prompt_perplexity, item: prompt_perplexity + item.prediction_perplexity, self.qa_items, 0)
        prediction_perplexity /= len(self.qa_items)

        return prompt_perplexity, answer_perplexity, prediction_perplexity


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
        super().__init__()
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
