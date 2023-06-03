from typing import TYPE_CHECKING
import string
import re
from abc import abstractmethod

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from dataclasses import dataclass, field
import copy
import random

from data_modules.constants import DEMONSTRATIONS, QUERY, BOTH, NUM_OF_DEMONSTRATIONS_TRIES


if TYPE_CHECKING:
    from typing import List
    from pandas import DataFrame
    from transformers import PreTrainedTokenizerFast

DEMONSTRATION_DELIMITER = "\n###\n"


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
    context_entities: "List[str]" = None
    question_entities: "List[str]" = None
    answer_entities: "List[str]" = None
    prompt_perplexity: "float" = field(default=None, repr=False)

    def __post_init__(self):
        self.context = self.context.strip()
        self.question = self.question.strip()
        self.answer = self.answer.strip()

    def format(self,
               demonstrations: str = "",
               include_answer: bool = False,
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
            query = f"context: {self.context}\nquestion: {self.question}\nanswer: {self.answer}"
        elif include_prediction:
            query = f"context: {self.context}\nquestion: {self.question}\nanswer: {self.prediction}"
        else:
            query = f"context: {self.context}\nquestion: {self.question}\nanswer:"

        if demonstrations:
            return demonstrations + DEMONSTRATION_DELIMITER + query
        else:
            return query

    def replace_entity(self, replacement_entity):
        qa_item = copy.deepcopy(self)
        if (
            not len(qa_item.context_entities) and
            not len(qa_item.question_entities) and
            not len(qa_item.answer_entities)
        ):
            return qa_item

        original_entity = (
            qa_item.question_entities +
            qa_item.answer_entities +
            qa_item.context_entities
        )[0]

        qa_item.context = qa_item.context.replace(original_entity, replacement_entity)
        qa_item.question = qa_item.question.replace(original_entity, replacement_entity)
        qa_item.answer = qa_item.answer.replace(original_entity, replacement_entity)

        qa_item.context_entities = [*map(lambda entity: entity.replace(original_entity, replacement_entity), qa_item.context_entities)]
        qa_item.question_entities = [*map(lambda entity: entity.replace(original_entity, replacement_entity), qa_item.question_entities)]
        qa_item.answer_entities = [*map(lambda entity: entity.replace(original_entity, replacement_entity), qa_item.answer_entities)]

        return qa_item

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

    @staticmethod
    def separate_last_sentence(text):
        sentences = text.split(".")
        sentences = [sentence.strip() for sentence in sentences]

        context = ". ".join(sentences[:-1]) + "."
        question = sentences[-1]
        return context, question


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

    def initialize_demonstrations(self, qa_items: "List[QuestionAnswerItem]"):
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
            demonstrations = DEMONSTRATION_DELIMITER.join([item.format(include_answer=True) for item in demonstrations])
            return demonstrations

        # Initialize by randomly sampling dataset
        for _ in range(NUM_OF_DEMONSTRATIONS_TRIES):
            demonstrations = random.sample(qa_items, self.num_demonstrations)
            demonstrations = DEMONSTRATION_DELIMITER.join([item.format(include_answer=True) for item in demonstrations])

            num_tokens = len(self.tokenize.encode(demonstrations))
            if num_tokens <= self.max_demonstrations_token_length:
                return demonstrations

        raise Exception("Could not initialize the demonstrations within the specified token length")

    def initialize_replacement_entity(self):
        if self.entity_augmentation is None:
            return None

        if "demographics" in self.entity_augmentation:
            replacement_entity = self.entity_augmentation.split("_")[-1]
            return replacement_entity

        raise Exception("Unlisted entity augmentation")

    @abstractmethod
    def collate_fn(self, batch: "List[QuestionAnswerItem]") -> dict:
        '''
        Coalesces and formats list of question-answer items for model prediction

        Args:
            batch (List[QuestionAnswerItem]):
                Subset from an arbitrary Question-Answering dataset
        '''

        raise NotImplementedError

    def calculate_accuracy(self):
        from functools import reduce
        accuracy = reduce(lambda accuracy, item: accuracy + item.calculate_accuracy(), self.qa_items, 0)
        accuracy /= len(self.qa_items)
        return accuracy

    def calculate_f1(self):
        from functools import reduce
        f1 = reduce(lambda f1, item: f1 + item.calculate_f1(), self.qa_items, 0)
        f1 /= len(self.qa_items)
        return f1

    def calculate_perplexity(self):
        from functools import reduce
        perplexity = reduce(lambda perplexity, item: perplexity + item.prompt_perplexity, self.qa_items, 0)
        perplexity /= len(self.qa_items)

        return perplexity

    def export(self, fpath):
        import json
        with open(fpath, "w") as outfile:
            for item in self:
                json_line = json.dumps({
                    "prompt": item.format(self.demonstrations),
                    "answer": item.answer,
                    "prediction": item.prediction,
                    "prompt_perplexity": item.prompt_perplexity
                })

                outfile.write(json_line + "\n")


@dataclass
class QuestionAnswerDataModule(LightningDataModule):
    model_name: str
    batch_size: int
    data_directory: str
    num_demonstrations: int = -1
    max_demonstrations_token_length: int = 400
    demonstration_indices: "List[int]" = None
    num_workers: int = 0
    prompt_augmentation: str = None
    entity_augmentation: str = None

    def __post_init__(self):
        super().__init__()
        self.datasets = {}
        self.tokenizer = self.initialize_tokenizer()

    @abstractmethod
    def setup(self, stage: str = None):
        raise NotImplementedError

    @abstractmethod
    def parse(self, fpath: str) -> QuestionAnswerDataset:
        raise NotImplementedError

    def train_dataloader(self):
        if isinstance(self.datasets["train"], list):
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
        if isinstance(self.datasets["validation"], list):
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
        if isinstance(self.datasets["test"], list):
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

    def initialize_tokenizer(self):
        from transformers import AutoTokenizer

        if "flan-t5" in self.model_name:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        elif "gpt" in self.model_name:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer

    def _load_pickle(self, fpath):
        import os
        import pickle

        root, _ = os.path.splitext(fpath)
        pickle_fpath = root + ".pkl"
        pickle_file = open(pickle_fpath, "rb")
        qa_items = pickle.load(pickle_file)
        return qa_items

    def _save_pickle(self, qa_items, fpath):
        import os
        import pickle

        root, _ = os.path.splitext(fpath)
        pickle_fpath = root + ".pkl"
        pickle_file = open(pickle_fpath, "wb")
        qa_items = pickle.dump(qa_items, pickle_file)
