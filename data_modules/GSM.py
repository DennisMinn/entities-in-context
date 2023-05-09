from typing import TYPE_CHECKING
import os
import re
import pandas as pd
from collections import defaultdict
from functools import reduce
import spacy
import json
from transformers import AutoTokenizer
from data_modules import QuestionAnswerItem, QuestionAnswerDataset, QuestionAnswerDataModule
from data_modules.entities import NER_MODEL_NAME, Entity

if TYPE_CHECKING:
    from typing import List, Union
    from pandas import DataFrame
    from torch import BatchEncoding
    from transformers import PreTrainedTokenizerFast

ANSWER_REGEX = PREDICTION_REGEX = re.compile(r"#### (\-?[0-9\.\,]+)")
ANSWER_EOL = "\n"
QUSETION_EOL = "<|endoftext|>"
INVALID_ANSWER = INVALID_PREDICTION = "[invalid]"
ACCEPTABLE_ENTITIES = ("PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "FAC", "LOC")


# Function derived from:
# https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py
class GSMItem(QuestionAnswerItem):
    '''
    Sample/Datum from GSM dataset. Subclassed from QuestionAnswerItem

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
    def get_answer(self):
        match = ANSWER_REGEX.search(self.answer)
        if match:
            match = match.group(1).strip()
            match = match.replace(",", "")
            return match
        else:
            return INVALID_ANSWER

    def get_prediction(self):
        match = PREDICTION_REGEX.search(self.prediction)
        if match:
            match = match.group(1).strip()
            match = match.replace(",", "")
            return match
        else:
            return INVALID_PREDICTION

    def is_correct(self):
        answer = self.get_answer()
        assert answer != INVALID_ANSWER

        prediction = self.get_prediction()
        return answer == prediction


class GSMDataset(QuestionAnswerDataset):
    def __init__(self,
                 GSM_items: "List[GSMItem]",
                 tokenizer: "PreTrainedTokenizerFast",
                 entities_dataframe: "DataFrame" = None,
                 entity_augmentation: str = None,
                 prompt_augmentation: str = None,
                 num_demonstrations: int = -1,
                 max_demonstrations_token_length: int = 400,
                 demonstration_indices: "List[int]" = None):

        super().__init__(
            question_answer_items=GSM_items,
            tokenizer=tokenizer,
            entities_dataframe=entities_dataframe,
            entity_augmentation=entity_augmentation,
            prompt_augmentation=prompt_augmentation,
            num_demonstrations=num_demonstrations,
            max_demonstrations_token_length=max_demonstrations_token_length,
            demonstration_indices=demonstration_indices
        )

    def collate_fn(self, batch: "List[GSMItem]") -> "dict[str, Union[List[GSMItem], BatchEncoding]]":
        '''
        Coalesces and formats list of GSM items for model prediction

        Args:
            batch (List[GSMItem]):
                Subset of GSM dataset
        '''
        # call QuestionAnswerItem.format() to convert GSMItems to strings
        formatted_batch = [item.format(self.demonstrations, include_answer=False) for item in batch]

        # call self.tokenizer to convert string to input for model
        batch_encoding = self.tokenizer(formatted_batch,
                                        padding='longest',
                                        max_length=self.tokenizer.model_max_length,
                                        truncation=True,
                                        return_tensors='pt')
        return {
            "batch": batch,
            "formatted_batch": formatted_batch,
            "BatchEncoding": batch_encoding
        }


class GSMDataModule(QuestionAnswerDataModule):
    def __init__(self,
                 model_name: str,
                 batch_size: int,
                 data_directory: str,
                 entities_metadata_fpath: str,
                 num_demonstrations: int = -1,
                 max_demonstrations_token_length: int = 400,
                 demonstration_indices: "List[List[int]]" = None,
                 num_workers: int = 0,
                 prompt_augmentation: str = None,
                 entity_augmentation: str = None):

        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_demonstrations = num_demonstrations
        self.max_demonstrations_token_length = max_demonstrations_token_length
        self.demonstration_indices = demonstration_indices
        self.num_workers = num_workers
        self.prompt_augmentation = prompt_augmentation
        self.entity_augmentation = entity_augmentation
        self.data_directory = data_directory

        self.entities_dataframe = pd.read_csv(entities_metadata_fpath)

        self.datasets = {}

    def parse(self, fpath) -> "List[GSMItem]":
        # TODO add GSM to entities_metadata
        GSM_entities = self.entities_dataframe.entity.filter(dataset="GSM")
        GSM_entities = GSM_entities.entity.aggregate()

        with open(fpath) as f:
            lines = f.readlines()

        GSM_items = []
        for line in lines:
            question, answer = json.loads(line).values()
            question = re.sub(r"<<.*?>>", " ", question)
            answer = re.sub(r"<<.*?>>", " ", answer)

            question_entities = GSM_entities.entity.annotate(question.split(".")[-1])
            answer_entities = GSM_entities.entity.annotate(answer)

            GSM_item = GSMItem(
                context="",
                question=question,
                answer=answer,
                context_entities=[],
                question_entities=question_entities,
                answer_entities=answer_entities
            )

            GSM_items.append(GSM_item)

        return GSM_items

    def setup(self, stage=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # TODO create constant referring to fpath
        train_data = self.parse(os.path.join(self.data_directory, "train.jsonl"))

        self.datasets["train"] = GSMDataset(
            GSM_items=train_data,
            tokenizer=self.tokenizer,
            entities_dataframe=self.entities_dataframe,
            entity_augmentation=self.entity_augmentation,
            prompt_augmentation=self.prompt_augmentation,
            num_demonstrations=self.num_demonstrations,
            demonstration_indices=self.demonstration_indices
        )

        if stage in ("validate", None):
            validation_data = self.parse(os.path.join(self.data_directory, "validation.jsonl"))

            self.datasets["validation"] = GSMDataset(
                GSM_items=validation_data,
                tokenizer=self.tokenizer,
                entities_dataframe=self.entities_dataframe,
                entity_augmentation=self.entity_augmentation,
                prompt_augmentation=self.prompt_augmentation,
                num_demonstrations=self.num_demonstrations,
                demonstration_indices=self.demonstration_indices
            )

        if stage in ("test", None):
            test_data = self.parse(os.path.join(self.data_directory, "test.jsonl"))

            self.datasets["test"] = GSMDataset(
                GSM_items=test_data,
                tokenizer=self.tokenizer,
                entities_dataframe=self.entities_dataframe,
                entity_augmentation=self.entity_augmentation,
                prompt_augmentation=self.prompt_augmentation,
                num_demonstrations=self.num_demonstrations,
                demonstration_indices=self.demonstration_indices
            )

    @staticmethod
    def entity_statistics(data_directory, tokenizer_name):
        ner_model = spacy.load(NER_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        file_names = os.listdir(data_directory)

        for file_name in file_names:
            fpath = os.path.join(data_directory, file_name)
            with open(fpath, "r") as f:
                lines = f.readlines()

            gsm = map(lambda line: json.loads(line), lines)
            gsm_text = map(lambda sample: sample["question"] + " " + sample["answer"], gsm)
            gsm_clean_text = map(lambda text: re.sub(r"<<.*?>>|####", " ", text), gsm_text)
            gsm_whitespace_clean_text = map(lambda text: re.sub(r"\s+", " ", text), gsm_clean_text)

            gsm_entities = reduce(lambda entities, text: entities + list(ner_model(text).ents), gsm_whitespace_clean_text, [])
            gsm_entities = filter(lambda entity: entity.label_ in ACCEPTABLE_ENTITIES, gsm_entities)

        entity_occurrences = defaultdict(int)
        for entity in gsm_entities:
            entity_occurrences[(str(entity), entity.label_)] += 1

        entities_metadata = []
        for entity, occurrences in entity_occurrences.items():
            text, label = entity
            length = len(text)
            token_length = len(tokenizer.encode(text, add_special_tokens=False))
            entity_metadata = Entity(
                text=text,
                occurrences=occurrences,
                label=label,
                model=NER_MODEL_NAME,
                tokenizer=tokenizer_name,
                length=length,
                token_length=token_length,
                dataset="GSM"
            )

            entities_metadata.append(entity_metadata)

        return entities_metadata
