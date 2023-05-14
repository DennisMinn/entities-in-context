from typing import TYPE_CHECKING
import os
import re
from collections import defaultdict
from functools import reduce
import spacy
import json
from transformers import AutoTokenizer
from data_modules import QuestionAnswerItem, QuestionAnswerDataset, QuestionAnswerDataModule
from data_modules.entities import NER_MODEL_NAME, Entity
from data_modules.constants import QUESTION, NEXT_LINE, ANSWER

if TYPE_CHECKING:
    from typing import List, Union
    from torch import BatchEncoding

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

    def calculate_accuracy(self):
        answer = self.get_answer()
        assert answer != INVALID_ANSWER

        prediction = self.get_prediction()
        return answer == prediction

    def format(self, demonstrations: str = "", include_answer: bool = True) -> str:
        if include_answer:
            query = QUESTION + self.question + NEXT_LINE + ANSWER + self.answer + NEXT_LINE
        else:
            query = QUESTION + self.question + NEXT_LINE + ANSWER

        return demonstrations + query

    def logging(self):
        return [
            self.question,
            self.answer,
            self.prediction,
            self.get_answer(),
            self.get_prediction()
        ]


class GSMDataset(QuestionAnswerDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calculate_valid_predictions(self):
        valid_predictions = reduce(lambda total, item: total + (item.get_prediction() != INVALID_ANSWER), self.qa_items, 0)
        valid_predictions /= len(self.qa_items)
        return valid_predictions

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
            "demonstrations": self.demonstrations,
            "BatchEncoding": batch_encoding
        }


class GSMDataModule(QuestionAnswerDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def parse(self, fpath) -> "List[GSMItem]":
        GSM_entities = self.entities_dataframe.entity.filter(dataset="GSM")
        GSM_entities = GSM_entities.entity.aggregate()

        with open(fpath) as f:
            lines = f.readlines()

        GSM_items = []
        for line in lines:
            question, answer = json.loads(line).values()

            question = re.sub(r"<<.*?>>", " ", question)
            question = re.sub(r"\s+", " ", question)
            answer = re.sub(r"<<.*?>>", " ", answer)
            answer = re.sub(r"\s+", " ", answer)

            question_entities = GSM_entities.entity.annotate(question)
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

        train_data = self.parse(os.path.join(self.data_directory, "train.jsonl"))

        self.datasets["train"] = GSMDataset(
            qa_items=train_data,
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
                qa_items=validation_data,
                tokenizer=self.tokenizer,
                entities_dataframe=self.entities_dataframe,
                entity_augmentation=self.entity_augmentation,
                prompt_augmentation=self.prompt_augmentation,
                num_demonstrations=-1,
                demonstration_indices=None
            )
            self.datasets["validation"].demonstrations = self.datasets["train"].demonstrations

        if stage in ("test", None):
            test_data = self.parse(os.path.join(self.data_directory, "test.jsonl"))

            self.datasets["test"] = GSMDataset(
                qa_items=test_data,
                tokenizer=self.tokenizer,
                entities_dataframe=self.entities_dataframe,
                entity_augmentation=self.entity_augmentation,
                prompt_augmentation=self.prompt_augmentation,
                num_demonstrations=-1,
                demonstration_indices=None
            )
            self.datasets["test"].demonstrations = self.datasets["train"].demonstrations

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
