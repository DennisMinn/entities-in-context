from typing import TYPE_CHECKING
import os
import re
from functools import reduce
import json
from data_modules import QuestionAnswerItem, QuestionAnswerDataset, QuestionAnswerDataModule

if TYPE_CHECKING:
    from typing import List, Union
    from torch import BatchEncoding

ANSWER_REGEX = PREDICTION_REGEX = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANSWER = INVALID_PREDICTION = "[invalid]"


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

    def logging(self):
        return [
            self.context,
            self.question,
            self.answer,
            self.prediction,
            self.get_answer(),
            self.get_prediction()
        ]

    @staticmethod
    def clean_text(text):
        text = re.sub(r"<<.*?>>", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"(\$) (\d+)", r"\1\2", text)
        return text


class GSMDataset(QuestionAnswerDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = "grade-school-math"

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
            "batch_encoding": batch_encoding
        }


class GSMDataModule(QuestionAnswerDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def parse(self, fpath) -> "List[GSMItem]":
        from tqdm.auto import tqdm
        from models import NERModel

        try:
            GSM_items = self._load_pickle(fpath)
            return GSM_items
        except FileNotFoundError:
            pass

        with open(fpath) as f:
            lines = f.readlines()

        GSM_items = []
        model = NERModel()

        for line in tqdm(lines, desc=f"Annotating {fpath}"):
            question, answer = json.loads(line).values()

            question = GSMItem.clean_text(question)
            answer = GSMItem.clean_text(answer)

            context, question = QuestionAnswerItem.separate_last_sentence(question)

            context_entities = list(model.annotate_entities(context))
            question_entities = list(model.annotate_entities(question))
            answer_entities = list(model.annotate_entities(answer))

            GSM_item = GSMItem(
                context=context,
                question=question,
                answer=answer,
                context_entities=context_entities,
                question_entities=question_entities,
                answer_entities=answer_entities
            )

            GSM_items.append(GSM_item)

        self._save_pickle(GSM_items, fpath)
        return GSM_items

    def setup(self, stage=None):
        train_data = self.parse(os.path.join(self.data_directory, "train.jsonl"))

        self.datasets["train"] = GSMDataset(
            qa_items=train_data,
            tokenizer=self.tokenizer,
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
                entity_augmentation=self.entity_augmentation,
                prompt_augmentation=self.prompt_augmentation,
                num_demonstrations=-1,
                demonstration_indices=None
            )
            self.datasets["test"].demonstrations = self.datasets["train"].demonstrations
