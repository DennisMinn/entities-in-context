from typing import List, TYPE_CHECKING
import os
from collections import defaultdict
import spacy
from transformers import AutoTokenizer
from data_modules import QuestionAnswerItem, QuestionAnswerDataset, QuestionAnswerDataModule
from data_modules.entities import NER_MODEL_NAME, Entity
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from typing import Union
    from torch import BatchEncoding

CONTEXT_START = "1"
NUM_TASKS = 20


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
    def __init__(self, task: int, **kwargs):
        super().__init__(**kwargs)
        self.task = task

    def collate_fn(self, batch: "List[bAbIItem]") -> "dict[str, Union[List[bAbIItem], BatchEncoding]]":
        '''
        Coalesces and formats list of bAbI items for model prediction

        Args:
            batch (List[bAbIItem]):
                Subset of bAbI dataset
        '''
        # call QuestionAnswerItem.format() to convert bAbIItems to strings
        formatted_batch = [item.format(self.demonstrations, include_answer=False) for item in batch]

        # call self.tokenizer to convert string to input for model
        batch_encoding = self.tokenizer(formatted_batch,
                                        padding='longest',
                                        max_length=self.tokenizer.model_max_length,
                                        truncation=True,
                                        return_tensors='pt')
        return {
            "task": self.task,
            "batch": batch,
            "demonstrations": self.demonstrations,
            "BatchEncoding": batch_encoding
        }


class bAbIDataModule(QuestionAnswerDataModule):
    def __init__(self, tasks: "List[int]", **kwargs):
        super().__init__(**kwargs)
        self.tasks = tasks

        assert (self.demonstration_indices is None
                or len(self.tasks) == len(self.demonstration_indices))

        assert (self.demonstration_indices is None
                or isinstance(self.demonstration_indices, List))

    def parse(self, fpath) -> "List[bAbIItem]":
        bAbI_entities = self.entities_dataframe.entity.filter(dataset="bAbI")
        bAbI_entities = bAbI_entities.entity.aggregate()

        with open(fpath) as f:
            lines = f.readlines()

        bAbI_items = []

        context = ''
        context_entities = set()

        for line in lines:
            line_id, text = line.split(' ', 1)
            if line_id == CONTEXT_START:
                context = ''
                context_entities = set()

            if '\t' in text:
                question, answer, _ = text.split('\t')
                question_entities = bAbI_entities.entity.annotate(question)
                answer_entities = bAbI_entities.entity.annotate(answer)

                bAbI_item = bAbIItem(
                    context=context,
                    question=question,
                    answer=answer,
                    context_entities=context_entities,
                    question_entities=question_entities,
                    answer_entities=answer_entities
                )

                bAbI_items.append(bAbI_item)

            else:
                context += text
                context_entities = context_entities.union(bAbI_entities.entity.annotate(text))

        return bAbI_items

    def load_tasks(self, stage=None):
        datasets = []
        for dataset_index, task_index in enumerate(self.tasks):
            task_path = os.path.join(self.data_directory, f"qa{task_index}_{stage}.txt")
            data = self.parse(task_path)

            if stage == "train":
                num_demonstrations = self.num_demonstrations
                demonstration_indices = self.demonstration_indices[dataset_index] if self.demonstration_indices else None
            else:
                num_demonstrations = -1
                demonstration_indices = None

            dataset = bAbIDataset(
                qa_items=data,
                task=task_index,
                tokenizer=self.tokenizer,
                entities_dataframe=self.entities_dataframe,
                entity_augmentation=self.entity_augmentation,
                prompt_augmentation=self.prompt_augmentation,
                num_demonstrations=num_demonstrations,
                max_demonstrations_token_length=self.max_demonstrations_token_length,
                demonstration_indices=demonstration_indices
            )

            datasets.append(dataset)

        return datasets

    def setup(self, stage=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.datasets = {}
        self.datasets["train"] = self.load_tasks("train")

        if stage in ("validate", None):
            self.datasets["validation"] = self.load_tasks("valid")

            for dataset_index in range(len(self.datasets["train"])):
                self.datasets["validation"][dataset_index].demonstrations = self.datasets["train"][dataset_index].demonstrations

        if stage in ("test", None):
            self.datasets["test"] = self.load_tasks("test")

            for dataset_index in range(len(self.datasets["train"])):
                self.datasets["test"][dataset_index].demonstrations = self.datasets["train"][dataset_index].demonstrations

    @staticmethod
    def entity_statistics(data_dir, tokenizer_name):
        # Log all entity occurrences
        ner_model = spacy.load(NER_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        entity_occurrences = defaultdict(int)

        file_names = os.listdir(data_dir)
        tasks_progress_bar = tqdm(total=len(file_names), desc="Gathering bAbI entity statistics")

        for file_name in file_names:
            fpath = os.path.join(data_dir, file_name)
            with open(fpath, "r") as f:
                lines = f.readlines()

            for line in lines:
                if "\t" in line:
                    continue

                line_id, text = line.split(" ", 1)
                annotations = ner_model(text)

                for entity in annotations.ents:
                    text, label = str(entity), entity.label_
                    entity_occurrences[(text, label)] += 1

            tasks_progress_bar.update(1)

        tasks_progress_bar.close()

        # Create entity metadata
        entities_metadata = []
        for entity, occurrences in entity_occurrences.items():
            text, label = entity
            length = len(text)
            token_length = len(tokenizer(text).tokens())
            entity_metadata = Entity(
                text=text,
                occurrences=occurrences,
                label=label,
                model=NER_MODEL_NAME,
                tokenizer=tokenizer_name,
                length=length,
                token_length=token_length,
                dataset="bAbI"
            )

            entities_metadata.append(entity_metadata)

        return entities_metadata
