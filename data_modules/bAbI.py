from typing import TYPE_CHECKING
import os
import pandas as pd
from collections import defaultdict
import spacy
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from data_modules import QuestionAnswerItem, QuestionAnswerDataset, QuestionAnswerDataModule
from data_modules.entities import NER_MODEL_NAME, Entity
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from typing import List, Union
    from pandas import DataFrame
    from torch import BatchEncoding
    from transformers import PreTrainedTokenizerFast

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
    def __init__(self,
                 bAbI_items: "List[bAbIItem]",
                 tokenizer: "PreTrainedTokenizerFast",
                 task: int,
                 entities_dataframe: "DataFrame" = None,
                 entity_augmentation: str = None,
                 prompt_augmentation: str = None,
                 num_demonstrations: int = -1,
                 demonstration_indices: "List[int]" = None):

        self.num_demonstrations = num_demonstrations
        self.demonstration_indices = demonstration_indices
        self.demonstrations = self.initialize_demonstrations(bAbI_items)
        self.bAbI_items = bAbI_items
        self.tokenizer = tokenizer
        self.entities_dataframe = entities_dataframe
        self.entity_augmentation = entity_augmentation
        self.replacement_entity = self.initialize_replacement_entity()
        self.prompt_augmentation = prompt_augmentation
        self.task = task

    def __getitem__(self, index: int) -> bAbIItem:
        if self.entity_augmentation is not None:
            self.bAbI_items[index].replace_entity(self.replacement_entity)

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
        batch_encoding = self.tokenizer(formatted_batch,
                                        padding='longest',
                                        max_length=512,
                                        truncation=True,
                                        return_tensors='pt')
        return {
          "task": self.task,
          "batch": batch,
          "BatchEncoding": batch_encoding
        }


class bAbIDataModule(QuestionAnswerDataModule):
    def __init__(self,
                 model_name: str,
                 batch_size: int,
                 tasks: "List[int]",
                 data_directory: str,
                 entities_metadata_fpath: str,
                 num_demonstrations: int = -1,
                 demonstration_indices: "List[List[int]]" = None,
                 num_workers: int = 0,
                 prompt_augmentation: str = None,
                 entity_augmentation: str = None):

        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_demonstrations = num_demonstrations
        self.demonstration_indices = demonstration_indices
        self.num_workers = num_workers
        self.prompt_augmentation = prompt_augmentation
        self.entity_augmentation = entity_augmentation
        self.data_directory = data_directory
        self.tasks = tasks

        self.entities_dataframe = pd.read_csv(entities_metadata_fpath)

        self.datasets = {}
        self.datasets["train"] = []
        self.datasets["validation"] = []
        self.datasets["test"] = []

    def parse(self, fpath) -> "List[bAbIItem]":
        # TODO Optimize parse
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
        if stage == "validation":
            stage = "valid"

        datasets = []
        for dataset_index, task_index in enumerate(self.tasks):
            task_path = os.path.join(self.data_directory, f"qa{task_index}_{stage}.txt")
            data = self.parse(task_path)

            if self.demonstration_indices is not None:
                demonstration_indices = self.demonstration_indices[dataset_index]
            else:
                demonstration_indices = None

            dataset = bAbIDataset(
                bAbI_items=data,
                task=task_index,
                tokenizer=self.tokenizer,
                entities_dataframe=self.entities_dataframe,
                entity_augmentation=self.entity_augmentation,
                prompt_augmentation=self.prompt_augmentation,
                num_demonstrations=self.num_demonstrations,
                demonstration_indices=demonstration_indices
            )

            datasets.append(dataset)

        return datasets

    def setup(self, stage=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.datasets = {}
        self.datasets["train"] = self.load_tasks("train")

        if stage in ("validate", None):
            self.datasets["validation"] = self.load_tasks("validation")

            for task_index in range(len(self.tasks)):
                self.datasets["validation"][task_index].demonstrations = self.datasets["train"][task_index].demonstrations

        if stage in ("test", None):
            self.datasets["test"] = self.load_tasks("test")

            for task_index in range(len(self.tasks)):
                self.datasets["test"][task_index].demonstrations = self.datasets["train"][task_index].demonstrations

    def train_dataloader(self):
        dataloaders = []
        for task in self.datasets["train"]:
            dataloaders.append(DataLoader(
                task,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
                collate_fn=task.collate_fn
            ))

        return dataloaders

    def val_dataloader(self):
        dataloaders = []
        for task in self.datasets["validation"]:
            dataloaders.append(DataLoader(
                task,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
                collate_fn=task.collate_fn
            ))

        return dataloaders

    def test_dataloader(self):
        dataloaders = []
        for task in self.datasets["test"]:
            dataloaders.append(DataLoader(
                task,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
                collate_fn=task.collate_fn
            ))

        return dataloaders

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
