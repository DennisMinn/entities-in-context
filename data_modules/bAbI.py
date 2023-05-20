from typing import List, TYPE_CHECKING
import os
from data_modules import QuestionAnswerItem, QuestionAnswerDataset, QuestionAnswerDataModule


if TYPE_CHECKING:
    from typing import Union
    from torch import BatchEncoding

CONTEXT_START = "1"


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
        self.dataset_name = "bAbI"
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
            "batch_encoding": batch_encoding
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
        from tqdm.auto import tqdm
        from models import NERModel

        try:
            bAbI_items = self._load_pickle(fpath)
            return bAbI_items
        except FileNotFoundError:
            pass

        with open(fpath) as f:
            lines = f.readlines()

        bAbI_items = []
        model = NERModel()

        context = ''
        context_entities = set()

        for line in tqdm(lines, desc=f"Annotating {fpath}"):
            line_id, text = line.split(' ', 1)
            if line_id == CONTEXT_START:
                context = ''
                context_entities = set()

            text = text.strip()
            if '\t' in text:
                question, answer, _ = text.split('\t')
                question_entities = model.annotate_entities(question)
                answer_entities = model.annotate_entities(answer)

                bAbI_item = bAbIItem(
                    context=context,
                    question=question,
                    answer=answer,
                    context_entities=list(context_entities),
                    question_entities=list(question_entities),
                    answer_entities=list(answer_entities)
                )

                bAbI_items.append(bAbI_item)

            else:
                context = context + " " + text if context else text
                context_entities = context_entities.union(model.annotate_entities(text))

        self._save_pickle(bAbI_items, fpath)
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
                entity_augmentation=self.entity_augmentation,
                prompt_augmentation=self.prompt_augmentation,
                num_demonstrations=num_demonstrations,
                max_demonstrations_token_length=self.max_demonstrations_token_length,
                demonstration_indices=demonstration_indices
            )

            datasets.append(dataset)

        return datasets

    def setup(self, stage=None):
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
