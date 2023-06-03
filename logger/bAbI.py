import wandb
from logger import QuestionAnswerLogger

NUM_SAMPLES = 100


class bAbILogger(QuestionAnswerLogger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self, trainer, pl_module, stage):
        num_tasks = len(trainer.datamodule.tasks)
        self.qa_items = {
            "train": [[] for _ in range(num_tasks)],
            "validation": [[] for _ in range(num_tasks)],
            "test": [[] for _ in range(num_tasks)]
        }

    def on_validation_batch_end(self,
                                trainer,
                                pl_module,
                                outputs,
                                batch,
                                batch_index,
                                dataset_index=0):

        self.qa_items["validation"][dataset_index] += outputs

    def on_validation_end(self, trainer, pl_module):
        import os
        for dataset_index, qa_items in enumerate(self.qa_items["validation"]):
            dataset = trainer.datamodule.datasets["validation"][dataset_index]
            task = dataset.task

            assert len(dataset) == len(qa_items)
            dataset.qa_items = qa_items

            accuracy = dataset.calculate_accuracy()
            f1 = dataset.calculate_f1()

            perplexity = dataset.calculate_perplexity()

            demonstrations = wandb.Table(
                data=[[trainer.datamodule.datasets["train"][dataset_index].demonstrations]],
                columns=["demonstrations"]
            )

            data_table = wandb.Table(
                data=[item.logging() for item in qa_items[:NUM_SAMPLES]],
                columns=["context", "question", "answer", "predictions"]
            )

            wandb.log({f"validation/task{task}/accuracy": accuracy})
            wandb.log({f"validation/task{task}/f1": f1})
            wandb.log({f"validation/task{task}/perplexity": perplexity})
            wandb.log({f"validation/task{task}/demonstrations": demonstrations})
            wandb.log({f"validation/task{task}/data": data_table})

            self.run[f"task{task}"] = {
                "accuracy": accuracy,
                "f1": f1,
                "perplexity": perplexity,
            }

            directory, _ = os.path.split(self.output_fpath)
            file_name = f"{self.run['name']}_task{task}_validation.jsonl"
            fpath = os.path.join(directory, file_name)
            dataset.export(fpath)
