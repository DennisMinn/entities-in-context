import wandb
from logger import QuestionAnswerLogger

NUM_SAMPLES = 100


class GSMLogger(QuestionAnswerLogger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self, trainer, pl_module, stage):
        self.qa_items = {
            "train": [],
            "validation": [],
            "test": []
        }

    def on_validation_batch_end(self,
                                trainer,
                                pl_module,
                                outputs,
                                batch,
                                batch_index,
                                dataset_index=0):

        self.qa_items["validation"] += outputs

    def on_validation_end(self, trainer, pl_module):
        import os
        qa_items = self.qa_items["validation"]
        dataset = trainer.datamodule.datasets["validation"]

        assert len(qa_items) == len(dataset)
        dataset.qa_items = self.qa_items["validation"]

        accuracy = dataset.calculate_accuracy()
        f1 = dataset.calculate_f1()

        perplexity = dataset.calculate_perplexity()
        valid_predictions = dataset.calculate_valid_predictions()

        demonstrations = wandb.Table(
            data=[[trainer.datamodule.datasets["validation"].demonstrations]],
            columns=["demonstrations"]
        )

        data_table = wandb.Table(
            data=[item.logging() for item in qa_items[:NUM_SAMPLES]],
            columns=["context", "question", "answer", "prediction", "numeric_answer", "numeric_prediction"]
        )

        wandb.log({"validation/accuracy": accuracy})
        wandb.log({"validation/f1": f1})
        wandb.log({"validation/perplexity": perplexity})
        wandb.log({"validation/valid_predictions": valid_predictions})
        wandb.log({"validation/demonstrations": demonstrations})
        wandb.log({"validation/data": data_table})

        self.run["validation"] = {
            "accuracy": accuracy,
            "f1": f1,
            "perplexity": perplexity,
        }

        directory, _ = os.path.split(self.output_fpath)
        file_name = f"{self.run['name']}_validation.jsonl"
        fpath = os.path.join(directory, file_name)
        dataset.export(fpath)
