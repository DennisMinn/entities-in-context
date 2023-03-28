from datetime import datetime
from functools import reduce
from pytorch_lightning.callbacks import Callback
import wandb

PROJECT_NAME = "Entities In Context"
index2stage = {0: "train", 1: "validation", 2: "test"}


def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def calculate_accuracy(data):
    accuracy = reduce(
        lambda accuracy, question_answer_item: accuracy + (question_answer_item.prediction == question_answer_item.answer),
        data, 0
    )

    accuracy /= len(data)
    return accuracy


class QuestionAnswerLogger(Callback):
    def setup(self, trainer, pl_module, stage):
        return
        self.run = wandb.init(project=PROJECT_NAME, name=timestamp())

    def on_validation_batch_end(self,
                                trainer,
                                pl_module,
                                outputs,
                                batch,
                                batch_idx,
                                dataloader_idx):

        print(batch_idx, dataloader_idx)
        stage = index2stage[dataloader_idx]
        data = trainer.datamodule.data[stage]
        batch_size = trainer.datamodule.batch_size

        start, end = batch_idx * batch_size, (batch_idx + 1) * batch_size
        data[start:end] = outputs

    def on_validation_end(self, trainer, pl_module):
        return
        # Logging Demonstrations
        demonstrations = wandb.Table(
            data=[[trainer.datamodule.datasets["train"].demonstrations]],
            columns=["demonstrations"]
        )
        wandb.log({"demonstrations": demonstrations})

        for stage in ["train", "validation", "test"]:
            # Logging Accuracy
            data = trainer.datamodule.data[stage]
            accuracy = calculate_accuracy(data)
            wandb.log({f"{stage}/accuracy": accuracy})

            if stage != "validation":
                continue

            # Logging Data
            data = wandb.Table(
                data=[[item.context, item.question, item.answer, item.prediction] for item in data],
                columns=["context", "question", "answer", "predictions"]
            )
            wandb.log({f"{stage}/data": data})

    def teardown(self, trainer, pl_module, stage):
        wandb.finish()
