from datetime import datetime
from functools import reduce
from pytorch_lightning.callbacks import Callback
import wandb

PROJECT_NAME = "Entities In Context"


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
        self.run = wandb.init(project=PROJECT_NAME, name=timestamp())

    def teardown(self, trainer, pl_module, stage):
        wandb.finish()
