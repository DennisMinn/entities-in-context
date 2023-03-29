import wandb
from itertools import repeat
from data_modules.bAbI import NUM_TASKS
from logger import QuestionAnswerLogger, calculate_accuracy

NUM_SAMPLES = 100


class bAbILogger(QuestionAnswerLogger):
    def __init__(self):
        super().__init__()
        self.outputs = {}
        for stage in ["train", "validation", "test"]:
            self.outputs[stage] = list(repeat([], NUM_TASKS))

    def on_validation_batch_end(self,
                                trainer,
                                pl_module,
                                outputs,
                                batch,
                                batch_index,
                                task_index):

        self.outputs["validation"][task_index] += outputs

    def on_test_batch_end(self,
                          trainer,
                          pl_module,
                          outputs,
                          batch,
                          batch_index,
                          task_index):

        self.outputs["test"][task_index] += outputs

    def on_validation_end(self, trainer, pl_module):
        for task_index, task_outputs in enumerate(self.outputs["validation"]):
            accuracy = calculate_accuracy(task_outputs)
            wandb.log({f"validation/task{task_index+1}/accuracy": accuracy})

            # Logging Demonstrations
            demonstrations = wandb.Table(
                data=[[trainer.datamodule.datasets["train"][task_index].demonstrations]],
                columns=["demonstrations"]
            )
            wandb.log({f"validation/task{task_index+1}/demonstrations": demonstrations})

            # Logging Data
            recorded_outputs = [item.logging() for item in task_outputs[:NUM_SAMPLES]]
            outputs_table = wandb.Table(
                data=recorded_outputs,
                columns=["context", "question", "answer", "predictions"]
            )
            wandb.log({f"validation/task{task_index+1}/data": outputs_table})

    def on_test_end(self, trainer, pl_module):
        for task_index, task_outputs in enumerate(self.outputs["test"]):
            accuracy = calculate_accuracy(task_outputs)
            wandb.log({f"test/task{task_index+1}/accuracy": accuracy})

            # Logging Demonstrations
            demonstrations = wandb.Table(
                data=[[trainer.datamodule.datasets["train"][task_index].demonstrations]],
                columns=["demonstrations"]
            )
            wandb.log({f"test/task{task_index+1}/demonstrations": demonstrations})

            # Logging Data
            recorded_outputs = [item.logging() for item in task_outputs[:NUM_SAMPLES]]
            outputs_table = wandb.Table(
                data=recorded_outputs,
                columns=["context", "question", "answer", "predictions"]
            )
            wandb.log({f"test/task{task_index+1}/data": outputs_table})
