import wandb
from logger import PROJECT_NAME, timestamp
from logger import QuestionAnswerLogger, calculate_accuracy, calculate_f1

NUM_SAMPLES = 100


class bAbILogger(QuestionAnswerLogger):
    def setup(self, trainer, pl_module, stage):
        if stage == "validate":
            stage = "validation"

        self.run = wandb.init(project=PROJECT_NAME, name=timestamp())

        self.outputs = {}
        tasks = trainer.datamodule.datasets[stage]
        self.outputs[stage] = [[] for _ in range(len(tasks))]

    def on_validation_batch_end(self,
                                trainer,
                                pl_module,
                                outputs,
                                batch,
                                batch_index,
                                dataloader_index):

        self.outputs["validation"][dataloader_index] += outputs

    def on_test_batch_end(self,
                          trainer,
                          pl_module,
                          outputs,
                          batch,
                          batch_index,
                          dataloader_index):

        self.outputs["test"][dataloader_index] += outputs

    def on_validation_end(self, trainer, pl_module):
        for dataloader_index, task_outputs in enumerate(self.outputs["validation"]):
            if len(task_outputs) == 0:
                continue

            task_index = trainer.datamodule.datasets["validation"][dataloader_index].task

            accuracy = calculate_accuracy(task_outputs)
            wandb.log({f"validation/task{task_index}/accuracy": accuracy})

            f1 = calculate_f1(task_outputs)
            wandb.log({f"validation/task{task_index}/f1": f1})

            # Logging Demonstrations
            demonstrations = wandb.Table(
                data=[[trainer.datamodule.datasets["train"][dataloader_index].demonstrations]],
                columns=["demonstrations"]
            )
            wandb.log({f"validation/task{task_index}/demonstrations": demonstrations})

            # Logging Data
            recorded_outputs = [item.logging() for item in task_outputs[:NUM_SAMPLES]]
            outputs_table = wandb.Table(
                data=recorded_outputs,
                columns=["context", "question", "answer", "predictions"]
            )
            wandb.log({f"validation/task{task_index}/data": outputs_table})

    def on_test_end(self, trainer, pl_module):
        for dataloader_index, task_outputs in enumerate(self.outputs["test"]):
            if len(task_outputs) == 0:
                continue

            task_index = trainer.datamodule.datasets["test"][dataloader_index].task

            accuracy = calculate_accuracy(task_outputs)
            wandb.log({f"test/task{task_index}/accuracy": accuracy})

            f1 = calculate_f1(task_outputs)
            wandb.log({f"test/task{task_index}/f1": f1})

            # Logging Demonstrations
            demonstrations = wandb.Table(
                data=[[trainer.datamodule.datasets["train"][dataloader_index].demonstrations]],
                columns=["demonstrations"]
            )
            wandb.log({f"test/task{task_index}/demonstrations": demonstrations})

            # Logging Data
            recorded_outputs = [item.logging() for item in task_outputs[:NUM_SAMPLES]]
            outputs_table = wandb.Table(
                data=recorded_outputs,
                columns=["context", "question", "answer", "predictions"]
            )
            wandb.log({f"test/task{task_index}/data": outputs_table})
