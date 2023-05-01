import wandb
from logger import PROJECT_NAME, timestamp
from logger import QuestionAnswerLogger, calculate_accuracy, calculate_f1

NUM_SAMPLES = 100


class bAbILogger(QuestionAnswerLogger):
    def __init__(self, file_name):
        self.file_name = file_name

    def setup(self, trainer, pl_module, stage):
        if stage == "validate":
            stage = "validation"

        datamodule = trainer.datamodule

        run_name = ""
        if datamodule.entity_augmentation is not None:
            run_name += f"{datamodule.entity_augmentation}_"
        if datamodule.prompt_augmentation is not None:
            run_name += f"{datamodule.prompt_augmentation}_"
        run_name += timestamp()

        self.run = {
            "name": run_name,
            "dataset": "bAbI",
            "model_name": datamodule.model_name,
            "batch_size": datamodule.batch_size,
            "tasks": datamodule.tasks,
            "data_directory": datamodule.data_directory,
            "num_demonstrations": datamodule.num_demonstrations,
            "demonstration_indices": datamodule.demonstration_indices,
            "num_workers": datamodule.num_workers,
            "prompt_augmentation": datamodule.prompt_augmentation,
            "entity_augmentation": datamodule.entity_augmentation
        }

        wandb.init(project=PROJECT_NAME, name=run_name, config=self.run)
        self.run["id"] = wandb.run.id

        self.outputs = {}
        self.task_prompt_perplexity = {}
        self.task_prediction_perplexity = {}
        self.task_target_perplexity = {}
        tasks = datamodule.datasets[stage]
        self.outputs[stage] = [[] for _ in range(len(tasks))]
        self.task_prompt_perplexity[stage] = [0 for _ in range(len(tasks))]
        self.task_prediction_perplexity[stage] = [0 for _ in range(len(tasks))]
        self.task_target_perplexity[stage] = [0 for _ in range(len(tasks))]
        self.tmp = None

    def on_validation_batch_end(self,
                                trainer,
                                pl_module,
                                outputs,
                                batch,
                                batch_index,
                                dataloader_index):

        bAbI_items, prompt_perplexity, prediction_perplexity, target_perplexity = outputs
        self.outputs["validation"][dataloader_index] += bAbI_items
        self.task_prompt_perplexity["validation"][dataloader_index] += sum(prompt_perplexity)
        self.task_prediction_perplexity["validation"][dataloader_index] += sum(prediction_perplexity)
        self.task_target_perplexity["validation"][dataloader_index] += sum(target_perplexity)

    def on_test_batch_end(self,
                          trainer,
                          pl_module,
                          outputs,
                          batch,
                          batch_index,
                          dataloader_index):

        self.outputs["test"][dataloader_index] += outputs

    def on_validation_end(self, trainer, pl_module):
        for dataloader_index, bAbI_items in enumerate(self.outputs["validation"]):
            if len(bAbI_items) == 0:
                continue

            n = len(bAbI_items)

            task_index = trainer.datamodule.datasets["validation"][dataloader_index].task
            accuracy = calculate_accuracy(bAbI_items)
            f1 = calculate_f1(bAbI_items)

            prompt_perplexity = self.task_prompt_perplexity["validation"][dataloader_index] / n
            prediction_perplexity = self.task_prediction_perplexity["validation"][dataloader_index] / n
            target_perplexity = self.task_target_perplexity["validation"][dataloader_index] / n

            recorded_outputs = [item.logging() for item in bAbI_items[:NUM_SAMPLES]]

            demonstrations = wandb.Table(
                data=[[trainer.datamodule.datasets["train"][dataloader_index].demonstrations]],
                columns=["demonstrations"]
            )
            outputs_table = wandb.Table(
                data=recorded_outputs,
                columns=["context", "question", "answer", "predictions"]
            )

            wandb.log({f"validation/task{task_index}/accuracy": accuracy})
            wandb.log({f"validation/task{task_index}/f1": f1})
            wandb.log({f"validation/task{task_index}/prompt_perplexity": prompt_perplexity})
            wandb.log({f"validation/task{task_index}/prediction_perplexity": prediction_perplexity})
            wandb.log({f"validation/task{task_index}/target_perplexity": target_perplexity})
            wandb.log({f"validation/task{task_index}/demonstrations": demonstrations})
            wandb.log({f"validation/task{task_index}/data": outputs_table})

            self.run[f"task{task_index}"] = {
                "accuracy": accuracy,
                "f1": f1,
                "prompt_perplexity": prompt_perplexity,
                "prediction_perplexity": prediction_perplexity,
                "target_perplexity": target_perplexity,
            }

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
