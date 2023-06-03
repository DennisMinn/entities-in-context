import json
from datetime import datetime
from pytorch_lightning.callbacks import Callback
import wandb

PROJECT_NAME = "entities-in-context"


def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


class QuestionAnswerLogger(Callback):
    def __init__(self, output_fpath, data_module):
        self.qa_items = {}
        self.output_fpath = output_fpath

        run_name = ""
        if data_module.entity_augmentation is not None:
            run_name += f"{data_module.entity_augmentation}_"
        if data_module.prompt_augmentation is not None:
            run_name += f"{data_module.prompt_augmentation}_"
        run_name += timestamp()

        self.run = {
            "name": run_name,
            "dataset": data_module.name,
            "model_name": data_module.model_name,
            "batch_size": data_module.batch_size,
            "data_directory": data_module.data_directory,
            "num_demonstrations": data_module.num_demonstrations,
            "demonstration_indices": data_module.demonstration_indices,
            "num_workers": data_module.num_workers,
            "prompt_augmentation": data_module.prompt_augmentation,
            "entity_augmentation": data_module.entity_augmentation
        }

        wandb.init(project=PROJECT_NAME, name=run_name, config=self.run)
        self.run["id"] = wandb.run.id

    def teardown(self, trainer, pl_module, stage):
        with open(self.output_fpath, "a") as f:
            f.write(json.dumps(self.run) + "\n")

        wandb.finish()
