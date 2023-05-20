import os
import json
from datetime import datetime
from pytorch_lightning.callbacks import Callback
import wandb

PROJECT_NAME = "Entities In Context"


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
        if not os.path.isfile(self.output_fpath):
            with open(self.output_fpath, "x") as f:
                run_list = list([])
        else:
            with open(self.output_fpath, "r") as f:
                run_list = list(json.load(f))
        run_list.append(self.run)

        with open(self.output_fpath, "w") as f:
            json.dump(run_list, f, indent=2, separators=(',', ': '))

        wandb.finish()
