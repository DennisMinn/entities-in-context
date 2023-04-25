import pytorch_lightning as pl
from models import QuestionAnswerModel
from data_modules.bAbI import bAbIDataModule
from logger.bAbI import bAbILogger
import pandas as pd
import argparse
import os

SELECTED_INDICES =  {1: [25, 175, 225, 40, 61, 82, 280, 75, 340, 735, 406, 337, 20, 385, 125, 380, 666, 462],
                     12: [0, 5, 225, 400, 41, 47, 535, 365, 175, 820, 516, 127, 650, 95, 305, 545, 691, 432],
                     14: [0, 180, 95, 205, 46, 62, 715, 665, 130, 735, 345, 466, 60, 355, 890, 441, 175, 117],
                     16: [0, 1, 40, 9, 10, 151, 597, 480, 583, 213, 856, 580, 795, 384, 735, 593, 405, 115]
                    }

if __name__ == "__main__":
    
    os.system("wandb offline")
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-s", "--start", type=int, help="start index")
    argParser.add_argument("-e", "--end", type=int, help="end index")
    argParser.add_argument("-f", "--file_name", help="file name")
    args = argParser.parse_args()

    model = QuestionAnswerModel("google/flan-t5-base")
    
    df = pd.read_csv("data/entities_metadata.csv")

    df = df[df['dataset'] == 'demographics']
    df = df.sort_values(by=['occurrences'], ignore_index=True)
    entity_names = df['text']
    
    prompt_augmentations = ['demonstrations', 'both', 'query']

    model = QuestionAnswerModel("google/flan-t5-base")

    for prompt_augmentation in prompt_augmentations:
        for index in range(args.start, args.end):
            entity_name = entity_names[index]
            datamodule = bAbIDataModule(
                    model_name="google/flan-t5-base",
                    batch_size=5,
                    demonstration_indices=list(SELECTED_INDICES.values()),
                    num_demonstrations=5,
                    data_directory="data/bAbI tasks_1-20_v1-2/en-valid",
                    entities_metadata_fpath="data/entities_metadata.csv",
                    entity_augmentation=f"demographics_{entity_name}",
                    prompt_augmentation=prompt_augmentation,
                    tasks=list(SELECTED_INDICES.keys())
            )

            datamodule.setup()

            logger = bAbILogger(args.file_name)

            trainer = pl.Trainer(max_epochs=1, callbacks=[logger], accelerator="auto")
            trainer.validate(model=model, datamodule=datamodule)
    os.system("wandb sync --sync-all")
