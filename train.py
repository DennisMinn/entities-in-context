import pytorch_lightning as pl
from models import QuestionAnswerModel
from data_modules.bAbI import bAbIDataModule
from logger.bAbI import bAbILogger

SELECTED_INDICES =  {1: [25, 175, 225, 40, 61, 82],
                     # 5: [0, 131, 20, 438, 200, 36, 42],
                     # 6: [0, 5, 25, 56, 82],
                     # 7: [5, 10, 85, 46, 52],
                     # 9: [0, 15, 20, 26, 67],
                     # 10: [0, 5, 75, 41, 57],
                     # 11: [100, 5, 50, 30, 36, 72],
                     12: [0, 5, 225, 400, 41, 47],
                     # 13: [0, 50, 25, 150, 31, 52],
                     14: [0, 180, 95, 205, 46, 62],
                     # 15: [0, 4, 8, 17, 26],
                     16: [0, 1, 40, 9, 10],
                     # 20: [0, 52, 32, 127, 13, 44, 59]
                    }

if __name__ == "__main__":

    model = QuestionAnswerModel("google/flan-t5-base")

    datamodule = bAbIDataModule(
            model_name="google/flan-t5-base",
            batch_size=5,
            demonstration_indices=list(SELECTED_INDICES.values()),
            data_directory="data/bAbI tasks_1-20_v1-2/en-valid",
            entities_metadata_fpath="data/entities_metadata.csv",
            entity_augmentation="demographics_Michael",
            prompt_augmentation="both",
            tasks=list(SELECTED_INDICES.keys())
    )

    datamodule.setup()

    logger = bAbILogger()

    trainer = pl.Trainer(max_epochs=1, callbacks=[logger], accelerator="auto")
    trainer.validate(model=model, datamodule=datamodule)
