import pytorch_lightning as pl
from models import QuestionAnswerModel
from data_modules.GSM import GSMDataModule
from logger.GSM import GSMLogger
import os

if __name__ == "__main__":
    model = QuestionAnswerModel("google/flan-t5-base")

    for num_demonstrations in [3, 4, 5]:
        datamodule = GSMDataModule(
                model_name="google/flan-t5-base",
                batch_size=5,
                demonstration_indices=[1, 4, 70, 88, 94, 106, 126, 132, 149, 196],
                num_demonstrations=num_demonstrations,
                data_directory="data/gsm",
                entities_metadata_fpath="data/entities_metadata.csv",
                entity_augmentation=None,
                prompt_augmentation=None,
        )

        datamodule.setup()
        logger = GSMLogger("runs.json", datamodule)

        trainer = pl.Trainer(max_epochs=1, callbacks=[logger], accelerator="auto", logger=False)
        trainer.validate(model=model, datamodule=datamodule)

        break

