import pytorch_lightning as pl
from models import QuestionAnswerModel
from data_modules.bAbI import bAbIDataModule
from logger.bAbI import bAbILogger

if __name__ == "__main__":

    model = QuestionAnswerModel("google/flan-t5-small")

    datamodule = bAbIDataModule(
            model_name="google/flan-t5-small",
            batch_size=5,
            num_demonstrations=5,
            data_directory="data/bAbI tasks_1-20_v1-2/en-valid",
            entities_metadata_fpath="data/entities_metadata.csv",
            entity_augmentation="demographics_Michael",
            tasks=[1, 4, 5]
    )

    datamodule.setup()

    logger = bAbILogger()

    trainer = pl.Trainer(max_epochs=1, callbacks=[logger], accelerator="auto")
    trainer.validate(model=model, datamodule=datamodule)
