import pytorch_lightning as pl
from data_modules import *
from data_modules.bAbI import *

from logger import *
from models import *

if __name__ == "__main__":

    model = QuestionAnswerModel("google/flan-t5-small")
    logger = QuestionAnswerLogger()

    trainer = pl.Trainer(max_epochs=1, callbacks=[logger], accelerator="auto")

    for i in range(1, 21, 1):
        datamodule = bAbIDataModule(
            model_name="google/flan-t5-small",
            batch_size=2,
            num_demonstrations=5,
            train_path=f"data/bAbI tasks_1-20_v1-2/en-valid/qa{i}_train.txt",
            validation_path=f"data/bAbI tasks_1-20_v1-2/en-valid/qa{i}_valid.txt",
            test_path=f"data/bAbI tasks_1-20_v1-2/en-valid/qa{i}_test.txt"
        )

        datamodule.setup()

        trainer.datamodule = datamodule

        train_dataloader = trainer.datamodule.train_dataloader()
        validation_dataloader = trainer.datamodule.val_dataloader()
        test_dataloader = trainer.datamodule.test_dataloader()

        trainer.validate(model=model, dataloaders=[train_dataloader, validation_dataloader, test_dataloader])
