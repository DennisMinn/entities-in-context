import pytorch_lightning as pl
from data_modules import *
from data_modules.bAbI import *

from logger import *
from models import *

SELECTED_INDICES = {1:[0, 30, 40, 61, 82],
                    2:[0, 10, 15, 21, 42],
                    3:[10, 15, 20, 35, 40],
                    4:[0, 1, 2, 7, 8],
                    5:[0, 10 ,20 ,36 ,42],
                    6:[0 ,5 ,25 ,56 ,82],
                    7:[5 ,10 ,15 ,46 ,52],
                    8:[5 ,20 ,25 ,36 ,42],
                    9:[0 ,15 ,20 ,26 ,67],
                   10:[0 ,5 ,15 ,41 ,57]}

if __name__ == "__main__":

    model = QuestionAnswerModel("google/flan-t5-small")
    logger = QuestionAnswerLogger()

    trainer = pl.Trainer(max_epochs=1, callbacks=[logger], accelerator="auto")

    for i in range(1, 11, 1):
        datamodule = bAbIDataModule(
            model_name="google/flan-t5-small",
            batch_size=5,
            demonstration_indices=SELECTED_INDICES[i],
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
