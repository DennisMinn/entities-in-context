import ast
import pytorch_lightning as pl
from models import QuestionAnswerModel
import argparse
import os


if __name__ == "__main__":
    os.system("wandb offline")
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_name", help="file name")
    parser.add_argument("--enable-wandb", type=bool, default=True)

    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--demonstration_indices", type=str)
    parser.add_argument("--num_demonstrations", type=int, default=-1)
    parser.add_argument("--data_directory", type=str, required=True)
    parser.add_argument("--entity_augmentation", type=str)
    parser.add_argument("--prompt_augmentation", type=str)
    parser.add_argument("--tasks", type=str)

    parser.add_argument("--max_new_tokens", type=int, required=True)

    args = parser.parse_args()

    callbacks = []
    if args.dataset_name == "bAbI":
        from data_modules.bAbI import bAbIDataModule
        from logger.bAbI import bAbILogger

        args.tasks = ast.literal_eval(args.tasks)
        args.demonstration_indices = ast.literal_eval(args.demonstration_indices)

        data_module = bAbIDataModule(
            model_name=args.model_name,
            batch_size=args.batch_size,
            data_directory=args.data_directory,
            tasks=args.tasks,
            demonstration_indices=args.demonstration_indices,
            num_demonstrations=args.num_demonstrations,
            entity_augmentation=args.entity_augmentation,
            prompt_augmentation=args.prompt_augmentation,
        )
        data_module.setup()

        if args.enable_wandb:
            callbacks.append(bAbILogger(output_fpath=args.file_name, data_module=data_module))

    elif args.dataset_name == "GSM":
        from data_modules.GSM import GSMDataModule
        from logger.GSM import GSMLogger

        args.demonstration_indices = ast.literal_eval(args.demonstration_indices)

        data_module = GSMDataModule(
            model_name=args.model_name,
            batch_size=args.batch_size,
            data_directory=args.data_directory,
            demonstration_indices=args.demonstration_indices,
            num_demonstrations=args.num_demonstrations,
            entity_augmentation=args.entity_augmentation,
            prompt_augmentation=args.prompt_augmentation,
        )

        if args.enable_wandb:
            callbacks.append(GSMLogger(output_fpath=args.file_name, data_module=data_module))

    model = QuestionAnswerModel(data_module, args.max_new_tokens)

    trainer = pl.Trainer(
        max_epochs=1,
        callbacks=callbacks,
        accelerator="auto",
        logger=False,
        num_sanity_val_steps=2
    )

    trainer.validate(model=model, datamodule=data_module)
