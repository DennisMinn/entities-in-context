import wandb
from logger import QuestionAnswerLogger

NUM_SAMPLES = 100


class GSMLogger(QuestionAnswerLogger):
    def setup(self, trainer, pl_module, stage):
        self.qa_items = {
            "train": [],
            "validation": [],
            "test": []
        }

    def on_validation_batch_end(self,
                                trainer,
                                pl_module,
                                outputs,
                                batch,
                                batch_index,
                                dataset_index=0):

        self.qa_items["validation"] += outputs

    def on_validation_end(self, trainer, pl_module):
        qa_items = self.qa_items["validation"]
        dataset = trainer.datamodule.datasets["validation"]

        assert len(qa_items) == len(dataset)
        dataset.qa_items = self.qa_items["validation"]

        accuracy = dataset.calculate_accuracy()
        f1 = dataset.calculate_f1()

        prompt_ppl, answer_ppl, prediction_ppl = dataset.calculate_perplexities()
        valid_predictions = dataset.calculate_valid_predictions()

        demonstrations = wandb.Table(
            data=[[trainer.datamodule.datasets["validation"].demonstrations]],
            columns=["demonstrations"]
        )

        data_table = wandb.Table(
            data=[item.logging() for item in qa_items[:NUM_SAMPLES]],
            columns=["question", "answer", "predictions", "numeric_answer", "numeric_prediction"]
        )

        wandb.log({"validation/accuracy": accuracy})
        wandb.log({"validation/f1": f1})
        wandb.log({"validation/prompt_perplexity": prompt_ppl})
        wandb.log({"validation/prediction_perplexity": prediction_ppl})
        wandb.log({"validation/answer_perplexity": answer_ppl})
        wandb.log({"validation/valid_predictions": valid_predictions})
        wandb.log({"validation/demonstrations": demonstrations})
        wandb.log({"validation/data": data_table})

        self.run = {
            "accuracy": accuracy,
            "f1": f1,
            "prompt_perplexity": prompt_ppl,
            "prediction_perplexity": prediction_ppl,
            "target_perplexity": answer_ppl,
        }
