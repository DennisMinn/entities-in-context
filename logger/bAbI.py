import wandb
from logger import QuestionAnswerLogger

NUM_SAMPLES = 100


class bAbILogger(QuestionAnswerLogger):
    def setup(self, trainer, pl_module, stage):
        num_tasks = len(trainer.datamodule.tasks)
        self.qa_items = {
            "train": [[] * num_tasks],
            "validation": [[] * num_tasks],
            "test": [[] * num_tasks]
        }

    def on_validation_batch_end(self,
                                trainer,
                                pl_module,
                                outputs,
                                batch,
                                batch_index,
                                dataset_index=0):

        self.qa_items["validation"][dataset_index] += outputs

    def on_validation_end(self, trainer, pl_module):
        for dataset_index, qa_items in enumerate(self.qa_items["validation"]):
            dataset = trainer.datamodule.datasets["validation"][dataset_index]
            task = dataset.task

            assert len(dataset) == len(qa_items)
            dataset.qa_items = qa_items

            accuracy = dataset.calculate_accuracy()
            f1 = dataset.calculate_f1()

            prompt_ppl, answer_ppl, prediction_ppl = dataset.calculate_perplexities()

            demonstrations = wandb.Table(
                data=[[trainer.datamodule.datasets["train"][dataset_index].demonstrations]],
                columns=["demonstrations"]
            )

            data_table = wandb.Table(
                data=[item.logging() for item in qa_items[:NUM_SAMPLES]],
                columns=["context", "question", "answer", "predictions"]
            )

            wandb.log({f"validation/task{task}/accuracy": accuracy})
            wandb.log({f"validation/task{task}/f1": f1})
            wandb.log({f"validation/task{task}/prompt_perplexity": prompt_ppl})
            wandb.log({f"validation/task{task}/prediction_perplexity": prediction_ppl})
            wandb.log({f"validation/task{task}/answer_perplexity": answer_ppl})
            wandb.log({f"validation/task{task}/demonstrations": demonstrations})
            wandb.log({f"validation/task{task}/data": data_table})

            self.run[f"task{task}"] = {
                "accuracy": accuracy,
                "f1": f1,
                "prompt_perplexity": prompt_ppl,
                "prediction_perplexity": prediction_ppl,
                "target_perplexity": answer_ppl,
            }
