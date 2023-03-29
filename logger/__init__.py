from datetime import datetime
from functools import reduce
from pytorch_lightning.callbacks import Callback
import wandb

PROJECT_NAME = "Entities In Context"


def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def calculate_accuracy(data):
    accuracy = reduce(
        lambda accuracy, question_answer_item: accuracy + (question_answer_item.prediction == question_answer_item.answer),
        data, 0
    )

    accuracy /= len(data)
    return accuracy

# These functions are directly taken from https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#:~:text=F1%20score%20is%20a%20common,those%20in%20the%20True%20Answer.
def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)

def calculate_f1(data):
    f1 = reduce(
        lambda f1, question_answer_item: f1 + compute_f1(question_answer_item.prediction, question_answer_item.answer),
        data, 0
    )

    f1 /= len(data)
    return f1

class QuestionAnswerLogger(Callback):
    def setup(self, trainer, pl_module, stage):
        self.run = wandb.init(project=PROJECT_NAME, name=timestamp())

    def teardown(self, trainer, pl_module, stage):
        wandb.finish()
