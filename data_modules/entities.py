import pandas as pd
from dataclasses import dataclass


@dataclass
class Entity:
    text: str
    occurrences: int
    label: str
    model: str
    length: int
    token_length: int
    tokenizer: str
    dataset: str


@pd.api.extensions.register_dataframe_accessor("entities")
class EntitiesDataFrame:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.entities = dict(zip(dataframe.text, dataframe.label))

    def filter(self,
               text,
               occurrences,
               label,
               model,
               length,
               token_length,
               tokenizer,
               dataset):
        pass

    def aggregate(self,
                  text,
                  occurrences,
                  label,
                  model,
                  length,
                  token_length,
                  tokenizer,
                  dataset):
        pass

    def get(self, text):
        pass
