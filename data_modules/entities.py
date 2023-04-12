import pandas as pd
from collections import namedtuple
from functools import reduce

NER_MODEL_NAME = "en_core_web_sm"

Entity = namedtuple('Entity', ['text', 'occurrences', 'label', 'model', 'length', 'token_length', 'tokenizer', 'dataset'])


@pd.api.extensions.register_dataframe_accessor("entity")
class EntitiesDataFrame:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.entities = dict([(entity["text"], Entity(**entity))
                              for entity in dataframe.to_dict("records")])

    def filter(self,
               text: str = None,
               max_occurrences: int = float("inf"),
               min_occurrences: int = 0,
               label: str = None,
               model: str = None,
               max_length: int = float("inf"),
               min_length: int = 0,
               max_token_length: int = float("inf"),
               min_token_length: int = 0,
               tokenizer: int = None,
               dataset: str = None):

        text_filter = (self.dataframe.text == text) | (text is None)
        occurrences_filter = (min_occurrences <= self.dataframe.occurrences) & (self.dataframe.occurrences <= max_occurrences)
        label_filter = (self.dataframe.label == label) | (label is None)
        model_filter = (self.dataframe.model == model) | (model is None)
        length_filter = (min_length <= self.dataframe.length) & (self.dataframe.length <= max_length)
        token_length_filter = (min_token_length <= self.dataframe.token_length) & (self.dataframe.token_length <= max_token_length)
        dataset_filter = (self.dataframe.dataset == dataset) | (dataset is None)

        dataframe_filter = reduce(
            lambda dataframe_filter, series_filter: dataframe_filter & series_filter,
            [text_filter, occurrences_filter, label_filter, model_filter, length_filter, token_length_filter, dataset_filter]
        )

        return self.dataframe[dataframe_filter]

    def aggregate(self):
        aggregate_dataframe = self.dataframe.groupby("text").agg({
            "occurrences": "sum",
            "label": lambda labels: labels.value_counts().index[0],
            "model": lambda model: model.iloc[0],
            "length": lambda length: length.iloc[0],
            "token_length": lambda token_length: token_length.iloc[0],
            "tokenizer": lambda tokenizer: tokenizer.iloc[0],
            "dataset": lambda dataset: dataset.iloc[0]
        })

        aggregate_dataframe = aggregate_dataframe.reset_index()
        return aggregate_dataframe

    def annotate(self, text):
        entities = [entity for entity_text, entity
                    in self.entities.items()
                    if entity_text in text]

        return entities

    def get(self, text):
        return self.entities[text]

    def topk(self, k):
        top_occurrences = self.dataframe["occurrences"].nlargest(k)
        top_entities_indices = self.dataframe["occurrences"].isin(top_occurrences),
        top_entities = self.dataframe.loc[top_entities_indices, "text"].tolist()[:k]
        top_entities = [self.entities[entity] for entity in top_entities]
        return top_entities
