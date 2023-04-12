import pandas as pd
from transformers import AutoTokenizer
from data_modules.entities import Entity

ENTITY = "firstname"
OCCURRENCES = "obs"


class DemographicsDataModule():
    @staticmethod
    def entity_statistics(fpath, tokenizer_name):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        dataframe = pd.read_csv(fpath, skipfooter=1)

        entities_metadata = []

        for row in dataframe.to_dict("records"):
            text, occurrences = row[ENTITY], row[OCCURRENCES]
            text = text[0] + text[1:].lower()

            length = len(text)
            token_length = len(tokenizer(text).tokens())

            entity_metadata = Entity(
                text=text,
                occurrences=occurrences,
                label="PERSON",
                model="not_specified",
                tokenizer=tokenizer_name,
                length=length,
                token_length=token_length,
                dataset="first_name_demographics"
            )

            entities_metadata.append(entity_metadata)

        return entities_metadata
