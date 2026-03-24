# FlexTPP Tokenization Implementation for Lab Events
# Uses continuous values directly: <Lab> <name> <value>

import pandas as pd


class LabTokenizer:

    def __init__(self):
        self.vocabulary = {}
        self._itemid_to_label = {}

    def build_vocabulary(self, d_labitems: pd.DataFrame, labevents: pd.DataFrame) -> dict:
        self._fit(d_labitems)

        # Build vocabulary with type token and lab name tokens
        tokens = ['<Lab>']
        tokens += sorted(set(
            f'<{label.replace(" ", "_")}>'
            for label in self._itemid_to_label.values()
        ))

        self.vocabulary = {token: idx for idx, token in enumerate(tokens)}
        return self.vocabulary

    def tokenize(self, labevents: pd.DataFrame) -> pd.DataFrame:
        labevents = labevents.copy()

        def get_tokenized(row):
            itemid = row['itemid']
            if itemid not in self._itemid_to_label:
                return None

            label = self._itemid_to_label[itemid]
            name_token = f'<{label.replace(" ", "_")}>'

            # Format: <Lab> <name> <value>
            valuenum = row.get('valuenum')
            if pd.notna(valuenum):
                return f'<Lab> {name_token} <{valuenum:.4g}>'

            return f'<Lab> {name_token}'

        labevents['tokenized_version'] = labevents.apply(get_tokenized, axis=1)
        return labevents

    def _fit(self, d_labitems: pd.DataFrame) -> None:
        def normalize(label) -> str | None:
            if pd.isna(label):
                return None
            return str(label).strip()

        self._itemid_to_label = (
            d_labitems.set_index('itemid')['label']
            .map(normalize)
            .dropna()
            .to_dict()
        )
