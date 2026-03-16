# ETHOS Tokenization Implementation for Lab Events

import pandas as pd
from tokenizer.quantile import QuantileCalculator


class LabTokenizer:

    def __init__(self, n_quantiles: int = 10):
        self.n_quantiles = n_quantiles
        self.vocabulary = {}
        self._itemid_to_label = {}
        self._quantile_calculators = {}  # itemid -> QuantileCalculator

    def build_vocabulary(self, d_labitems: pd.DataFrame, labevents: pd.DataFrame) -> dict:
        self._fit(d_labitems, labevents)

        # Build vocabulary with lab tokens and quantile tokens
        # Replace spaces with underscores to avoid splitting issues
        lab_tokens = sorted(set(f'<LAB_{label.replace(" ", "_")}>' for label in self._itemid_to_label.values()))
        quantile_tokens = [f'<Q{i}>' for i in range(1, self.n_quantiles + 1)]

        all_tokens = lab_tokens + quantile_tokens
        self.vocabulary = {token: idx for idx, token in enumerate(all_tokens)}
        return self.vocabulary

    def tokenize(self, labevents: pd.DataFrame) -> pd.DataFrame:
        labevents = labevents.copy()

        def get_tokenized(row):
            itemid = row['itemid']
            if itemid not in self._itemid_to_label:
                return None

            label = self._itemid_to_label[itemid]
            # Replace spaces with underscores to avoid splitting issues
            lab_token = f'<LAB_{label.replace(" ", "_")}>'

            # Get quantile for the value
            if itemid in self._quantile_calculators:
                quantile = self._quantile_calculators[itemid](row.get('valuenum'))
                if quantile:
                    return f'{lab_token} <{quantile}>'

            return lab_token

        labevents['tokenized_version'] = labevents.apply(get_tokenized, axis=1)
        return labevents

    def _fit(self, d_labitems: pd.DataFrame, labevents: pd.DataFrame) -> None:
        # Build itemid to label mapping
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

        # Fit quantile calculators for each itemid
        for itemid in self._itemid_to_label.keys():
            item_values = labevents[labevents['itemid'] == itemid]['valuenum']
            if len(item_values.dropna()) > 0:
                qc = QuantileCalculator(n=self.n_quantiles)
                qc.fit(item_values)
                self._quantile_calculators[itemid] = qc
