# FlexTPP Tokenization Implementation for Medication Codes (ATC)

import pandas as pd


class MedicationTokenizer:

    def __init__(
        self,
        code_to_name: dict | None = None,
        pre_translation: dict | None = None,
    ):
        self._code_to_name = code_to_name or {}
        self._pre_translation = pre_translation
        self.vocabulary = {}

    def build_vocabulary(self, data: pd.DataFrame) -> dict:
        data = self._divide_code_into_parts(data.copy())

        all_tokens = (
            data['tokenized_version']
            .str.split(' ')
            .explode()
            .dropna()
            .unique()
        )

        self.vocabulary = {token: idx for idx, token in enumerate(sorted(all_tokens))}
        return self.vocabulary

    def _divide_code_into_parts(self, data: pd.DataFrame) -> pd.DataFrame:
        # FlexTPP: <MED> <atc_code>
        atc_codes = data['atc_code'].astype(str)

        if self._pre_translation:
            atc_codes = atc_codes.map(self._pre_translation)

        data['tokenized_version'] = '<MED> <' + atc_codes + '>'
        return data
