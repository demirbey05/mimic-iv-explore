# ETHOS Tokenization Implementation for Medication Codes (ATC)

import numpy as np
import pandas as pd


class MedicationTokenizer:

    def __init__(
        self,
        code_to_name: dict | None = None,
        pre_translation: dict | None = None,
    ):
        # code_to_name : maps 3-char ATC prefix → human-readable name (e.g. "A10" → "DRUGS USED IN DIABETES")
        # pre_translation : maps source code → ATC code (e.g. NDC → ATC)
        self._code_to_name = code_to_name or {}
        self._pre_translation = pre_translation
        self.vocabulary = {}

    def build_vocabulary(self, data: pd.DataFrame) -> dict:
        parts = self._process_atc_codes(data['atc_code'])

        all_tokens = (
            pd.concat([parts['atc_part1'], parts['atc_part2'], parts['atc_part3']])
            .dropna()
            .unique()
        )

        self.vocabulary = {token: idx for idx, token in enumerate(sorted(all_tokens))}
        return self.vocabulary

    def _divide_code_into_parts(self, data: pd.DataFrame) -> pd.DataFrame:
        parts = self._process_atc_codes(data['atc_code'])

        data['tokenized_version'] = (
            parts['atc_part1'].fillna('')
            + parts['atc_part2'].apply(lambda x: (' ' + x) if pd.notna(x) else '')
            + parts['atc_part3'].apply(lambda x: (' ' + x) if pd.notna(x) else '')
        ).str.strip()

        return data

    def _process_atc_codes(self, _atc_codes: pd.Series) -> pd.DataFrame:
        atc_codes = _atc_codes.astype(str)

        if self._pre_translation:
            atc_codes = atc_codes.map(self._pre_translation)

        # Token 1: first 3 chars → look up name, prefix with "ATC_"
        atc_part1 = atc_codes.str[:3].map(
            lambda v: ('ATC_' + self._code_to_name.get(v, v)) if pd.notna(v) and v else np.nan
        )

        # Token 2: 4th character → "ATC_4_{char}"
        atc_part2 = atc_codes.str[3:4].map(
            lambda v: f'ATC_4_{v}' if v else np.nan
        )

        # Token 3: chars from index 4 onwards → "ATC_SUFFIX_{chars}"
        atc_part3 = atc_codes.str[4:].map(
            lambda v: f'ATC_SUFFIX_{v}' if v else np.nan
        )

        return pd.DataFrame({
            'atc_part1': atc_part1,
            'atc_part2': atc_part2,
            'atc_part3': atc_part3,
        })
