# ETHOS Tokenization Implementation for Medication Codes (ATC)

import pandas as pd


class MedicationTokenizer:

    def __init__(self):
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

        # ATC Tokenization Rules:
        # Token 1 : chars[0:3]  — always present  → <ATC_F_{chars}>
        # Token 2 : chars[3:5]  — present if len >= 4  → <ATC_3_5_{chars}>
        # Token 3 : chars[5:]   — present if len >= 6  → <ATC_6_{chars}>
        #
        # Examples:
        #   "A10"      → "<ATC_F_A10>"
        #   "A10B"     → "<ATC_F_A10> <ATC_3_5_B>"
        #   "A10BA"    → "<ATC_F_A10> <ATC_3_5_BA>"
        #   "A10BA02"  → "<ATC_F_A10> <ATC_3_5_BA> <ATC_6_02>"

        code = data['atc_code'].astype(str)
        length = code.str.len()

        token1 = '<ATC_F_' + code.str[:3] + '>'
        token2 = code.str[3:5].where(length >= 4, '')
        token3 = code.str[5:].where(length >= 6, '')

        data['tokenized_version'] = (
            token1
            + token2.apply(lambda x: (' <ATC_3_5_' + x + '>') if x else '')
            + token3.apply(lambda x: (' <ATC_6_' + x + '>') if x else '')
        )

        return data
