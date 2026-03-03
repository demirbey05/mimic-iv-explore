# ETHOS Tokenization Implementation for Diagnosis Codes

import pandas as pd


class DiagnoseTokenizer:

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

        # ICD-10-CM Tokenization Rules:
        # Token 1 : chars[0:3]  — always present  → <ICD_F_{chars}>
        # Token 2 : chars[3:5]  — present if len >= 4  → <ICD_3_5_{chars}>
        # Token 3 : chars[5:]   — present if len >= 6  → <ICD_6_{chars}>
        #
        # Examples:
        #   "A00"    → "<ICD_F_A00>"
        #   "A000"   → "<ICD_F_A00> <ICD_3_5_0>"
        #   "A0001"  → "<ICD_F_A00> <ICD_3_5_01>"
        #   "A00011" → "<ICD_F_A00> <ICD_3_5_01> <ICD_6_1>"

        code = data['icd_code'].astype(str)
        length = code.str.len()

        token1 = '<ICD_F_' + code.str[:3] + '>'
        token2 = code.str[3:5].where(length >= 4, '')
        token3 = code.str[5:].where(length >= 6, '')

        data['tokenized_version'] = (
            token1
            + token2.apply(lambda x: (' <ICD_3_5_' + x + '>') if x else '')
            + token3.apply(lambda x: (' <ICD_6_' + x + '>') if x else '')
        )

        return data
