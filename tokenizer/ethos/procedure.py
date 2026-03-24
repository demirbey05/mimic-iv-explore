# ETHOS Tokenization Implementation for Procedure Codes (ICD-PCS)

import pandas as pd


class ProcedureTokenizer:

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

        # ICD-PCS Tokenization Rules:
        # Each character is a separate token, up to 7 characters.
        # Token : char  → <ICD_PCS_{char}>
        #
        # Examples:
        #   "02100"     → "<ICD_PCS_0> <ICD_PCS_2> <ICD_PCS_1> <ICD_PCS_0> <ICD_PCS_0>"
        #   "0210098"   → "<ICD_PCS_0> <ICD_PCS_2> <ICD_PCS_1> <ICD_PCS_0> <ICD_PCS_0> <ICD_PCS_9> <ICD_PCS_8>"

        def tokenize(code_str: str) -> str:
            return ' '.join(f'<ICD_PCS_{c}>' for c in code_str[:7])

        data['tokenized_version'] = data['icd_code'].astype(str).apply(tokenize)

        return data
