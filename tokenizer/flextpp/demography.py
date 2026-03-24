# FlexTPP Tokenization Implementation for Demographic Data
# Format: <Type> <value> for each demographic field

import pandas as pd


class DemographyTokenizer:

    def __init__(self):
        self.vocabulary = {}

    def build_vocabulary(self, patients: pd.DataFrame, admissions: pd.DataFrame) -> dict:
        tokens = {'<SEX>', '<MARITAL>', '<RACE>'}

        for val in patients['gender'].dropna().unique():
            tokens.add(f'<{val.strip().upper()}>')

        for val in admissions['marital_status'].dropna().unique():
            tokens.add(f'<{self._normalize(val)}>')

        for val in admissions['race'].dropna().unique():
            tokens.add(f'<{self._normalize(val)}>')

        self.vocabulary = {token: idx for idx, token in enumerate(sorted(tokens))}
        return self.vocabulary

    def tokenize(self, patients: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
        df = admissions.merge(patients[['subject_id', 'gender']], on='subject_id', how='left')

        # Format: <SEX> <M> <MARITAL> <MARRIED> <RACE> <WHITE>
        df['gender_token'] = df['gender'].apply(
            lambda x: f'<SEX> <{x.strip().upper()}>' if pd.notna(x) else ''
        )
        df['marital_token'] = df['marital_status'].apply(
            lambda x: f'<MARITAL> <{self._normalize(x)}>' if pd.notna(x) else ''
        )
        df['race_token'] = df['race'].apply(
            lambda x: f'<RACE> <{self._normalize(x)}>' if pd.notna(x) else ''
        )

        df['tokenized_version'] = (
            df['gender_token']
            + ' ' + df['marital_token']
            + ' ' + df['race_token']
        ).str.strip()

        return df

    def _normalize(self, val: str) -> str:
        return val.strip().upper().replace(' ', '_').replace('/', '_').replace('-', '_')
