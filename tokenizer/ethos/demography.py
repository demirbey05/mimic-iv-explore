# ETHOS Tokenization Implementation for Demographic Data

import pandas as pd


class DemographyTokenizer:

    def __init__(self):
        self.vocabulary = {}

    def build_vocabulary(self, patients: pd.DataFrame, admissions: pd.DataFrame) -> dict:
        tokens = set()

        for val in patients['gender'].dropna().unique():
            tokens.add(self._gender_token(val))

        for val in admissions['marital_status'].dropna().unique():
            tokens.add(self._marital_token(val))

        for val in admissions['race'].dropna().unique():
            tokens.add(self._race_token(val))

        self.vocabulary = {token: idx for idx, token in enumerate(sorted(tokens))}
        return self.vocabulary

    def tokenize(self, patients: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
        # Join patients into admissions on subject_id
        df = admissions.merge(patients[['subject_id', 'gender']], on='subject_id', how='left')

        df['gender_token'] = df['gender'].map(self._gender_token)
        df['marital_token'] = df['marital_status'].map(self._marital_token)
        df['race_token'] = df['race'].map(self._race_token)

        df['tokenized_version'] = (
            df['gender_token'].fillna('')
            + ' ' + df['marital_token'].fillna('')
            + ' ' + df['race_token'].fillna('')
        ).str.strip()

        return df

    def _gender_token(self, val: str) -> str:
        return f'SEX_{val.strip().upper()}'

    def _marital_token(self, val: str) -> str:
        return 'MARITAL_' + val.strip().upper().replace(' ', '_').replace('/', '_')

    def _race_token(self, val: str) -> str:
        return 'RACE_' + val.strip().upper().replace(' ', '_').replace('/', '_').replace('-', '_')
