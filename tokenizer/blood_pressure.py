# ETHOS Tokenization Implementation for Blood Pressure

import pandas as pd
from tokenizer.quantile import QuantileCalculator


class BloodPressureTokenizer:

    def __init__(self, n_quantiles: int = 10):
        self.n_quantiles = n_quantiles
        self.vocabulary = {}
        self._systolic_q = QuantileCalculator(n=n_quantiles)
        self._diastolic_q = QuantileCalculator(n=n_quantiles)

    def fit(self, omr: pd.DataFrame) -> 'BloodPressureTokenizer':
        bp = self._extract_bp(omr)
        self._systolic_q.fit(bp['systolic'])
        self._diastolic_q.fit(bp['diastolic'])
        return self

    def build_vocabulary(self, omr: pd.DataFrame) -> dict:
        self.fit(omr)
        tokens = (
            ['<Blood_Pressure>']
            + [f'Q{i}' for i in range(1, self.n_quantiles + 1)]
        )
        self.vocabulary = {token: idx for idx, token in enumerate(tokens)}
        return self.vocabulary

    def tokenize(self, omr: pd.DataFrame) -> pd.DataFrame:
        bp = self._extract_bp(omr)
        bp['systolic_token'] = bp['systolic'].map(self._systolic_q)
        bp['diastolic_token'] = bp['diastolic'].map(self._diastolic_q)
        bp['tokenized_version'] = (
            '<Blood_Pressure> '
            + bp['systolic_token'].fillna('')
            + ' '
            + bp['diastolic_token'].fillna('')
        ).str.strip()
        return bp

    def _extract_bp(self, omr: pd.DataFrame) -> pd.DataFrame:
        bp = omr[omr['result_name'] == 'Blood Pressure'].copy()
        bp[['systolic', 'diastolic']] = bp['result_value'].str.split('/', expand=True)
        bp['systolic'] = pd.to_numeric(bp['systolic'], errors='coerce')
        bp['diastolic'] = pd.to_numeric(bp['diastolic'], errors='coerce')
        return bp
