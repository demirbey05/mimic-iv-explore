# FlexTPP Tokenization Implementation for Blood Pressure
# Uses continuous values directly: <BP> <systolic> <diastolic>

import pandas as pd


class BloodPressureTokenizer:

    def __init__(self):
        self.vocabulary = {}

    def build_vocabulary(self, omr: pd.DataFrame) -> dict:
        self.vocabulary = {'<BP>': 0}
        return self.vocabulary

    def tokenize(self, omr: pd.DataFrame) -> pd.DataFrame:
        bp = self._extract_bp(omr)

        # Format: <BP> <systolic> <diastolic>
        bp['tokenized_version'] = bp.apply(
            lambda row: f"<BP> <{row['systolic']:.1f}> <{row['diastolic']:.1f}>"
            if pd.notna(row['systolic']) and pd.notna(row['diastolic'])
            else None,
            axis=1
        )
        return bp

    def _extract_bp(self, omr: pd.DataFrame) -> pd.DataFrame:
        bp = omr[omr['result_name'] == 'Blood Pressure'].copy()
        bp[['systolic', 'diastolic']] = bp['result_value'].str.split('/', expand=True)
        bp['systolic'] = pd.to_numeric(bp['systolic'], errors='coerce')
        bp['diastolic'] = pd.to_numeric(bp['diastolic'], errors='coerce')
        return bp
