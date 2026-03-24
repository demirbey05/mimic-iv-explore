# FlexTPP Tokenization Implementation for Procedure Codes

import pandas as pd


class ProcedureTokenizer:

    def __init__(self, icd_9_10_conversion_map: dict = None):
        self.vocabulary = {}
        self.icd_9_10_conversion_map = icd_9_10_conversion_map
        if self.icd_9_10_conversion_map is None:
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_file = os.path.join(base_dir, '..', 'data', 'icd_pcs_9_to_10_mapping.csv.gz')
            if os.path.exists(data_file):
                df = pd.read_csv(data_file, dtype=str)
                self.icd_9_10_conversion_map = dict(zip(df['icd_9'], df['icd_10']))
                self._pre_translation = True
            else:
                self.icd_9_10_conversion_map = {}
                self._pre_translation = False
        else:
            self._pre_translation = True

    def build_vocabulary(self, data: pd.DataFrame) -> dict:
        if self._pre_translation:
            data = self._translate_icd_9_to_10(data.copy())
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
        # FlexTPP: <PROC> <code>
        code = data['icd_code'].astype(str)
        data['tokenized_version'] = '<PROC> <' + code + '>'
        return data

    def _translate_icd_9_to_10(self, data: pd.DataFrame) -> pd.DataFrame:
        unmapped_mask = ~data['icd_code'].astype(str).isin(self.icd_9_10_conversion_map)
        missing_codes = data.loc[unmapped_mask, 'icd_code'].unique()
        if len(missing_codes) > 0:
            import warnings
            warnings.warn(f"{len(missing_codes)} ICD-9 procedure codes were missed during conversion to ICD-10. Examples: {missing_codes[:5]}")
        data['icd_code'] = data['icd_code'].astype(str).map(self.icd_9_10_conversion_map).fillna(data['icd_code'])
        return data