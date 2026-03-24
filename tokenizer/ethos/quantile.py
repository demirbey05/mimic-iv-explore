# Generic Quantile Calculator

import numpy as np
import pandas as pd


class QuantileCalculator:

    def __init__(self, n: int = 10):
        self.n = n
        self._bins = None

    def fit(self, values: pd.Series) -> 'QuantileCalculator':
        clean = pd.to_numeric(values, errors='coerce').dropna()
        self._bins = np.nanpercentile(clean, np.linspace(0, 100, self.n + 1))
        # Ensure unique edges
        self._bins = np.unique(self._bins)
        return self

    def __call__(self, value) -> str | None:
        if self._bins is None:
            raise RuntimeError("Call fit() before using QuantileCalculator")
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        idx = int(np.searchsorted(self._bins[1:-1], v, side='right')) + 1
        return f'Q{idx}'
