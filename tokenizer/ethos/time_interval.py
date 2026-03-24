# ETHOS Tokenization Implementation for Time Intervals

import pandas as pd
from datetime import timedelta


class TimeIntervalTokenizer:
    """
    Tokenizes time intervals between events.
    Only creates tokens for gaps >= 5 minutes.
    """

    # Time interval boundaries in minutes
    INTERVALS = [
        (5, 15, '<TIME_5m-15m>'),
        (15, 60, '<TIME_15m-1h>'),
        (60, 120, '<TIME_1h-2h>'),
        (120, 360, '<TIME_2h-6h>'),
        (360, 720, '<TIME_6h-12h>'),
        (720, 1440, '<TIME_12h-1d>'),
        (1440, 4320, '<TIME_1d-3d>'),
        (4320, 10080, '<TIME_3d-1w>'),
        (10080, 20160, '<TIME_1w-2w>'),
        (20160, 43200, '<TIME_2w-1mt>'),
        (43200, 129600, '<TIME_1mt-3mt>'),
        (129600, 259200, '<TIME_3mt-6mt>'),
        (259200, float('inf'), '<TIME_>=6mt>'),
    ]

    def __init__(self):
        self.vocabulary = {}

    def build_vocabulary(self) -> dict:
        """Build vocabulary of all time interval tokens."""
        tokens = [interval[2] for interval in self.INTERVALS]
        self.vocabulary = {token: idx for idx, token in enumerate(tokens)}
        return self.vocabulary

    def get_interval_token(self, minutes: float) -> str | None:
        """
        Get the appropriate time interval token for a given duration in minutes.
        Returns None if the interval is less than 5 minutes.
        """
        if pd.isna(minutes) or minutes < 5:
            return None

        for min_bound, max_bound, token in self.INTERVALS:
            if min_bound <= minutes < max_bound:
                return token

        return None

    def tokenize_gap(self, time_delta: timedelta) -> str | None:
        """
        Tokenize a timedelta gap between events.
        Returns None if the gap is less than 5 minutes.
        """
        if pd.isna(time_delta):
            return None

        minutes = time_delta.total_seconds() / 60
        return self.get_interval_token(minutes)

    def tokenize_timestamps(self, timestamps: pd.Series) -> list[str | None]:
        """
        Given a sorted series of timestamps, return a list of time interval tokens.
        The first element is always None (no gap before first event).
        """
        if len(timestamps) == 0:
            return []

        tokens = [None]  # No gap before first event

        for i in range(1, len(timestamps)):
            prev_time = timestamps.iloc[i - 1]
            curr_time = timestamps.iloc[i]

            if pd.isna(prev_time) or pd.isna(curr_time):
                tokens.append(None)
            else:
                gap = curr_time - prev_time
                tokens.append(self.tokenize_gap(gap))

        return tokens
