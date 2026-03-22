"""
Lightweight preprocessing utilities for CIC-IDS-2018 data.

Provides a quick-clean function used by notebooks for exploratory work.
For the full pipeline, see full_preprocessing.py.
"""

import numpy as np


def clean_data(df):
    """Clean a raw CIC-IDS-2018 dataframe.

    Steps:
        1. Drop identifier columns (Flow ID, Src IP, Dst IP, Timestamp).
        2. Replace inf / -inf with NaN.
        3. Drop rows containing any NaN.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw dataframe loaded from a CIC-IDS-2018 CSV.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe with identifiers removed and no inf/NaN values.
    """
    cols_to_drop = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df
