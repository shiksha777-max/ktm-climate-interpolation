"""
data_loader.py
==============
Load and preprocess the Kathmandu Station 1030 temperature dataset.
Source: Open Data Nepal / CBS Statistical Year Book 2015
"""

import numpy as np
import pandas as pd
from pathlib import Path

MONTHS     = ['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec']
MONTH_NUMS = np.arange(1, 13, dtype=float)


def load_data(csv_path=None):
    """
    Load the Kathmandu Station 1030 temperature CSV.

    Parameters
    ----------
    csv_path : str or Path, optional
        Defaults to ../data/kathmandu_temperature_station1030.csv

    Returns
    -------
    df : pd.DataFrame  columns: Year, Jan…Dec, Annual_Mean
    """
    if csv_path is None:
        here     = Path(__file__).parent
        csv_path = here.parent / "data" / "kathmandu_temperature_station1030.csv"

    df = pd.read_csv(csv_path)
    df['Annual_Mean'] = df[MONTHS].mean(axis=1)
    return df


def get_monthly_means(df):
    """Compute multi-year monthly mean temperatures (12-element array)."""
    return df[MONTHS].mean().values


def subsample_months(y_full, n_known, equally_spaced=True):
    """
    Select n_known months from the 12-month array.

    Parameters
    ----------
    y_full        : array of 12 temperature values
    n_known       : int, number of months to keep
    equally_spaced: if True, pick evenly; else pick randomly

    Returns
    -------
    x_known, y_known : index arrays (1-based month numbers) and temps
    """
    if equally_spaced:
        idx = np.linspace(0, 11, n_known, dtype=int)
    else:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(12, n_known, replace=False))
    return MONTH_NUMS[idx], y_full[idx]


def get_year_data(df, year):
    """Return 12-month temperature array for a specific year."""
    row = df[df['Year'] == year]
    if row.empty:
        raise ValueError(f"Year {year} not found in dataset.")
    return row[MONTHS].values[0]


def dataset_info(df):
    """Print a summary of the dataset."""
    print("=" * 60)
    print("  Kathmandu Station 1030 — Dataset Summary")
    print("  Source: Open Data Nepal / CBS Statistical Year Book 2015")
    print("=" * 60)
    print(f"  Years      : {int(df['Year'].min())} – {int(df['Year'].max())}")
    print(f"  Records    : {len(df)} years × 12 months")
    print(f"  Temp range : {df[MONTHS].min().min():.1f}°C – "
          f"{df[MONTHS].max().max():.1f}°C")
    print(f"  Coldest    : January  ({df[MONTHS].mean()['Jan']:.2f}°C avg)")
    print(f"  Warmest    : July     ({df[MONTHS].mean()['Jul']:.2f}°C avg)")
    print("=" * 60)
    print()
    print(df.to_string(index=False))
    print()
