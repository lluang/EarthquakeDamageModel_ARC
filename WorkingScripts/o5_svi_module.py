"""
Social Vulnerability Index (SVI) Module

This module reads and processes CDC SVI 2022 data to assign mapped SVI vulnerability levels
to census tracts. These mapped values can be used to estimate shelter-seeking behavior
or risk exposure based on social vulnerability.

Key functions:
- `read_svi_data`: Load raw SVI data (FIPS + composite vulnerability score)
- `configure_svi_map`: Generate a function to map raw SVI score to a discrete vulnerability level
- `process_svi`: Apply mapping to create a cleaned SVI dataframe for merging

Expected columns in input CSV:
- FIPS: tract identifier
- RPL_THEMES: overall percentile rank of social vulnerability (0 to 1)
"""

import os
import geopandas as gpd
import pandas as pd

def read_svi_data():
    """
    Read the CDC SVI 2022 CSV and return the relevant columns.

    Returns
    -------
    DataFrame
        Contains FIPS code and composite SVI score (RPL_THEMES).
    """
    svi_path = "Data/SVI/SVI_2022_US.csv"
    svi = pd.read_csv(svi_path)
    return svi


def configure_svi_map(svi_factor_set):
    """
    Return a function that maps SVI values (0–1) to shelter-seeking assumptions.

    Parameters
    ----------
    svi_factor_set : list of float
        Three values representing proportions for [low, medium, high] SVI levels.

    Returns
    -------
    function
        A function mapping an SVI score to its assigned shelter-seeking proportion.
    """
    def map_range(val):
        if 0 <= val < 0.5:
            return svi_factor_set[0]
        elif 0.5 <= val < 0.8:
            return svi_factor_set[1]
        elif 0.8 <= val <= 1.0:
            return svi_factor_set[2]
        else:
            return None
    return map_range


def process_svi(svi_factor_set):
    """
    Load and process SVI data, applying a mapped shelter-seeking vulnerability level.

    Parameters
    ----------
    svi_factor_set : list of float
        Proportions to apply for [low, medium, high] SVI bins.

    Returns
    -------
    DataFrame
        DataFrame with FIPS, raw SVI value, and mapped vulnerability factor.
    """
    svi_data = read_svi_data()
    svi_data = svi_data[["FIPS", "RPL_THEMES"]].rename(columns={"RPL_THEMES": "SVI_Value"})
    map_range = configure_svi_map(svi_factor_set)
    svi_data["SVI_Value_Mapped"] = svi_data["SVI_Value"].apply(map_range)
    return svi_data
