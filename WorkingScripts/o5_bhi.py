import geopandas as gpd
import pandas as pd
import numpy as np

"""
Building Habitability Index (BHI) Estimation Module

This module estimates utility service disruption and habitability impacts
at the census tract level based on post-earthquake building damage outcomes.

Inputs:
- `df`: GeoDataFrame with damage estimates from structural analysis
- `bldng_usability`: Nested dict of functional (FU), partially usable (PU), and non-usable (NU)
  building assumptions by damage level
- `ul_severity`: Dictionary mapping 'low', 'medium', 'high' damage categories to
  expected percent of utility loss for FU/PU categories

Steps:
1. Read damage results and census population data
2. Compute damage distribution ratios and assign tract-level risk labels
3. Estimate total FU / PU / NU buildings per tract
4. Compute low/high BHI factors incorporating utility service loss
5. Join population estimates to final dataframe
"""


def tract_damage_lvl(damage_dist):
    """
    Assign a categorical risk level to a census tract based on its damage profile.

    Parameters
    ----------
    damage_dist : dict
        Dictionary containing keys 'perc_extreme', 'perc_complete'

    Returns
    -------
    str
        One of "low", "medium", or "high"
    """
    destroyed = damage_dist["perc_complete"]
    major = damage_dist["perc_extreme"]  # Can substitute or extend with moderate

    if (destroyed > 0.34) or (major > 0.34):
        return "high"
    elif (0.1 < destroyed <= 0.34) or (0.15 < major <= 0.34):
        return "medium"
    else:
        return "low"


def process_bhi(df, bldng_usability, ul_severity):
    """
    Compute BHI (Building Habitability Index) factors for each tract.

    Parameters
    ----------
    df : GeoDataFrame
        Contains tract-level damage estimates from prior modeling steps.
    bldng_usability : dict
        Structure usability assumptions by damage category.
    ul_severity : dict
        Utility loss severity by risk level, containing [low, high] ranges.

    Returns
    -------
    GeoDataFrame
        Updated dataframe with BHI factors and joined census population.
    """
    # Step 0: Load population data
    pop_data = pd.read_csv("Data/census_pop/USDECENNIALPL2020.csv").iloc[1:].reset_index(drop=True)
    pop_data = pop_data[["GEO_ID", "NAME", "P1_001N"]]
    pop_data["GEO_ID"] = pop_data["GEO_ID"].str.replace("1400000US", "", regex=False)

    # Step 1: Compute damage distribution ratios
    df["perc_slight"] = df["Total_Num_Building_Slight"] / df["Total_Num_Building"]
    df["perc_moderate"] = df["Total_Num_Building_Moderate"] / df["Total_Num_Building"]
    df["perc_extreme"] = df["Total_Num_Building_Extensive"] / df["Total_Num_Building"]
    df["perc_complete"] = df["Total_Num_Building_Complete"] / df["Total_Num_Building"]

    df["risk_level"] = df[["perc_slight", "perc_moderate", "perc_extreme", "perc_complete"]].apply(
        lambda row: tract_damage_lvl(row.to_dict()), axis=1
    )

    # Step 2: Compute FU / PU / NU counts
    df["num_FU"] = sum(df[f"Total_Num_Building_{level}"] * bldng_usability[level]["FU"] for level in bldng_usability)
    df["num_PU"] = sum(df[f"Total_Num_Building_{level}"] * bldng_usability[level]["PU"] for level in bldng_usability)
    df["num_NU"] = sum(df[f"Total_Num_Building_{level}"] * bldng_usability[level]["NU"] for level in bldng_usability)

    # Step 3: Assign utility loss severity based on risk
    df["perc_FU_NH_low"] = df["risk_level"].apply(lambda rl: ul_severity[rl]["FU"][0])
    df["perc_FU_NH_high"] = df["risk_level"].apply(lambda rl: ul_severity[rl]["FU"][1])
    df["perc_PU_NH_low"] = df["risk_level"].apply(lambda rl: ul_severity[rl]["PU"][0])
    df["perc_PU_NH_high"] = df["risk_level"].apply(lambda rl: ul_severity[rl]["PU"][1])

    # Step 4: Compute BHI factor (low/high) using utility impact
    df["BHI_factor_low"] = (
        df["num_FU"] * df["perc_FU_NH_low"] +
        df["num_PU"] * df["perc_PU_NH_low"] +
        df["num_NU"]
    ) / df["Total_Num_Building"]

    df["BHI_factor_high"] = (
        df["num_FU"] * df["perc_FU_NH_high"] +
        df["num_PU"] * df["perc_PU_NH_high"] +
        df["num_NU"]
    ) / df["Total_Num_Building"]

    # Step 5: Adjust for residential share of total buildings
    resi_df = pd.read_csv("Data/building_data_csv/aggregated_building_data.csv")
    resi_df["CENSUSCODE"] = resi_df["CENSUSCODE"].astype(int)
    df["GEOID"] = df["GEOID"].astype(int)

    df = df.merge(resi_df, left_on="GEOID", right_on="CENSUSCODE")
    df["total_resi_count"] = (
        df["RESIDENTIAL_MULTI FAMILY"] +
        df["RESIDENTIAL_OTHER"] +
        df["RESIDENTIAL_SINGLE FAMILY"]
    )
    df["resi_prop"] = df["total_resi_count"] / df["TOTAL_BUILDING_COUNT"]
    df["BHI_factor_low"] *= df["resi_prop"]
    df["BHI_factor_high"] *= df["resi_prop"]

    # Step 6: Merge population and finalize columns
    pop_data["GEO_ID"] = pop_data["GEO_ID"].astype(int)
    df = df.merge(pop_data[["GEO_ID", "P1_001N"]], how="inner", left_on="GEOID", right_on="GEO_ID")
    df = df.rename(columns={"P1_001N": "population"}).drop(columns=["GEO_ID"])

    final_cols = [
        "GEOID", "max_intensity", "resi_prop", "geometry",
        "Total_Num_Building", "Total_Num_Building_Slight", "Total_Num_Building_Moderate",
        "Total_Num_Building_Extensive", "Total_Num_Building_Complete",
        "risk_level", "num_FU", "perc_FU_NH_low", "perc_FU_NH_high",
        "num_PU", "perc_PU_NH_low", "perc_PU_NH_high",
        "num_NU", "BHI_factor_low", "BHI_factor_high", "population"
    ]
    return df[final_cols].sort_values(by="max_intensity", ascending=False).reset_index(drop=True)
