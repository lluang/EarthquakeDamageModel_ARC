"""
Earthquake Damage Estimation Module

This module estimates earthquake-induced building damage at the census tract level
using fragility functions and peak ground acceleration (PGA) values.

Main components:
- `read_damage_functions`: Loads fragility curves for various building types and seismic code levels.
- `build_damage_estimates`: Combines tract-level hazard intensity with building inventory
  to estimate the probability and count of buildings experiencing slight, moderate, extensive,
  and complete damage.

Inputs:
- A census-tract-level GeoDataFrame (`event_results`) with minimum PGA values and building counts
- A CSV table of fragility function parameters (`DamageFunctionVariables.csv`)

Outputs:
- A GeoDataFrame with estimated damage counts per tract, disaggregated by severity


# The damage function variable table explained:

Each row represents a combination of building type and building code
building type refers to the (such as structure or materials of the building: frames, wall types)
building codes are the classification of its seismic codes
for example: HC (high code) is the most resistant. 

Each column (median moderate, median extensive, median complete ...) refers to a damage state
The values represents the PGA (Peak Ground Acceleration) value at which such damage state will occur.
For example, if MedianModerate = 0.22, it means given the building type and building code,
moderate damage can be expected at a PGA of 0.22.

Those columns were followed by the beta columns (BetaSlight, BetaModerate, BetaExtensive...)
Those are lognormal standard deviations (used to compute confidence intervals or model uncertainty).
Small beta means this certain type of building has similar falling threshold (smaller uncertainty)

"""


import os
import geopandas as gpd
import pandas as pd
from scipy.stats import norm
import numpy as np
import time

def read_damage_functions():
    """
    Load damage function variable table from CSV and extract column metadata.

    Returns
    -------
    tuple
        (DataFrame with all values, list of building types, list of median columns, list of beta columns)
    """
    path = os.path.join(os.getcwd(), "Tables", "DamageFunctionVariables.csv")
    df = pd.read_csv(path).drop(columns=["Unnamed: 0"], errors="ignore")

    median_columns = [col for col in df.columns if col.lower().startswith("median")]
    beta_columns = [col for col in df.columns if col.lower().startswith("beta")]
    building_types = df["BLDG_TYPE"].unique()

    return df, building_types, median_columns, beta_columns


def build_damage_estimates(event_results, intensity_metric):
    """
    Estimate earthquake building damage by combining PGA intensity with fragility curves.

    Parameters
    ----------
    event_results : GeoDataFrame
        Contains census tract-level PGA values and building counts per structural type.

    Returns
    -------
    GeoDataFrame
        A dataframe with total estimated building damage counts (slight, moderate, extensive, complete)
        for each tract.
    """
    # Set up labels for each damage level
    pga_levels = ['slight', 'mod', 'ext', 'comp']

    # Load damage function parameters
    dmgfvarsDF, list_bldgtypes, median_columns, beta_columns = read_damage_functions()

    # ----------------------------------------------------------------------
    # Assumption 1: Use highest seismic code (e.g., HC > MC > LC > PC)
    priority_order = {"HC": 1, "MC": 2, "LC": 3, "PC": 4}
    dmgfvarsDF["priority"] = dmgfvarsDF["BUILDINGCO"].map(priority_order)
    dmgfvarsDF = dmgfvarsDF.sort_values(["BLDG_TYPE", "priority"])
    dmgfvars_hc = dmgfvarsDF.groupby("BLDG_TYPE").first().reset_index().drop(columns=["priority"])

    # ----------------------------------------------------------------------
    # Estimate total number of buildings per tract
    event_results['Total_Num_Building'] = (
        event_results['OTHER_OTHER'] +
        event_results['RESIDENTIAL_MULTI FAMILY'] +
        event_results['RESIDENTIAL_OTHER'] +
        event_results['RESIDENTIAL_SINGLE FAMILY']
    )

    # Step 1: Compute probability of damage by structure and level
    prob_dict = {}

    # Add check to ensure all building types exist in event_results before looping
    intensity_metric = "{}_intensity".format(intensity_metric)
    print("Using {}".format(intensity_metric))
    for bldg_type in list_bldgtypes:
        for i in range(len(pga_levels)):
            beta = dmgfvars_hc.loc[dmgfvars_hc['BLDG_TYPE'] == bldg_type, beta_columns[i]].item()
            PGA_median = dmgfvars_hc.loc[dmgfvars_hc['BLDG_TYPE'] == bldg_type, median_columns[i]].item()

            prob_key = f'P_{pga_levels[i]}_{bldg_type}'
            prob_dict[prob_key] = norm.cdf((1 / beta) * np.log(event_results[intensity_metric] / PGA_median))

    df = pd.concat([event_results, pd.DataFrame(prob_dict)], axis=1)

    # ----------------------------------------------------------------------
    # Step 2–3: Estimate exclusive damage counts
    num_slight, num_moderate, num_extensive, num_complete = {}, {}, {}, {}
    ex_slight, ex_moderate, ex_extensive, ex_complete = {}, {}, {}, {}

    for bldg_type in list_bldgtypes:
        bldg_count_col = f"{bldg_type}_COUNT"

        # Cumulative damage counts
        num_slight[f'numSlight_{bldg_type}'] = df[bldg_count_col] * df[f'P_slight_{bldg_type}']
        num_moderate[f'numModerate_{bldg_type}'] = num_slight[f'numSlight_{bldg_type}'] * df[f'P_mod_{bldg_type}']
        num_extensive[f'numExtensive_{bldg_type}'] = num_moderate[f'numModerate_{bldg_type}'] * df[f'P_ext_{bldg_type}']
        num_complete[f'numComplete_{bldg_type}'] = num_extensive[f'numExtensive_{bldg_type}'] * df[f'P_comp_{bldg_type}']

        # Exclusive counts (subtract from previous level)
        ex_slight[f'numSlight_{bldg_type}'] = num_slight[f'numSlight_{bldg_type}'] - num_moderate[f'numModerate_{bldg_type}']
        ex_moderate[f'numModerate_{bldg_type}'] = num_moderate[f'numModerate_{bldg_type}'] - num_extensive[f'numExtensive_{bldg_type}']
        ex_extensive[f'numExtensive_{bldg_type}'] = num_extensive[f'numExtensive_{bldg_type}'] - num_complete[f'numComplete_{bldg_type}']
        ex_complete[f'numComplete_{bldg_type}'] = num_complete[f'numComplete_{bldg_type}']

    # Combine all exclusive counts into dataframe
    df = pd.concat([df, pd.DataFrame({
        **ex_slight,
        **ex_moderate,
        **ex_extensive,
        **ex_complete
    })], axis=1)

    # ----------------------------------------------------------------------
    # Step 4: Total damage count per tract
    df["Total_Num_Building_Slight"] = df.filter(like="numSlight").sum(axis=1)
    df["Total_Num_Building_Moderate"] = df.filter(like="numModerate").sum(axis=1)
    df["Total_Num_Building_Extensive"] = df.filter(like="numExtensive").sum(axis=1)
    df["Total_Num_Building_Complete"] = df.filter(like="numComplete").sum(axis=1)

    # Final output selection
    df_final = df[[
        'GEOID', 'max_intensity', 'min_intensity', 'mean_intensity', 'geometry',
        'Total_Num_Building',
        'Total_Num_Building_Slight',
        'Total_Num_Building_Moderate',
        'Total_Num_Building_Extensive',
        'Total_Num_Building_Complete'
    ]]

    return df_final
