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
    Assign a categorical risk level to a census tract based on its damage profile.  Corresponds
    to levels of damage in the Mass Care Planning Tool 

    Parameters
    ----------
    damage_dist : dict
        Dictionary containing keys 'perc_extreme', 'perc_complete'

    Returns
    -------
    level of damage : str

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
    Compute RBHI (Residential Building Habitability Index) factors for each tract.

    RBHI_factor_{range} = Number of Buildings by damage[level] / Number of buildings *
                    building usability[type] * percent residential

    Note: BHI should be Building Non-Habitability Index (BNHI) instead of BHI to match 
the CMU Heinz school team final report

    level = slight, moderate, extensive, complete
    type = Fully Usable (FU), Partially Usable (PU), Not Usable (NU)
    Range = Range of non-habitability possibilities values are low, high

    Parameters
    ----------
    df : GeoDataFrame
        Contains tract-level damage estimates from prior modeling steps. (o4)
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

    # Risk level corresponds to Mass Care Planning Tool damage levels high, medium, low
    # Note that this is per tract. Red Cross normally counts households at each damage level

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
    resi_df["CENSUSCODE"] = resi_df["CENSUSCODE"].astype(str).str.zfill(11)
    df["GEOID"] = df["GEOID"].astype(str)

    df = df.merge(resi_df, left_on="GEOID", right_on="CENSUSCODE")
    df["total_resi_count"] = (
        df["RESIDENTIAL_MULTI FAMILY"] +
        df["RESIDENTIAL_OTHER"] +
        df["RESIDENTIAL_SINGLE FAMILY"]
    )
    # BHI factor is multiplied by the proportion of residential buildings to get a residential BHI
    # TODO: change name to RBNHI (Residential Building Non-Habitability Index)
    df["resi_prop"] = df["total_resi_count"] / df["TOTAL_BUILDING_COUNT"]
    df["RBHI_factor_low"] = df["BHI_factor_low"] * df["resi_prop"]
    df["RBHI_factor_high"] = df["BHI_factor_high"] * df["resi_prop"]
    
    # Calculate households impacted at each level of damage based on damage distribution levels
    # Moderate, extensive, and complete correspond to low, medium, and high damage levels

    df['residences_slight'] = (df['Total_Num_Building_Slight'] * df['resi_prop']).round(decimals=2)
    df['residences_moderate'] = (df['Total_Num_Building_Moderate'] * df['resi_prop']).round(decimals=2)
    df['residences_extensive'] = (df['Total_Num_Building_Extensive'] * df['resi_prop']).round(decimals=2)
    df['residences_complete'] = (df['Total_Num_Building_Complete'] * df['resi_prop']).round(decimals=2)

    # Step 6: Merge population and finalize columns
    pop_data["GEO_ID"] = pop_data["GEO_ID"].astype(str)
    df = df.merge(pop_data[["GEO_ID", "P1_001N"]], how="inner", left_on="GEOID", right_on="GEO_ID")
    df = df.rename(columns={"P1_001N": "population"}).drop(columns=["GEO_ID"])
    df["population"] = df["population"].astype(float)
    
    # Calculate population impacted at each level of damage using the Red Cross terms of low, medium, high
    df['population_none'] = (df['population'] * df['perc_slight']).round(decimals=2)
    df['population_low'] = (df['population'] * df['perc_moderate']).round(decimals=2)
    df['population_medium'] = (df['population'] * df['perc_extreme']).round(decimals=2)
    df['population_high'] = (df['population'] * df['perc_complete']).round(decimals=2)

    final_cols = [
        "GEOID", "max_intensity", "resi_prop", "geometry", "total_resi_count",
        "Total_Num_Building", "Total_Num_Building_Slight", "Total_Num_Building_Moderate",
        "Total_Num_Building_Extensive", "Total_Num_Building_Complete",
        "perc_slight", "perc_moderate", "perc_extreme", "perc_complete",
        "residences_slight", "residences_moderate", "residences_extensive", "residences_complete",
        "risk_level", "num_FU", "perc_FU_NH_low", "perc_FU_NH_high",
        "num_PU", "perc_PU_NH_low", "perc_PU_NH_high",
        "num_NU", "BHI_factor_low", "BHI_factor_high", "RBHI_factor_low", "RBHI_factor_high", "population",
        "population_none", "population_low", "population_medium", "population_high"
    ]
    
    return df[final_cols].sort_values(by="max_intensity", ascending=False).reset_index(drop=True)

def process_bhi_county(df, bldng_usability, ul_severity):
    """
    Compute RBHI (Residential Building Habitability Index) factors for each county.

    RBHI_factor_{range} = Number of Buildings by damage[level] / Number of buildings *
                    building usability[type] * percent residential
    
    Note: BHI should be Building Non-Habitability Index (BNHI) to match the CMU Heinz school team final report
    
    level = slight, moderate, extensive, complete
    type = Fully Usable (FU), Partially Usable (PU), Not Usable (NU)
    Range = Range of non-habitability possibilities values are low, high

    Parameters
    ----------
    df : GeoDataFrame
        Contains county-level damage estimates based on tract level from earlier steps. (o4)
    bldng_usability : dict
        Structure usability assumptions by damage category.
    ul_severity : dict
        Utility loss severity by risk level, containing [low, high] ranges.

    Returns
    -------
    GeoDataFrame
        Updated dataframe with BHI factors and joined census population by county.
    """
    # Step 0: Load population data and aggregate to county level
    pop_data = pd.read_csv("Data/census_pop/USDECENNIALPL2020.csv").iloc[1:].reset_index(drop=True)
    pop_data = pop_data[["GEO_ID", "NAME", "P1_001N"]]
    pop_data["GEO_ID"] = pop_data["GEO_ID"].str.replace("1400000US", "", regex=False)
    pop_data['P1_001N'] = pop_data['P1_001N'].astype(int)
    pop_data["countyfips"] = pop_data["GEO_ID"].str.slice(0,5)
    countypop_agg_dict = {'P1_001N' : 'sum'}
    countylevel_pop_data = pop_data.groupby("countyfips").agg(countypop_agg_dict).reset_index()
    
    # Step 1: Compute damage distribution ratios by county

    county_agg_dict = {
        'max_intensity' : 'max',
        'min_intensity' : 'min', 
        'mean_intensity' : 'mean',
        'Total_Num_Building' : 'sum',
        'Total_Num_Building_Slight' : 'sum',
        'Total_Num_Building_Moderate' : 'sum',
        'Total_Num_Building_Extensive' : 'sum',
        'Total_Num_Building_Complete' : 'sum'
    } 
    df["countyfips"] = df["GEOID"].astype(str).str.slice(-1,5) # first 5 characters of GEOID are state and county FIPS
    
    df_county = df.groupby("countyfips").agg(county_agg_dict).reset_index()

    df_county["perc_slight"] = df_county["Total_Num_Building_Slight"] / df_county["Total_Num_Building"]
    df_county["perc_moderate"] = df_county["Total_Num_Building_Moderate"] / df_county["Total_Num_Building"]
    df_county["perc_extreme"] = df_county["Total_Num_Building_Extensive"] / df_county["Total_Num_Building"]
    df_county["perc_complete"] = df_county["Total_Num_Building_Complete"] / df_county["Total_Num_Building"]

    # Risk level corresponds to Mass Care Planning Tool damage levels high, medium, low
    df_county["risk_level"] = df_county[["perc_slight", "perc_moderate", "perc_extreme", "perc_complete"]].apply(
        lambda row: tract_damage_lvl(row.to_dict()), axis=1
    )

    # Step 2: Compute FU / PU / NU counts
    df_county["num_FU"] = sum(df_county[f"Total_Num_Building_{level}"] * bldng_usability[level]["FU"] for level in bldng_usability)
    df_county["num_PU"] = sum(df_county[f"Total_Num_Building_{level}"] * bldng_usability[level]["PU"] for level in bldng_usability)
    df_county["num_NU"] = sum(df_county[f"Total_Num_Building_{level}"] * bldng_usability[level]["NU"] for level in bldng_usability)


    # Step 3: Assign utility loss severity based on risk
    df_county["perc_FU_NH_low"] = df_county["risk_level"].apply(lambda rl: ul_severity[rl]["FU"][0])
    df_county["perc_FU_NH_high"] = df_county["risk_level"].apply(lambda rl: ul_severity[rl]["FU"][1])
    df_county["perc_PU_NH_low"] = df_county["risk_level"].apply(lambda rl: ul_severity[rl]["PU"][0])
    df_county["perc_PU_NH_high"] = df_county["risk_level"].apply(lambda rl: ul_severity[rl]["PU"][1])

    # Step 4: Compute BHI factor (low/high) using utility impact
    df_county["BHI_factor_low"] = (
        df_county["num_FU"] * df_county["perc_FU_NH_low"] +
        df_county["num_PU"] * df_county["perc_PU_NH_low"] +
        df_county["num_NU"]
    ) / df_county["Total_Num_Building"]

    df_county["BHI_factor_high"] = (
        df_county["num_FU"] * df_county["perc_FU_NH_high"] +
        df_county["num_PU"] * df_county["perc_PU_NH_high"] +
        df_county["num_NU"]
    ) / df["Total_Num_Building"]

    # Step 5: Adjust for residential share of total buildings by county
    resi_df = pd.read_csv("Data/building_data_csv/aggregated_building_data.csv")
    resi_df["CENSUSCODE"] = resi_df["CENSUSCODE"].astype(str)
    resi_df["countyfips"] = resi_df["CENSUSCODE"].str.slice(0,5)
    resi_agg_dict = {
        'RESIDENTIAL_MULTI FAMILY' : 'sum',
        'RESIDENTIAL_OTHER' : 'sum',
        'RESIDENTIAL_SINGLE FAMILY' : 'sum',
        'TOTAL_BUILDING_COUNT' : 'sum',
        'STATE_ID' : 'first'
    }
    resi_county_df = resi_df.groupby("countyfips").agg(resi_agg_dict).reset_index()
    # Aggregate to county level
    df_county["countyfips"] = df_county["countyfips"].astype(str)

    df_county = df_county.merge(resi_county_df, left_on="countyfips", right_on="countyfips")
    df_county["total_resi_count"] = (
        df_county["RESIDENTIAL_MULTI FAMILY"] +
        df_county["RESIDENTIAL_OTHER"] +
        df_county["RESIDENTIAL_SINGLE FAMILY"]
    )
    # BHI factor is multiplied by the proportion of residential buildings to get a residential BHI
    df_county["resi_prop"] = df_county["total_resi_count"] / df_county["TOTAL_BUILDING_COUNT"]
    df_county["RBHI_factor_low"] = df_county["BHI_factor_low"] * df_county["resi_prop"]
    df_county["RBHI_factor_high"] = df_county["BHI_factor_high"] * df_county["resi_prop"]

    # Calculate households and populationimpacted at each level of damage based on damage distribution levels
    # Moderate, extensive, and complete correspond to low, medium, and high damage levels

    df_county['residences_slight'] = (df_county['Total_Num_Building_Slight'] * df_county['resi_prop']).round(decimals=1)
    df_county['residences_moderate'] = (df_county['Total_Num_Building_Moderate'] * df_county['resi_prop']).round(decimals=1)
    df_county['residences_extensive'] = (df_county['Total_Num_Building_Extensive'] * df_county['resi_prop']).round(decimals=1)
    df_county['residences_complete'] = (df_county['Total_Num_Building_Complete'] * df_county['resi_prop']).round(decimals=1)

    # Step 6: Merge population and finalize columns
    countylevel_pop_data["GEO_ID"] = countylevel_pop_data["countyfips"].astype(str)
    df_county = df_county.merge(countylevel_pop_data[["GEO_ID", "P1_001N"]], how="inner", left_on="GEOID", right_on="GEO_ID")
    df_county = df_county.rename(columns={"P1_001N": "population"}).drop(columns=["GEO_ID"])

    # Calculate population impacted at each level of damage using the Red Cross terms of low, medium, high
    df_county['population_none'] = (df_county['population'] * df_county['perc_slight']).round(decimals=0)
    df_county['population_low'] = (df_county['population'] * df_county['perc_moderate']).round(decimals=0)
    df_county['population_medium'] = (df_county['population'] * df_county['perc_extreme']).round(decimals=0)
    df_county['population_high'] = (df_county['population'] * df_county['perc_complete']).round(decimals=0)

    final_cols = [
        "GEOID", "max_intensity", "resi_prop", "geometry",
        "Total_Num_Building", "Total_Num_Building_Slight", "Total_Num_Building_Moderate",
        "Total_Num_Building_Extensive", "Total_Num_Building_Complete",
        "residences_slight", "residences_moderate", "residences_extensive", "residences_complete",
        "risk_level", "num_FU", "perc_FU_NH_low", "perc_FU_NH_high",
        "num_PU", "perc_PU_NH_low", "perc_PU_NH_high",
        "num_NU", "RBHI_factor_low", "RBHI_factor_high", "population",
        "population_slight", "population_moderate", "population_extensive", "population_complete"
    ]
    
    return df_county[final_cols].sort_values(by="max_intensity", ascending=False).reset_index(drop=True)