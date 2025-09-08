
# ========== O1 ====================================
from WorkingScripts.o1_getshakemap import FEEDURL
from WorkingScripts.o1_getshakemap import fetch_earthquake_data, retrieve_event_data, download_and_extract_shakemap
# ========== O2 ====================================
from WorkingScripts.o2_download_census import download_census
from WorkingScripts.o2_census_intersect import shakemap_into_census_geo
# ========== O3 ====================================
from WorkingScripts.o3_clip_eventdata_buildingstocks import building_clip_analysis
from WorkingScripts.o3_get_building_structure import o3_get_building_structures
# ========== O4 ====================================
from WorkingScripts.o4_TractLevel_DamageAssessmentModel import build_damage_estimates
# ========== O5 ====================================
from WorkingScripts.o5_bhi import process_bhi
from WorkingScripts.o5_svi_module import process_svi

import os
import pandas as pd
import time

# Set to true if user wishes to rebuild building centroid data
DOWNLOAD_BUILDING_CENTROID = False

def main(**config):
    """
    config is the dictionary with user specified arguments
    """
    # ==============================================
    # user parameters
    # ==============================================
    EVENT_ID = config["event_id"]

    # ==============================================
    # o1 - retrieve shakemap for specified event ID
    # ==============================================
    # o1 parameters
    feed_url = FEEDURL.format(EVENT_ID)
    # o1 process
    jdict = fetch_earthquake_data(feed_url=feed_url)
    event = retrieve_event_data(jdict)
    download_and_extract_shakemap(event)

    # ================================================
    # o2 - Download US Census Tract Shapemap (Optional)
    # ================================================    
    # download national census data if missing
    download_census()

    # ================================================
    # o2 - Overlay US Census Tract Data onto ShakeMap
    # ================================================
    # clip census and shakemaps, min,max,mean pga per census tract
    event_dir = os.path.join(os.getcwd(), 'Data', 'Shakemap', EVENT_ID)
    shakemap_into_census_geo(event_dir)

    # ================================================
    # o3 - Download Building Centroid Data (Optional)
    # ================================================
    # download and extract the building data
    if DOWNLOAD_BUILDING_CENTROID:
        start_time = time.time()
        o3_get_building_structures()
        end_time = time.time()
        print(f"Function took {end_time - start_time:.4f} seconds to run.")

    # ================================================
    # o3 - Building Centroids
    #     Perform building clip analysis for a specific event ID
    # ================================================
    event_results = building_clip_analysis(EVENT_ID)

    # ========================================================
    # o4 - Apply Damage Functions using Building Code Data
    # ========================================================
    o4out = build_damage_estimates(event_results, config["intensity_metric"])

    # ================================================
    # o5 - Implement BHI
    # ================================================
    df = process_bhi(o4out, config["BLDNG_USABILITY"], config["UL_SEVERITY"])

    df["population"] = df["population"].astype(int)
    df["shelter_seeking_low"] = df["BHI_factor_low"]*df["population"]
    df["shelter_seeking_high"] = df["BHI_factor_high"]*df["population"]
    cols = ["GEOID", "max_intensity", "population", 
            "Total_Num_Building", "risk_level", "geometry",
            "BHI_factor_low", "BHI_factor_high",
            "shelter_seeking_low", "shelter_seeking_high",
            "Total_Num_Building_Slight", "Total_Num_Building_Moderate", 
            "Total_Num_Building_Extensive", "Total_Num_Building_Complete"]
    df = df[cols]
    df["GEOID"] = df["GEOID"].astype(str)
    

    # ================================================
    # o6 - Download SVI data 
    # ================================================
    # apply SVI 
    svi = process_svi(config["SVI_THRESHOLD"])
    svi["FIPS"] = svi["FIPS"].astype(str)
    
    # ================================================
    # o7 - Combine SVI and BHI, Format Output Data
    # ================================================
    df = df.merge(svi, left_on = "GEOID", right_on="FIPS")
    df["shelter_seeking_low"] = df["shelter_seeking_low"]*df["SVI_Value_Mapped"] 
    df["shelter_seeking_high"] = df["shelter_seeking_high"]*df["SVI_Value_Mapped"]
    df = df.drop(columns=["FIPS"])
    
    columns = [
        "GEOID",
        "max_intensity",
        "population",
        "Total_Num_Building",
        "risk_level",
        "BHI_factor_low",
        "BHI_factor_high",
        "shelter_seeking_low",
        "shelter_seeking_high",
        "Total_Num_Building_Slight",
        "Total_Num_Building_Moderate",
        "Total_Num_Building_Extensive",
        "Total_Num_Building_Complete",
        "SVI_Value",
        "SVI_Value_Mapped"]

    df = df[columns]
    df.to_csv("Data/model_output_{}.csv".format(config["event_id"]), index=False)
    print("lower bound")
    print(df["shelter_seeking_low"].sum())
    print("upper bound")
    print(df["shelter_seeking_high"].sum())

