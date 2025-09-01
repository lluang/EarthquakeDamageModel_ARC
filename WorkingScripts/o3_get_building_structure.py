
"""
Building Footprint Data Aggregation Module

This module automates the process of downloading, extracting, and aggregating
state-level building footprint data from the U.S. Federal GeoPlatform 
(USA_Structures). It supports:

- Scraping state-level GDB download links
- Downloading and extracting ZIP archives
- Reading GDB files and selecting relevant columns
- Remapping occupancy classifications
- Aggregating and pivoting building counts by census tract
- Producing a unified CSV of building data across all states
"""

import os
import time
import zipfile
import requests
import glob
import pandas as pd
import numpy as np
import geopandas as gpd
from bs4 import BeautifulSoup


def make_data_path():
    """
    Create and return paths to data subdirectories for building data.

    This function ensures that the following folders exist under the
    parent directory of the current working directory:
    - Data/building_data_csv: for saving per-state aggregated CSVs
    - Data/building_data_gdb: for extracted raw GDB files
    - Data/building_stock_data: optional output directory

    Returns
    -------
    tuple of str
        Paths to (building_data_csv, building_data_gdb, building_stock_data)
    """
    parent = os.path.dirname(os.getcwd())
    data_path = os.path.join(parent, "Data")

    # Create base data directory and subdirectories if needed
    building_data_csv = os.path.join(data_path, "building_data_csv")
    building_data_gdb = os.path.join(data_path, "building_data_gdb")
    building_stock_data = os.path.join(data_path, "building_stock_data")

    for path in [building_data_csv, building_data_gdb, building_stock_data]:
        os.makedirs(path, exist_ok=True)

    return building_data_csv, building_data_gdb, building_stock_data


def fetch_state_links():
    """
    Scrape the USA_Structures GeoPlatform site for state-level download links.

    This function searches for all downloadable ZIP files containing building
    structure data, based on the presence of 'Deliverable' in the href.

    Returns
    -------
    dict
        A dictionary mapping state names (str) to their corresponding ZIP URLs (str).

    Raises
    ------
    ValueError
        If the request to the webpage fails or no valid links are found.
    """
    url = "https://disasters.geoplatform.gov/USA_Structures/"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Failed to fetch webpage. Status code: {response.status_code}")
        raise ValueError("Could not access USA_Structures page.")

    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.find_all("a", href=True)

    # Filter for links containing 'Deliverable' (GDB ZIP files)
    deliverable_links = {
        link.text.strip(): link["href"]
        for link in links
        if "Deliverable" in link["href"]
    }

    if not deliverable_links:
        # empty dictionary
        raise ValueError("No Deliverable ZIP links found on page.")

    return deliverable_links


def download_and_extract_zip(state_name, state_links):
    """
    Download and extract a ZIP archive of building data for a given state.

    Parameters
    ----------
    state_name : str
        Name of the state (must match key in `state_links`).
    state_links : dict
        Dictionary mapping state names to ZIP URLs (from `fetch_state_links()`).

    Raises
    ------
    ValueError
        If the download fails or the state name is not in the links.
    """
    if state_name not in state_links:
        raise ValueError(f"No download link found for state: {state_name}")

    url = state_links[state_name]
    output_dir = os.path.join(os.getcwd(), "Data", "building_data_gdb")
    os.makedirs(output_dir, exist_ok=True)

    zip_path = os.path.join(output_dir, f"{state_name}_Structures.zip")

    print(f"Downloading {state_name} ZIP from {url}...")
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        # rmeove zip file once complete
        os.remove(zip_path)
        print(f"Extracted and cleaned up ZIP for {state_name}")
    else:
        raise ValueError(f"Failed to download ZIP for {state_name}. HTTP {response.status_code}")


def gdb_path_by_state(stateid):
    """
    Construct the full path to the GDB file for a given state ID.

    Assumes that the extracted building data is stored under:
    Data/building_data_gdb/[folder_ending_with_stateid]/

    Parameters
    ----------
    stateid : str
        Two-letter state abbreviation (e.g., 'CA', 'TX').

    Returns
    -------
    str
        Full file path to the .gdb file for the given state.

    Raises
    ------
    ValueError
        If no folder matching the state ID is found.
    """
    parent_dir = os.getcwd()
    building_data_dir = os.path.join(parent_dir, "Data", "building_data_gdb")

    # Find folder that ends with the state ID
    folders = [
        f for f in os.listdir(building_data_dir)
        if os.path.isdir(os.path.join(building_data_dir, f))
    ]
    matches = [f for f in folders if f.endswith(stateid)]

    if not matches:
        raise ValueError(f"No folder ending with '{stateid}' found in {building_data_dir}")

    return os.path.join(building_data_dir, matches[0], f"{stateid}_Structures.gdb")


def read_building_data(stateid):
    """
    Load building data for a given state.

    If an aggregated CSV file already exists for the state, returns its type and None.
    Otherwise, reads the raw GDB file and returns a GeoDataFrame with selected columns.

    Parameters
    ----------
    stateid : str
        Two-letter state abbreviation (e.g., 'CA', 'TX').

    Returns
    -------
    tuple
        ('csv', None) if preprocessed CSV exists,
        ('gdb', GeoDataFrame) if GDB needs to be read and processed.
    """
    
    # gdb file path
    building_data_directory = gdb_path_by_state(stateid)

    # get the csv file
    csv_dir = os.path.join("Data", "building_data_csv")
    csv_path = os.path.join(csv_dir, f"{stateid}_building_data.csv")
    if os.path.exists(csv_path):
        # print(f"Aggregated csv file found for {stateid}")
        return 'csv', None
    else:
        print(f"Reading {building_data_directory}")
        # Read the GDB file and return a GeoDataFrame with specified columns
        cols = ['BUILD_ID', 'OCC_CLS', 'PRIM_OCC', 'CENSUSCODE', 'LONGITUDE', 'LATITUDE']
        gdb_df = gpd.read_file(building_data_directory, columns=cols)
        return 'gdb', gdb_df


def remap_occupancy_classes(gdf):
    """
    Remap occupancy class and primary occupancy values in building data.

    Parameters
    ----------
    gdf : GeoDataFrame
        Raw building data read from a state's GDB file, containing at least:
        'BUILD_ID', 'OCC_CLS', 'PRIM_OCC', 'CENSUSCODE', 'LONGITUDE', 'LATITUDE'.

    Returns
    -------
    GeoDataFrame
        A simplified and remapped building dataset with standardized occupancy labels.
    """
    # Select relevant columns
    building_data = gdf[["BUILD_ID", "OCC_CLS", "PRIM_OCC", "CENSUSCODE", "LONGITUDE", "LATITUDE"]].copy()

    # Remap OCC_CLS (if not Residential, other)
    building_data["OCC_CLS"] = building_data["OCC_CLS"].apply(
    lambda x: "RESIDENTIAL" if x == "Residential" else "OTHER"
    )

    # Remap PRIM_OCC
    prim_occ_mapping = {
        val: "OTHER"
        for val in building_data["PRIM_OCC"].unique()
        if val not in ["Single Family Dwelling", "Multi - Family Dwelling"]
    }
    prim_occ_mapping.update({
        "Single Family Dwelling": "SINGLE FAMILY",
        "Multi - Family Dwelling": "MULTI FAMILY"
    })
    building_data["PRIM_OCC"] = building_data["PRIM_OCC"].map(prim_occ_mapping)

    return building_data


def aggregate_building_counts(gdf):
    """
    Aggregate building counts by census tract and occupancy categories.

    Parameters
    ----------
    gdf : GeoDataFrame
        Raw building data from a state's GDB file.

    Returns
    -------
    DataFrame
        Aggregated building counts by CENSUSCODE, OCC_CLS, and PRIM_OCC,
        with one row per unique combination and a 'COUNT' column.
    """
    # Standardize occupancy classes and keep relevant columns
    building_data = remap_occupancy_classes(gdf)

    # Group by tract ID, occupancy class, and primary occupancy
    grouped = (
        building_data
        .groupby(["CENSUSCODE", "OCC_CLS", "PRIM_OCC"])
        .agg(COUNT=("BUILD_ID", "count"))
        .reset_index()
    )
    return grouped


def pivot_building_data(count_building_data):
    """
    Pivot aggregated building counts to wide format by occupancy type.

    Parameters
    ----------
    count_building_data : DataFrame
        Aggregated building counts with columns: CENSUSCODE, OCC_CLS, PRIM_OCC, COUNT

    Returns
    -------
    DataFrame
        Pivoted DataFrame with separate columns for each OCC_CLS / PRIM_OCC pair,
        as well as total residential and total building counts.
    """
    df = count_building_data.copy()

    # Pivot to wide format: one row per CENSUSCODE, one column per (OCC_CLS, PRIM_OCC)
    df_pivot = df.pivot_table(
        index="CENSUSCODE",
        columns=["OCC_CLS", "PRIM_OCC"],
        values="COUNT",
        aggfunc="sum",
        fill_value=0
    )
    # Flatten MultiIndex column names
    df_pivot.columns = [f"{occ_cls}_{prim_occ}" for occ_cls, prim_occ in df_pivot.columns]
    df_pivot = df_pivot.reset_index()
    # Safely sum expected categories (default to 0 if missing)
    residential_cols = [
        "RESIDENTIAL_SINGLE FAMILY",
        "RESIDENTIAL_MULTI FAMILY",
        "RESIDENTIAL_OTHER"]
    other_col = "OTHER_OTHER"

    for col in residential_cols + [other_col]:
        if col not in df_pivot.columns:
            df_pivot[col] = 0

    df_pivot["TOTAL_RESIDENTIAL"] = (
        df_pivot["RESIDENTIAL_SINGLE FAMILY"] +
        df_pivot["RESIDENTIAL_MULTI FAMILY"] +
        df_pivot["RESIDENTIAL_OTHER"]
    )

    df_pivot["TOTAL_BUILDING"] = df_pivot["TOTAL_RESIDENTIAL"] + df_pivot["OTHER_OTHER"]

    return df_pivot


def aggregate_building_data():
    """
    Aggregate building data across all states and save to a single CSV file.

    This function reads all per-state building data CSVs in the 
    'Data/building_data_csv' directory, adds state identifiers, 
    calculates total building counts per row, and outputs a combined 
    'aggregated_building_data.csv'.

    Returns
    -------
    None
        Writes output to disk.
    """
    csv_dir = os.path.join(os.path.dirname(os.getcwd()), "Data", "building_data_csv")
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]

    dfs = []
    for file in csv_files:
        df = pd.read_csv(os.path.join(csv_dir, file))
        stateid = file.split("_")[0]
        df["STATE_ID"] = stateid
        dfs.append(df)

    if not dfs:
        print("No building data CSVs found to aggregate.")
        raise ValueError

    building_data = pd.concat(dfs, ignore_index=True)

    # Drop known irrelevant column if it exists
    building_data = building_data.drop(columns=["OTHER_SINGLE FAMILY"], errors="ignore")

    # Ensure required columns exist
    for col in [
        "OTHER_OTHER", 
        "RESIDENTIAL_MULTI FAMILY", 
        "RESIDENTIAL_OTHER", 
        "RESIDENTIAL_SINGLE FAMILY"
    ]:
        if col not in building_data.columns:
            building_data[col] = 0

    # Compute total building count
    building_data["TOTAL_BUILDING_COUNT"] = (
        building_data["OTHER_OTHER"] +
        building_data["RESIDENTIAL_MULTI FAMILY"] +
        building_data["RESIDENTIAL_OTHER"] +
        building_data["RESIDENTIAL_SINGLE FAMILY"]
    )

    # Save to file
    output_path = os.path.join(csv_dir, "aggregated_building_data.csv")
    building_data.to_csv(output_path, index=False)
    print(f"Saved aggregated building data to {output_path}")
    return None


def o3_get_building_structures():
    """
    Download, extract, process, and aggregate building footprint data for all U.S. states and territories.

    This function performs the full pipeline to prepare building-level structural data from
    the USA_Structures GeoPlatform. It includes:
    - Scraping available state/territory ZIP links
    - Downloading and extracting GDB files for each area (if not already processed)
    - Reading building geometry and occupancy attributes
    - Aggregating building counts per census tract
    - Saving per-state CSVs
    - Merging all state files into a single nationwide CSV (`aggregated_building_data.csv`)

    if a state's processed CSV already exists, it will skip reprocessing that state. 
    Intermediate GDB and CSV files are saved under the `Data/` directory.

    Returns
    -------
    None
        Outputs are written to disk.

    Example
    -------
    >>> from WorkingScripts.o3_building_module import o3_get_building_structures
    >>> o3_get_building_structures()

    # Outputs:
    # - Data/building_data_csv/{STATE_ID}_building_data.csv for each state/territory
    # - Data/building_data_gdb/{STATE}_Structures.gdb folders (raw GDBs)
    # - Data/building_data_csv/aggregated_building_data.csv (final merged file)
    """
    # Full mapping of area names to abbreviations
    STATE_ABBREVIATIONS = {
        "Alabama": "AL", "Alaska": "AK", "American Samoa": "AS", "Arizona": "AZ",
        "Arkansas": "AR", "California": "CA", "Colorado": "CO", "Connecticut": "CT",
        "Delaware": "DE", "D.C.": "DC", "District of Columbia": "DC", "Guam": "GU",
        "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
        "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
        "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
        "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Missouri": "MO",
        "Mississippi": "MS", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
        "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
        "North Carolina": "NC", "North Dakota": "ND", "Northern Mariana Islands": "MP",
        "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
        "Puerto Rico": "PR", "Rhode Island": "RI", "South Carolina": "SC",
        "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
        "Vermont": "VT", "Virgin Islands": "VI", "Virginia": "VA", "Washington": "WA",
        "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
    }

    # print("Fetching state/territory download links...")
    state_links = fetch_state_links()

    csv_dir = os.path.join(os.getcwd(), "Data", "building_data_csv")
    os.makedirs(csv_dir, exist_ok=True)

    # print("Checking and processing available states...")
    for state_name, url in state_links.items():
        stateid = STATE_ABBREVIATIONS.get(state_name)
        if not stateid:
            # print(f"Skipping unknown or unmapped area: {state_name}")
            continue

        csv_path = os.path.join(csv_dir, f"{stateid}_building_data.csv")
        if os.path.exists(csv_path):
            # print(f"{state_name} ({stateid}): CSV already exists, skipping.")
            continue

        try:
            download_and_extract_zip(state_name, state_links)
        except Exception as e:
            # print(f"{state_name}: Failed to download or extract. Error: {e}")
            raise ValueError

    # print("Reading and processing GDBs...")
    for state_name in state_links.keys():
        stateid = STATE_ABBREVIATIONS.get(state_name)
        if not stateid:
            continue

        csv_path = os.path.join(csv_dir, f"{stateid}_building_data.csv")
        if os.path.exists(csv_path):
            # print(f"{state_name} ({stateid}): CSV already exists, skipping.")
            continue

        # print(f"Processing {state_name} ({stateid})...")
        filetype, gdf = read_building_data(stateid)

        if filetype == 'csv':
            # print("CSV exists. Skipping.")
            continue

        count_building_data = aggregate_building_counts(gdf)
        df_pivot = pivot_building_data(count_building_data)

        output_path = os.path.join(csv_dir, f"{stateid}_building_data.csv")
        df_pivot.to_csv(output_path, index=False)
        # print(f"Saved: {output_path}")

    print("Combining all state CSVs into nationwide dataset...")
    aggregate_building_data()
    print("Aggregation complete.")

