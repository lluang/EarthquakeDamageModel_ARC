"""
Download and merge 2024 TIGER/Line Census Tract shapefiles into a single GeoPackage.

This script:
- Scrapes ZIP file links from the 2024 Census Tract TIGER/Line directory
- Downloads and extracts all shapefiles
- Merges them into a nationwide GeoPackage for spatial analysis
- Cleans up intermediate ZIPs and extracted files
"""

import os
import glob
import zipfile
import requests
import pandas as pd
import geopandas as gpd
from bs4 import BeautifulSoup

# Base URL to the 2024 Census Tract TIGER/Line shapefiles
BASE_URL = "https://www2.census.gov/geo/tiger/TIGER2024/TRACT/"

def download_census():
    """
    Download, extract, and merge 2024 Census Tract shapefiles into a single GeoPackage.

    If the output GeoPackage already exists, the function does nothing.
    Otherwise, it:
    - Scrapes ZIP links from the Census TIGER/Line site (if ZIPs not already present)
    - Downloads and extracts the shapefiles
    - Merges them into one GeoPackage layer ("tracts")
    - Deletes all intermediate ZIPs and shapefiles

    Returns
    -------
    None

     Example
    -------
    >>> from WorkingScripts.o2_census_download import download_census
    >>> download_census()
    >>> # Output: GeoPackage saved to ./Data/merged_shapefile/Nationwide_Tracts.gpkg
    """
    # Define local directories
    download_folder = os.path.join(os.getcwd(), "Data", "census_shp")
    extracted_folder = os.path.join(os.getcwd(), "Data", "extracted_census_shp")
    merged_shapefile_folder = os.path.join(os.getcwd(), "Data", "merged_shapefile")
    output_gpkg = os.path.join(merged_shapefile_folder, "Nationwide_Tracts.gpkg")

    # Ensure directories exist
    os.makedirs(download_folder, exist_ok=True)
    os.makedirs(extracted_folder, exist_ok=True)
    os.makedirs(merged_shapefile_folder, exist_ok=True)

    if not os.path.isfile(output_gpkg):
        print(f"No existing GeoPackage found at {output_gpkg}.")

        zip_files = glob.glob(os.path.join(download_folder, "*.zip"))
        if zip_files:
            print(f"Found {len(zip_files)} existing ZIP files in {download_folder}. Skipping download.")
        else:
            # Step 1: Scrape ZIP file links
            print(f"Connecting to {BASE_URL}...")
            response = requests.get(BASE_URL)
            if response.status_code != 200:
                raise ValueError(f"Failed to access {BASE_URL} (HTTP {response.status_code})")

            soup = BeautifulSoup(response.text, "html.parser")
            zip_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.zip')]
            print(f"Found {len(zip_links)} ZIP files. Starting download...")

            # Step 2: Download ZIP files
            for zip_file in zip_links:
                file_url = BASE_URL + zip_file
                local_path = os.path.join(download_folder, zip_file)

                file_response = requests.get(file_url, stream=True)
                if file_response.status_code == 200:
                    with open(local_path, 'wb') as f:
                        for chunk in file_response.iter_content(chunk_size=1024):
                            f.write(chunk)
                else:
                    raise ValueError(f"Failed to download {zip_file} (HTTP {file_response.status_code})")

        # Step 3: Extract ZIP files
        print("Extracting ZIP files...")
        for zip_file in os.listdir(download_folder):
            if zip_file.endswith(".zip"):
                with zipfile.ZipFile(os.path.join(download_folder, zip_file), 'r') as zip_ref:
                    zip_ref.extractall(extracted_folder)

        # Step 4: Merge all shapefiles
        print("Merging shapefiles...")
        shapefiles = glob.glob(os.path.join(extracted_folder, "**", "*.shp"), recursive=True)
        if not shapefiles:
            raise ValueError("No shapefiles found to merge.")

        gdf_list = []
        for shp in shapefiles:
            gdf = gpd.read_file(shp)
            gdf_list.append(gdf)

        merged_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=gdf_list[0].crs)
        merged_gdf.to_file(output_gpkg, layer="tracts", driver="GPKG")
        print(f"Saved merged Census Tracts to: {output_gpkg}")

    else:
        print(f"GeoPackage already exists at: {output_gpkg}")

    # Step 5: Cleanup intermediate files
    print("Cleaning up intermediate files...")
    for file in glob.glob(os.path.join(download_folder, "*.zip")):
        os.remove(file)

    for file in glob.glob(os.path.join(extracted_folder, "*")):
        if os.path.isfile(file):
            os.remove(file)

    print("Census shapefile download and merge complete.")
    return None
