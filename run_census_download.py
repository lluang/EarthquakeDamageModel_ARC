"""
Download and merge 2024 TIGER/Line Census Tract shapefiles into a single GeoPackage.

This script is a convenience wrapper that executes the download and merge process.

Example:
    To run from the command line:
    $ python run_census_download.py
"""
from WorkingScripts.o2_download_census import download_census

if __name__ == "__main__":
    print("Starting the download of census data...")
    download_census()
    print("Census data download process has finished.")
