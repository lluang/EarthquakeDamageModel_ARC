"""
ShakeMap and Census Tract Geospatial Processing

This module provides functions to:
- Load and manage ShakeMap shapefiles (mi, pga, pgv) for earthquake events
- Locate and download Census Tract boundaries (2019 TIGER/Line shapefiles)
- Clip ShakeMap layers to census tract geometries
- Compute max, min, and mean earthquake intensity statistics for each tract

Data Sources:
- Census Tract Shapefile: tl_2019_us_tract.shp
    - Represents small geographic areas (~2,500–8,000 people)
- County Shapefile: tl_2019_us_county.shp
    - Larger administrative units, e.g., counties, parishes, boroughs
"""


import os
import requests
import geopandas as gpd
from pyogrio.errors import DataSourceError 

def get_shakemap_files(eventdir):
    """
    Construct file paths for ShakeMap shapefiles in a given event directory.
    --Parameters
    eventdir : str
        Path to the ShakeMap event directory, which should contain
        the shapefiles for Modified Mercalli Intensity (mi), 
        Peak Ground Velocity (pgv), and Peak Ground Acceleration (pga).
    --Returns
    tuple of str
        File paths to 'mi.shp', 'pgv.shp', and 'pga.shp'.
    """
    mi = os.path.join(eventdir, "mi.shp")
    pgv = os.path.join(eventdir, "pgv.shp")
    pga = os.path.join(eventdir, "pga.shp")
    return mi, pgv, pga

def get_shakemap_dir():
    """
    Determine the directory path for storing ShakeMap files.
    The function checks if a "ShakeMaps" folder exists in the parent
    directory of the current working directory. If it does not exist,
    the folder is created.
    --Returns
    str
        Absolute path to the "ShakeMaps" directory.
    """
    parent_dir = os.path.dirname(os.getcwd())
    shakemap_dir = os.path.join(parent_dir, "ShakeMaps")

    if not os.path.exists(shakemap_dir):
        os.mkdir(shakemap_dir)

    return shakemap_dir


def download_census_tracts(data_dir):
    """
    Checks for the existence of the 2019 Census Tract shapefile in GeoPackage format.

    This function does not actually download the data — it only verifies whether
    the expected output file exists. If not, it raises an error to prompt download elsewhere.
    --Parameters
    data_dir : str
        Path to the base directory where shapefiles or GeoPackages are stored.
    --Returns
    str
        Path to the existing 'Nationwide_Tracts.gpkg' GeoPackage.
    --Raises
    ValueError
        If the Census Tract GeoPackage does not exist in the expected location.
    """
    extract_path = os.path.join(data_dir, "merged_shapefile", "Nationwide_Tracts.gpkg")

    if os.path.exists(extract_path):
        print(f"Census tracts shapefile already exists at {extract_path}")
        return extract_path
    else:
        print(f"Census tracts shapefile not found at {extract_path}")
        raise ValueError("Census tract shapefile is missing. Please download it manually.")


def clip_shakemap_to_tracts(shakemap_gdf, tracts_gdf, output_layer, GPKG_PATH):
    """
    Clip a ShakeMap layer to the boundaries of census tracts.

    This function overlays the ShakeMap geometries (e.g., PGA, PGV, or MMI)
    onto census tract geometries and retains only intersecting areas.
    The result is saved as a new layer in the specified GeoPackage.

    Parameters
    ----------
    shakemap_gdf : GeoDataFrame
        ShakeMap layer (already loaded from file).
    tracts_gdf : GeoDataFrame
        Census tract geometries (Nationwide_Tracts.gpkg).
    output_layer : str
        Name for the output layer in the GeoPackage.
    GPKG_PATH : str
        Path to the output GeoPackage file.

    Returns
    -------
    GeoDataFrame
        The clipped result as a new GeoDataFrame.
    """
    # Ensure both datasets are in WGS84 (EPSG:4326)
    shakemap_gdf = shakemap_gdf.to_crs(epsg=4326)
    tracts_gdf = tracts_gdf.to_crs(epsg=4326)
    # Perform spatial intersection (clip)
    clipped_gdf = gpd.overlay(shakemap_gdf, tracts_gdf, how="intersection")
    # Save clipped layer to GeoPackage
    clipped_gdf.to_file(GPKG_PATH, layer=output_layer, driver="GPKG", mode="w")
    print(f"Saved {output_layer} (clipped ShakeMap to tracts) to {GPKG_PATH}")
    return clipped_gdf


def calculate_shakemap_statistics(shakemap_gdf, tracts_gdf, output_layer, GPKG_PATH):
    """
    Calculate summary statistics (max, min, mean) of ShakeMap intensity per census tract.
    This function joins the clipped ShakeMap data to census tracts, computes
    intensity statistics (e.g., for PGA), and saves the output as a new layer
    in a GeoPackage.

    --Parameters
    shakemap_gdf : GeoDataFrame
        Clipped ShakeMap data (result of spatial intersection).
    tracts_gdf : GeoDataFrame
        Census tract geometries.
    output_layer : str
        Layer name for output in the GeoPackage.
    GPKG_PATH : str
        Path to the GeoPackage to store results.
    --Returns
    None
        The function writes output to disk and returns nothing.
    """
    # Ensure consistent CRS
    shakemap_gdf = shakemap_gdf.to_crs(epsg=4326)
    tracts_gdf = tracts_gdf.to_crs(epsg=4326)
    # Spatial join: attach ShakeMap intensity to tracts
    joined = gpd.sjoin(tracts_gdf, shakemap_gdf, how="inner", predicate="intersects")
    # Remove duplicate columns from join
    right_cols = [col for col in joined.columns if "_right" in col]
    joined = joined.drop(columns=right_cols)
    # Clean up column names
    joined.columns = [col.replace("_left", "") for col in joined.columns]
    # Aggregate intensity stats by tract
    aggregated = joined.groupby("GEOID").agg({
        "PARAMVALUE": ["max", "min", "mean"],
        "geometry": "first"
    }).reset_index()

    # Flatten MultiIndex columns
    aggregated.columns = ["GEOID", "max_intensity", "min_intensity", "mean_intensity", "geometry"]

    # Create final GeoDataFrame
    result = gpd.GeoDataFrame(aggregated, geometry="geometry", crs=tracts_gdf.crs)

    # Save to GeoPackage
    result.to_file(GPKG_PATH, layer=output_layer, driver="GPKG", mode="w")
    print(f"Saved {output_layer} (tract-level ShakeMap statistics) to {GPKG_PATH}")
    return None


def shakemap_into_census_geo(eventdir):
    """
    Process a ShakeMap event by clipping its data to census tracts and computing tract-level statistics.

    This function performs the following steps:
    1. Loads the PGA ShakeMap shapefile from a given event directory.
    2. Loads the nationwide 2019 Census Tract geometries.
    3. Clips the ShakeMap layer to tract boundaries.
    4. Computes max, min, and mean ShakeMap intensity values for each tract.
    5. Saves both outputs to a GeoPackage in the event directory.

    --Parameters
    eventdir : str
        Path to the directory containing ShakeMap shapefiles (mi, pga, pgv)
        and where output GeoPackage will be saved.
    --Returns
    None
        All outputs are written to disk.

    Example
    -------
    >>> from WorkingScripts.o2_census_intersect import shakemap_into_census_geo
    >>> shakemap_into_census_geo(eventdir = "./Data/Shakemap/us70006vll")

    # Outputs:
    # - eqmodel_outputs.gpkg
    #     - 'shakemap_tractclip_pga': ShakeMap geometries clipped to census tracts
    #     - 'tract_shakemap_pga': max, min, and mean PGA values per tract
    """
    # Load ShakeMap shapefiles (only PGA used here)
    _, _, pga_path = get_shakemap_files(eventdir)
    try:
        pga_gdf = gpd.read_file(pga_path)
    except DataSourceError:
        print("\n\n\n ============= USER MESSAGE ========================")
        print(f"Failed to read shapefile at {pga_path}: file not found - please check that shakemap is available on USGS")
        print("=====================================\n\n\n")
        raise ValueError

    # Load Census Tract geometries
    data_dir = os.path.join(os.getcwd(), "Data")
    tracts_path = os.path.join(data_dir, "merged_shapefile", "Nationwide_Tracts.gpkg")
    tracts_gdf = gpd.read_file(tracts_path)

    # Define output GeoPackage path
    gpkg_path = os.path.join(eventdir, "eqmodel_outputs.gpkg")

    # Clip ShakeMap to tract boundaries
    clipped = clip_shakemap_to_tracts(pga_gdf, tracts_gdf, "shakemap_tractclip_pga", gpkg_path)

    # Compute and save tract-level summary statistics
    calculate_shakemap_statistics(clipped, tracts_gdf, "tract_shakemap_pga", gpkg_path)

    return None