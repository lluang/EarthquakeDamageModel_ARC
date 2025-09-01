"""
ShakeMap Retrieval and Processing Module

This module provides functions to:
- Fetch earthquake event data from the USGS GeoJSON feed using an event ID
- Parse key event metadata including magnitude, coordinates, depth, and URL
- Download and extract ShakeMap shapefiles (mi, pga, pgv) for qualifying events
- Save relevant metadata in a structured format for further analysis
"""

import requests
import json
import os
import zipfile
import io
import datetime
import geopandas as gpd

SHAKEMAP_DIR = "{}/Data/Shakemap".format(os.getcwd())
FEEDURL = "https://earthquake.usgs.gov/fdsnws/event/1/query.geojson?eventid={}"

def fetch_earthquake_data(feed_url):
    """
    Fetch earthquake data from a specified USGS GeoJSON feed URL.
    
    --Parameters
    feed_url : str
        A fully formatted USGS GeoJSON feed URL, typically constructed
        using an earthquake event ID.
    --Returns
    dict or None
        Parsed JSON response from USGS as a dictionary if successful.
        Returns None if the request fails.
    --Raises
    ValueError
        If the request to the USGS feed fails or returns an error status code.

    Example
    -------
    >>> feed_url = FEEDURL.format("us7000dflf")
    >>> event_json = fetch_earthquake_data(feed_url)
    """
    try:
        response = requests.get(feed_url, timeout=10)  # 10s timeout to avoid hangs
        response.raise_for_status()  # Raise HTTPError for 4xx/5xx responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {feed_url}: {e}")
        print("Check that the USGS event ID is correct and try again.")
        raise ValueError("Failed to fetch earthquake data.")

def retrieve_event_data(event_json):
    """
    Parse relevant metadata from a USGS earthquake event GeoJSON dictionary.
    --Parameters
    event_json : dict
        A dictionary containing the full GeoJSON response for a single earthquake
        event from the USGS feed.
    --Returns
    dict
        A dictionary containing parsed event metadata including:
        - event ID, magnitude, location, depth, time, place, URL, and ShakeMap URL.
    --Raises
    ValueError
        If the ShakeMap product is not available in the event data.

        
    Example
    -------   
    >>> event = retrieve_event_data(event_json)
    >>> print(event["magnitude"], event["shakemap_url"])
    """
    event_id = event_json["id"]
    magnitude = event_json["properties"]["mag"]
    # Check if ShakeMap product is available
    if "shakemap" not in event_json["properties"]["products"]:
        print(f"Issue: {event_id}: No ShakeMap available.")
        print("Please check the event ID and try again.")
        raise ValueError("ShakeMap not available.")
    # Extract coordinates: [longitude, latitude, depth]
    lon, lat, depth = event_json["geometry"]["coordinates"]
    # Build metadata dictionary
    event_data = {
        "eventid": event_id,
        "magnitude": magnitude,
        "lon": lon,
        "lat": lat,
        "depth": depth,
        "title": event_json["properties"]["title"],
        "time": event_json["properties"]["time"],
        "place": event_json["properties"]["place"],
        "url": event_json["properties"]["url"],
        "shakemap_url": event_json["properties"]["products"]["shakemap"][0]
                            ["contents"]["download/shape.zip"]["url"]}
    return event_data


def download_and_extract_shakemap(event):
    """
    Download and extract the ShakeMap ZIP archive for a given earthquake event.
    This function checks whether the ShakeMap for the event already exists locally.
    If not, it downloads the ZIP file from the provided URL and extracts it into
    a dedicated event folder.

    --Parameters
    event : dict
        Dictionary containing earthquake event metadata. Must include:
        - 'eventid': str
        - 'shakemap_url': str (URL pointing to downloadable ShakeMap ZIP)

    --Returns
    str
        Path to the extracted ShakeMap folder.

    --Raises
    ValueError
        If the download or ZIP extraction fails.

    Example
    -------
    >>> event_folder = download_and_extract_shakemap(event)
    >>> print("Files extracted to:", event_folder)
    """
    event_id = event["eventid"]
    shakemap_url = event["shakemap_url"]
    event_folder = os.path.join(SHAKEMAP_DIR, event_id)

    # Skip if ShakeMap already downloaded (no overwrite)
    if os.path.exists(event_folder):
        print(f"{event_id}: ShakeMap already downloaded.")
        return event_folder

    os.makedirs(event_folder, exist_ok=True)
    zip_path = os.path.join(event_folder, "shakemap.zip")

    # Step 1: Download ZIP
    try:
        print(f"Downloading ShakeMap for event {event_id}...")
        response = requests.get(shakemap_url, stream=True, timeout=15)
        response.raise_for_status()

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading ShakeMap for {event_id}: {e}")
        raise ValueError("ShakeMap download failed.")

    # Step 2: Extract ZIP
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(event_folder)
        print(f"Extracted ShakeMap for event {event_id} to {event_folder}")
    except zipfile.BadZipFile:
        print(f"Error: Invalid ZIP file for event {event_id}")
        raise ValueError("Invalid ZIP archive.")

    # Step 3: Remove ZIP after extraction
    os.remove(zip_path)
    print(f"Success! ShakeMap for event {event_id} is ready.")

    return event_folder




