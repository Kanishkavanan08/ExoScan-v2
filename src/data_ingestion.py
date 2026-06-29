import os
import pandas as pd
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
import lightkurve as lk

def fetch_confirmed_tess_planets():
    print("Querying NASA Exoplanet Archive for confirmed TESS planets...")
    
    # Query the Planetary Systems composite parameters table
    query_result = NasaExoplanetArchive.query_criteria(
        table="pscomppars", 
        select="pl_name, hostname, pl_rade, pl_orbper, discoverymethod", 
        where="disc_facility like '%TESS%'"
    )
    
    # Convert to a Pandas DataFrame for easier handling
    df = query_result.to_pandas()
    print(f"Successfully retrieved {len(df)} confirmed TESS planets.")
    
    # Save the raw tabular data to our data folder
    output_path = os.path.join("data", "raw", "tess_confirmed_planets.csv")
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    
    return df

def download_lightcurve_data(target_name):
    print(f"\nSearching for raw light curve data for {target_name}...")
    
    # Search the MAST archive for the target's light curve
    search_result = lk.search_lightcurve(target_name, author="SPOC", exptime=120)
    
    if len(search_result) > 0:
        print(f"Found {len(search_result)} observation quarters. Downloading the first one...")
        # Download the first light curve file
        lc = search_result[0].download()
        print("Download complete! Here is a summary of the light curve:")
        print(lc)
    else:
        print("No high-cadence light curve data found for this target.")

if __name__ == "__main__":
    # 1. Fetch the tabular data
    tess_data = fetch_confirmed_tess_planets()
    
    # 2. Pick a famous TESS discovery to download the raw time-series data for
    # TOI-700 is a well-known system with multiple planets, including one in the habitable zone.
    download_lightcurve_data("TOI-700")