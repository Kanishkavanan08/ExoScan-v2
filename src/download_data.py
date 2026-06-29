import os
import numpy as np
import lightkurve as lk
from scipy.interpolate import interp1d

def fetch_and_process_star(target_name):
    print(f"Querying NASA MAST Archive for target: {target_name}...")
    
    try:
        # 1. Search for data collected by the TESS Space Telescope
        search_result = lk.search_lightcurve(target_name, mission="TESS")
        if len(search_result) == 0:
            print(f"No TESS data found for {target_name}.")
            return False
            
        # Download the first available observation sector
        print(f"Downloading observational telemetry telemetry...")
        lc = search_result[0].download()
        
        # 2. Clean the astrophysics data (Remove bad sensor readings & normalize brightness)
        lc_clean = lc.remove_nans().normalize()
        raw_flux = lc_clean.flux.value  # Extracts raw numeric light values
        
        print(f"Raw data fetched successfully ({len(raw_flux)} data points).")
        
        # 3. Resample the data using linear interpolation to fit the 1D CNN input size (2000)
        current_indices = np.linspace(0, len(raw_flux) - 1, num=len(raw_flux))
        target_indices = np.linspace(0, len(raw_flux) - 1, num=2000)
        
        interpolator = interp1d(current_indices, raw_flux, kind='linear')
        ml_ready_flux = interpolator(target_indices)
        
        # 4. Save the actual processed NASA data array to disk
        os.makedirs(os.path.join("data", "processed"), exist_ok=True)
        save_path = os.path.join("data", "processed", f"{target_name}_ml_ready.npy")
        np.save(save_path, ml_ready_flux)
        
        print(f"Success! Real data processed and cached at: {save_path}\n")
        return True
        
    except Exception as e:
        print(f"Failed to fetch live NASA telemetry. Error: {e}")
        return False

if __name__ == "__main__":
    # Test download capability on the real star TOI-700
    fetch_and_process_star("TOI-700")