import os
import numpy as np
import lightkurve as lk
from scipy.interpolate import interp1d

def preprocess_lightcurve(target_name, fixed_length=2000):
    print(f"Preprocessing {target_name} for Machine Learning...")
    
    # 1. Fetch the raw data
    search_result = lk.search_lightcurve(target_name, author="SPOC", exptime=120)
    if len(search_result) == 0:
        print("No data found.")
        return
        
    lc = search_result[0].download()
    
    # 2. Clean and Flatten 
    # remove_nans() and remove_outliers() strip bad data. 
    # flatten() removes long-term stellar trends so we only see planetary transits.
    clean_lc = lc.remove_nans().remove_outliers().flatten(window_length=401)
    
    time = clean_lc.time.value
    flux = clean_lc.flux.value
    
    # 3. Interpolate to a Fixed Length
    # AI models require identical input sizes. We scale all curves to exactly 2000 points.
    time_new = np.linspace(time.min(), time.max(), fixed_length)
    interpolator = interp1d(time, flux, kind='linear', fill_value="extrapolate")
    flux_fixed = interpolator(time_new)
    
    # 4. Normalize the data between 0 and 1
    # Neural Networks converge faster when data is scaled uniformly.
    flux_normalized = (flux_fixed - np.min(flux_fixed)) / (np.max(flux_fixed) - np.min(flux_fixed))
    
    # 5. Save the array as a binary .npy file
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    output_path = os.path.join("data", "processed", f"{target_name}_ml_ready.npy")
    np.save(output_path, flux_normalized)
    
    print(f"Success! Preprocessed vector saved to: {output_path}")
    print(f"Final array shape: {flux_normalized.shape}")

if __name__ == "__main__":
    preprocess_lightcurve("TOI-700")