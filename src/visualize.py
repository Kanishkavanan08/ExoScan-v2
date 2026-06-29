import os
import numpy as np
import matplotlib.pyplot as plt

def plot_transit_detection(target_name, probability_score):
    print(f"Generating transit visualization for {target_name}...")
    
    # 1. Load the preprocessed numpy data array
    data_path = os.path.join("data", "processed", f"{target_name}_ml_ready.npy")
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
        
    light_curve = np.load(data_path)
    time_steps = np.arange(len(light_curve))
    
    # 2. Initialize the plot
    plt.figure(figsize=(12, 5))
    plt.plot(time_steps, light_curve, color='#1f77b4', label='Normalized Flux (Starlight)', alpha=0.8)
    
    # 3. If the CNN model flagged a high-probability transit, highlight the expected dip area
    if probability_score >= 0.5:
        # For our mock/synthetic data window, the transit dip typically falls between 800 and 1150
        plt.axvspan(800, 1150, color='red', alpha=0.15, label=f'CNN Detected Transit Zone ({probability_score*100:.1f}%)')
        plt.title(f"ExoScan Transit Detection Pipeline - {target_name} (POSSIBLE EXOPLANET)", fontsize=14, pad=15)
    else:
        plt.title(f"ExoScan Transit Detection Pipeline - {target_name} (STELLAR NOISE)", fontsize=14, pad=15)
        
    # 4. Stylize the chart
    plt.xlabel("Time Bin / Data Points", fontsize=11)
    plt.ylabel("Relative Flux Brightness", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower left')
    
    # 5. Create output directory and save the chart
    os.makedirs("results", exist_ok=True)
    save_path = os.path.join("results", f"{target_name}_transit_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visual graph successfully saved to: {save_path}\n")

if __name__ == "__main__":
    # Test run the plotting module using our target file
    plot_transit_detection("TOI-700", 0.9412)