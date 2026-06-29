import os
import lightkurve as lk
import matplotlib.pyplot as plt

def visualize_lightcurve(target_name):
    print(f"Fetching light curve for {target_name}...")
    
    # Search for high-cadence data (2-minute exposures)
    search_result = lk.search_lightcurve(target_name, author="SPOC", exptime=120)
    
    if len(search_result) > 0:
        lc = search_result[0].download()
        
        # Clean the data: remove empty values and extreme outliers to make the graph readable
        clean_lc = lc.remove_nans().remove_outliers()
        
        # Plot using Lightkurve's built-in matplotlib integration
        ax = clean_lc.plot(color='black', alpha=0.5, label='Normalized Flux')
        ax.set_title(f"Raw Light Curve Time-Series: {target_name}")
        
        # Ensure the output directory exists
        os.makedirs(os.path.join("data", "processed"), exist_ok=True)
        
        # Save the figure to your processed data folder
        output_path = os.path.join("data", "processed", f"{target_name}_lightcurve.png")
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Visualization successfully saved to: {output_path}")
        
        # Display the interactive plot window on your screen
        plt.show()
    else:
        print("No light curve data found for this target.")

if __name__ == "__main__":
    visualize_lightcurve("TOI-700")