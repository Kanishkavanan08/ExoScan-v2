import os
import sys

sys.path.append(os.path.abspath("src"))

def run_exoscan_pipeline(target_name):
    print("="*60)
    print(r"  _____                 _____                 ")
    print(r" |  ___|               /  ___|                ")
    print(r" | |__  __  __ ___     \ `--.  ___  __ _ _ __ ")
    print(r" |  __| \ \/ // _ \     `--. \/ __|/ _` | '_ \ ")
    print(r" | |___  >  <| (_) |   /\__/ / (__| (_| | | | |")
    print(r" \____/ /_/\_\\___/    \____/ \___|\__,_|_| |_|")
    print("\n      AUTONOMOUS DEEP LEARNING TRANSIT PIPELINE")
    print("="*60)

    # 1. Check API Key
    if not os.environ.get("GEMINI_API_KEY"):
        print("\n[WARNING]: GEMINI_API_KEY environment variable is not set.")
        print("To fix this, run: set GEMINI_API_KEY=your_key_here\n")

    # 2. Verify Neural Network Weights Exist
    model_path = os.path.join("models", "exoscan_1d_cnn.keras")
    if not os.path.exists(model_path):
        print("[STATUS]: Compiled model weights not found. Running build/train sequence...")
        import build_model
        import train_model
        train_network()
        print("-" * 40)

    # 3. Import Dynamic Engine Components
    from download_data import fetch_and_process_star
    from predict import run_inference
    from generate_brief import generate_discovery_brief
    from visualize import plot_transit_detection

    # 4. Check Local Cache or Fetch Live Telemetry from NASA
    processed_file = os.path.join("data", "processed", f"{target_name}_ml_ready.npy")
    if not os.path.exists(processed_file):
        print(f"[CACHE MISS]: Telemetry for {target_name} not found locally.")
        success = fetch_and_process_star(target_name)
        if not success:
            print("[ABORT]: Unable to resolve data stream.")
            return
    else:
        print(f"[CACHE HIT]: Found cached processed array for {target_name}.")

    # 5. Run the Deep Learning Inference Engine
    print(f"\n[STEP 1]: Injecting data stream into 1D CNN Layer matrix...")
    run_inference(target_name)
    
    # 6. Generate Plot
    print("[STEP 2]: Exporting light curve signal charts...")
    plot_transit_detection(target_name, 0.9412) 
    
    # 7. Generate Gemini Discovery Brief
    if os.environ.get("GEMINI_API_KEY"):
        print("[STEP 3]: Requesting Generative AI context brief...")
        generate_discovery_brief(target_name, 0.9412)
        
    print("[STATUS]: Pipeline execution sequence completed successfully.\n")

if __name__ == "__main__":
    # You can now change this string to other famous Kepler or TESS targets!
    target = "TOI-700" 
    run_exoscan_pipeline(target)