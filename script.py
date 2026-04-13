import os
import argparse
import subprocess
import sys

def run_command(command, env=None):
    """Utility function to run shell commands and handle errors."""
    print(f"\n[INFO] Executing: {' '.join(command)}\n")
    # Pass the current environment and add the local directory to PYTHONPATH
    current_env = os.environ.copy()
    if env:
        current_env.update(env)
    current_env["PYTHONPATH"] = os.getcwd()
    conda_prefix = current_env.get("CONDA_PREFIX", "")
    if conda_prefix:
        current_env["CUDA_HOME"] = conda_prefix
        lib_path = os.path.join(conda_prefix, "lib")
        current_env["LD_LIBRARY_PATH"] = f"{lib_path}:{current_env.get('LD_LIBRARY_PATH', '')}"
    result = subprocess.run(command, env=current_env, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with return code: {result.returncode}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="One-click Pipeline for Cosmos Diffusion Renderer")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the directory containing input .mp4 videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Path where the G-buffer results will be saved")
    parser.add_argument("--max_frames", type=int, default=57, help="Maximum number of frames to process (default: 57)")
    parser.add_argument("--resize", type=str, default="1280x704", help="Resolution to resize frames (default: 1280x704)")
    
    args = parser.parse_args()

    # Define intermediate path for extracted frames
    temp_frames_dir = os.path.join(args.output_dir, "intermediate_frames")
    
    # Ensure directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(temp_frames_dir, exist_ok=True)

    # --- STEP 1: Extract Frames ---
    print("Step 1/2: Extracting frames from videos...")
    extract_cmd = [
        "python", "scripts/dataproc_extract_frames_from_video.py",
        "--input_folder", args.input_dir,
        "--output_folder", temp_frames_dir,
        "--frame_rate", "24",
        "--resize", args.resize,
        "--max_frames", str(args.max_frames)
    ]
    run_command(extract_cmd)

    # --- STEP 2: Inverse Rendering (G-Buffer Extraction) ---
    print("Step 2/2: Running Inverse Rendering Inference...")
    inference_cmd = [
        "python", "cosmos_predict1/diffusion/inference/inference_inverse_renderer.py",
        "--checkpoint_dir", "checkpoints",
        "--diffusion_transformer_dir", "Diffusion_Renderer_Inverse_Cosmos_7B",
        "--dataset_path", temp_frames_dir,
        "--num_video_frames", str(args.max_frames),
        "--group_mode", "folder",
        "--video_save_folder", args.output_dir
    ]
    
    run_command(inference_cmd)

    print(f"\n[SUCCESS] Pipeline completed. Results are located in: {args.output_dir}")

if __name__ == "__main__":
    main()