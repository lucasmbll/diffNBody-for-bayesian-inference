import sys
import argparse
import multiprocessing
import os
import subprocess
import yaml
from run_experiments import main as run_single_experiment

def run_experiment(config_path):
    # Assumes run_experiments.py is in the same directory
    script_path = os.path.join(os.path.dirname(__file__), "run_experiments.py")
    result = subprocess.run(
        [sys.executable, script_path, "--config", config_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    print(f"Experiment {config_path} finished with return code {result.returncode}")
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

def run_multiple_experiments(config_dir, pattern="*.yaml"):
    """
    Run experiments for all config files in a directory.
    
    Parameters:
    -----------
    config_dir : str
        Directory containing config files
    pattern : str
        File pattern to match (default: "*.yaml")
    """
    import glob
    
    config_files = glob.glob(os.path.join(config_dir, pattern))
    
    if not config_files:
        print(f"No config files found in {config_dir} matching pattern {pattern}")
        return
    
    print(f"Found {len(config_files)} config files to process:")
    for config_file in config_files:
        print(f"  - {config_file}")
    
    for i, config_file in enumerate(config_files):
        print(f"\n{'='*60}")
        print(f"Processing experiment {i+1}/{len(config_files)}: {os.path.basename(config_file)}")
        print(f"{'='*60}")
        
        try:
            run_single_experiment(config_file)
            print(f"✓ Successfully completed experiment: {os.path.basename(config_file)}")
        except Exception as e:
            print(f"✗ Error in experiment {os.path.basename(config_file)}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(
        description="Run up to 4 experiments in parallel, each with its own YAML config."
    )
    parser.add_argument(
        "configs",
        nargs="+",
        help="Paths to YAML config files (max 4)."
    )
    parser.add_argument("--config_dir", type=str, required=False, 
                       help="Directory containing YAML config files")
    parser.add_argument("--pattern", type=str, default="*.yaml",
                       help="File pattern to match (default: *.yaml)")
    
    args = parser.parse_args()
    configs = args.configs
    if len(configs) > 4:
        print("Error: Maximum 4 config files allowed.")
        sys.exit(1)

    processes = []
    for config_path in configs:
        p = multiprocessing.Process(target=run_experiment, args=(config_path,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    if args.config_dir:
        run_multiple_experiments(args.config_dir, args.pattern)

if __name__ == "__main__":
    main()
