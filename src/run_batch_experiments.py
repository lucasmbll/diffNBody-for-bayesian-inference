import os
import sys
import argparse
import time
import datetime
import yaml
import traceback
from pathlib import Path

def run_single_experiment(config_path, experiment_id, total_experiments):
    """
    Run a single experiment and handle errors gracefully.
    
    Parameters:
    -----------
    config_path : str
        Path to the configuration file
    experiment_id : int
        Current experiment number (1-indexed)
    total_experiments : int
        Total number of experiments
        
    Returns:
    --------
    success : bool
        Whether the experiment completed successfully
    runtime : float
        Runtime in seconds
    error_msg : str or None
        Error message if failed, None if successful
    """
    print("\n" + "="*80)
    print(f"STARTING EXPERIMENT {experiment_id}/{total_experiments}")
    print(f"Config: {config_path}")
    print(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # Import here to avoid issues with CUDA device setting
        from run_experiments import main, set_cuda_device_from_config
        
        # Set CUDA device from config
        set_cuda_device_from_config(config_path)
        
        # Run the experiment
        main(config_path)
        
        end_time = time.time()
        runtime = end_time - start_time
        
        print("\n" + "="*80)
        print(f"EXPERIMENT {experiment_id}/{total_experiments} COMPLETED SUCCESSFULLY")
        print(f"Runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
        print("="*80)
        
        return True, runtime, None
        
    except Exception as e:
        end_time = time.time()
        runtime = end_time - start_time
        error_msg = str(e)
        
        print("\n" + "="*80)
        print(f"EXPERIMENT {experiment_id}/{total_experiments} FAILED")
        print(f"Runtime: {runtime:.2f} seconds")
        print(f"Error: {error_msg}")
        print("="*80)
        print("Full traceback:")
        traceback.print_exc()
        print("="*80)
        
        return False, runtime, error_msg

def validate_config_files(config_paths):
    """
    Validate that all config files exist and are readable.
    
    Parameters:
    -----------
    config_paths : list
        List of config file paths
        
    Returns:
    --------
    valid_configs : list
        List of valid config file paths
    """
    valid_configs = []
    invalid_configs = []
    
    for config_path in config_paths:
        if not os.path.exists(config_path):
            invalid_configs.append(f"{config_path} - File not found")
            continue
            
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Basic validation
            if 'mode' not in config:
                invalid_configs.append(f"{config_path} - Missing 'mode' field")
                continue
                
            if config.get('mode') != 'sampling':
                invalid_configs.append(f"{config_path} - Mode is not 'sampling' (found: {config.get('mode')})")
                continue
                
            valid_configs.append(config_path)
            
        except yaml.YAMLError as e:
            invalid_configs.append(f"{config_path} - YAML parsing error: {str(e)}")
        except Exception as e:
            invalid_configs.append(f"{config_path} - Validation error: {str(e)}")
    
    if invalid_configs:
        print("Invalid configuration files found:")
        for invalid in invalid_configs:
            print(f"  ❌ {invalid}")
        print()
    
    if valid_configs:
        print("Valid configuration files:")
        for valid in valid_configs:
            print(f"  ✅ {valid}")
        print()
    
    return valid_configs

def save_batch_summary(results, output_file):
    """
    Save a summary of the batch run results.
    
    Parameters:
    -----------
    results : list
        List of (config_path, success, runtime, error_msg) tuples
    output_file : str
        Path to save the summary
    """
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    total_experiments = len(results)
    successful = sum(1 for _, success, _, _ in results if success)
    failed = total_experiments - successful
    total_runtime = sum(runtime for _, _, runtime, _ in results)
    
    summary = {
        'batch_info': {
            'timestamp': datetime.datetime.now().isoformat(),
            'total_experiments': total_experiments,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total_experiments if total_experiments > 0 else 0,
            'total_runtime_seconds': total_runtime,
            'total_runtime_hours': total_runtime / 3600,
            'average_runtime_per_experiment': total_runtime / total_experiments if total_experiments > 0 else 0
        },
        'experiments': []
    }
    
    for i, (config_path, success, runtime, error_msg) in enumerate(results, 1):
        experiment_info = {
            'experiment_id': i,
            'config_path': config_path,
            'success': success,
            'runtime_seconds': runtime,
            'runtime_minutes': runtime / 60,
            'error_message': error_msg
        }
        summary['experiments'].append(experiment_info)
    
    # Save as YAML
    with open(output_file, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False, indent=2)
    
    print(f"\nBatch summary saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Run multiple MCMC sampling experiments sequentially")

    # Input methods
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--configs", nargs='+', help="List of config file paths")
    input_group.add_argument("--config-dir", help="Directory containing config files (will run all .yaml files)")
    input_group.add_argument("--config-list", help="Text file containing list of config paths (one per line)")

    # Options
    parser.add_argument("--continue-on-error", action="store_true",
                       help="Continue running other experiments if one fails")
    parser.add_argument("--summary-file", default="batch_summary.yaml",
                       help="File to save batch run summary (default: batch_summary.yaml)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show which experiments would be run without actually running them")

    args = parser.parse_args()

    # Collect config files
    config_paths = []

    if args.configs:
        config_paths = args.configs
    elif args.config_dir:
        config_dir = Path(args.config_dir)
        if not config_dir.exists():
            print(f"Error: Directory {args.config_dir} does not exist")
            sys.exit(1)

        config_paths = [str(p) for p in config_dir.glob("*.yaml")]
        config_paths.extend([str(p) for p in config_dir.glob("*.yml")])
        config_paths.sort()

        if not config_paths:
            print(f"Error: No .yaml or .yml files found in {args.config_dir}")
            sys.exit(1)

    elif args.config_list:
        if not os.path.exists(args.config_list):
            print(f"Error: Config list file {args.config_list} does not exist")
            sys.exit(1)

        with open(args.config_list, 'r') as f:
            config_paths = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    print(f"Found {len(config_paths)} config files to process")

    # Validate config files
    valid_configs = validate_config_files(config_paths)

    if not valid_configs:
        print("Error: No valid config files found")
        sys.exit(1)

    print(f"Will run {len(valid_configs)} valid experiments")

    if args.dry_run:
        print("\nDRY RUN - Would run the following experiments:")
        for i, config_path in enumerate(valid_configs, 1):
            print(f"  {i:2d}. {config_path}")
        print(f"\nTotal: {len(valid_configs)} experiments")
        return

    # Confirm before starting
    response = input(f"\nProceed with running {len(valid_configs)} experiments? [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("Aborted by user")
        return

    # Run experiments
    print(f"\nStarting batch run of {len(valid_configs)} experiments...")
    print(f"Continue on error: {args.continue_on_error}")
    print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    batch_start_time = time.time()
    results = []

    for i, config_path in enumerate(valid_configs, 1):
        success, runtime, error_msg = run_single_experiment(config_path, i, len(valid_configs))
        results.append((config_path, success, runtime, error_msg))

        if not success and not args.continue_on_error:
            print(f"\nStopping batch run due to failure in experiment {i}")
            print("Use --continue-on-error to continue running despite failures")
            break

    batch_end_time = time.time()
    batch_runtime = batch_end_time - batch_start_time

    # Print final summary
    total_experiments = len(results)
    successful = sum(1 for _, success, _, _ in results if success)
    failed = total_experiments - successful

    print("\n" + "="*80)
    print("BATCH RUN COMPLETED")
    print("="*80)
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/total_experiments*100:.1f}%")
    print(f"Total runtime: {batch_runtime:.2f} seconds ({batch_runtime/3600:.2f} hours)")
    print(f"Average per experiment: {batch_runtime/total_experiments:.2f} seconds")
    print(f"End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if failed > 0:
        print(f"\nFailed experiments:")
        for i, (config_path, success, runtime, error_msg) in enumerate(results, 1):
            if not success:
                print(f"  {i:2d}. {config_path} - {error_msg}")

    print("="*80)

    summary_dir = os.path.join("src", "multiple_exp_summary")
    os.makedirs(summary_dir, exist_ok=True)
    base_summary_file = os.path.basename(args.summary_file)
    summary_file_path = os.path.join(summary_dir, base_summary_file)
    save_batch_summary(results, summary_file_path)

    print(f"\nBatch run completed. Check {summary_file_path} for detailed results.")

if __name__ == "__main__":
    main()
