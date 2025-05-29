import sys
import argparse
import multiprocessing
import os
import subprocess

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

def main():
    parser = argparse.ArgumentParser(
        description="Run up to 4 experiments in parallel, each with its own YAML config."
    )
    parser.add_argument(
        "configs",
        nargs="+",
        help="Paths to YAML config files (max 4)."
    )
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

if __name__ == "__main__":
    main()
