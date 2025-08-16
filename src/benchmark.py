import os
import yaml
import datetime 
import argparse
import shutil
import numpy as np
import time
import matplotlib.pyplot as plt

def gpu_config(config):
    cuda_num = config.get("cuda_visible_devices", None)
    if cuda_num is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_num)
        print(f"CUDA device set to: {cuda_num}")

def calculate_energy_drift(sol_ys, masses, G, length, softening):
    """Calculate relative energy drift over the simulation."""
    from utils import calculate_energy_variable_mass
    
    # Initial energy
    initial_pos = sol_ys[0, 0]
    initial_vel = sol_ys[0, 1]
    _, _, initial_energy = calculate_energy_variable_mass(
        initial_pos, initial_vel, masses, G, length, softening)
    
    # Final energy
    final_pos = sol_ys[-1, 0]
    final_vel = sol_ys[-1, 1]
    _, _, final_energy = calculate_energy_variable_mass(
        final_pos, final_vel, masses, G, length, softening)
    
    # Relative energy drift
    relative_drift = abs((final_energy - initial_energy) / initial_energy)
    return relative_drift

def run_single_benchmark(model_fn, data_params, masses, G, length, softening, 
                        solver, dt, data_key):
    """Run a single benchmark simulation and return energy drift and computation time."""
    
    print(f"  Running {solver} with dt={dt}")
    
    try:
        # Measure computation time
        start_time = time.time()
        
        # Run simulation
        initial_field, final_field, final_observable_field, sol_ts, sol_ys, _ = model_fn(
            data_params, dt, data_key, solver
        )
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        # Calculate energy drift
        energy_drift = calculate_energy_drift(sol_ys, masses, G, length, softening)
        
        return energy_drift, computation_time, True
        
    except Exception as e:
        print(f"    Error: {e}")
        return float('inf'), float('inf'), False

def create_benchmark_plots(benchmark_results, timesteps, base_dir):
    """Create benchmark plots: energy drift vs timestep and performance scatter plots."""
    
    solvers = list(benchmark_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(solvers)))
    
    # Plot 1: Average energy drift vs timestep for all solvers
    plt.figure(figsize=(12, 8))
    
    for i, solver in enumerate(solvers):
        solver_data = benchmark_results[solver]
        valid_timesteps = []
        valid_drifts = []
        
        for dt in timesteps:
            if dt in solver_data and solver_data[dt]['success_rate'] > 0:
                valid_timesteps.append(dt)
                valid_drifts.append(solver_data[dt]['avg_energy_drift'])
        
        if valid_timesteps:
            plt.loglog(valid_timesteps, valid_drifts, 'o-', 
                      color=colors[i], label=solver, linewidth=2, markersize=8)
    
    plt.xlabel('Timestep (dt)', fontsize=14)
    plt.ylabel('Relative Energy Drift', fontsize=14)
    plt.title('Energy Conservation vs Timestep by Solver', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'energy_drift_vs_timestep.png'), dpi=150)
    plt.close()
    print("Energy drift vs timestep plot saved")
    
    # Plot 2: Performance scatter plots (one per timestep)
    for dt in timesteps:
        plt.figure(figsize=(10, 8))
        
        valid_solvers = []
        energy_drifts = []
        comp_times = []
        success_rates = []
        
        for i, solver in enumerate(solvers):
            if (dt in benchmark_results[solver] and 
                benchmark_results[solver][dt]['success_rate'] > 0):
                
                valid_solvers.append(solver)
                energy_drifts.append(benchmark_results[solver][dt]['avg_energy_drift'])
                comp_times.append(benchmark_results[solver][dt]['avg_comp_time'])
                success_rates.append(benchmark_results[solver][dt]['success_rate'])
        
        if valid_solvers:
            # Create scatter plot with solver names as colors
            for j, solver in enumerate(valid_solvers):
                color_idx = solvers.index(solver)
                # Size marker by success rate
                marker_size = 50 + 200 * success_rates[j]
                plt.scatter(comp_times[j], energy_drifts[j], s=marker_size, 
                          color=colors[color_idx], alpha=0.7, 
                          label=f"{solver} (success: {success_rates[j]:.2f})",
                          edgecolors='black')
                
                # Add text annotation
                plt.annotate(solver, (comp_times[j], energy_drifts[j]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        plt.xlabel('Computation Time (seconds)', fontsize=14)
        plt.ylabel('Relative Energy Drift', fontsize=14)
        plt.title(f'Energy Drift vs Computation Time (dt = {dt})', fontsize=16)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        if valid_solvers:
            plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, f'performance_dt_{dt}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    print("Performance scatter plots saved")
    
    # Plot 3: Computation time vs timestep
    plt.figure(figsize=(12, 8))
    
    for i, solver in enumerate(solvers):
        solver_data = benchmark_results[solver]
        valid_timesteps = []
        valid_times = []
        
        for dt in timesteps:
            if dt in solver_data and solver_data[dt]['success_rate'] > 0:
                valid_timesteps.append(dt)
                valid_times.append(solver_data[dt]['avg_comp_time'])
        
        if valid_timesteps:
            plt.loglog(valid_timesteps, valid_times, 's-', 
                      color=colors[i], label=solver, linewidth=2, markersize=8)
    
    plt.xlabel('Timestep (dt)', fontsize=14)
    plt.ylabel('Computation Time (seconds)', fontsize=14)
    plt.title('Computation Time vs Timestep by Solver', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'computation_time_vs_timestep.png'), dpi=150)
    plt.close()
    print("Computation time vs timestep plot saved")

def save_benchmark_summary(benchmark_results, timesteps, base_dir):
    """Save a summary of benchmark results to a text file."""
    
    summary_path = os.path.join(base_dir, 'benchmark_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("BENCHMARK SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        for solver in benchmark_results.keys():
            f.write(f"Solver: {solver}\n")
            f.write("-" * 30 + "\n")
            
            solver_data = benchmark_results[solver]
            
            for dt in timesteps:
                if dt in solver_data:
                    data = solver_data[dt]
                    f.write(f"  dt = {dt}:\n")
                    f.write(f"    Energy drift: {data['avg_energy_drift']:.2e}\n")
                    f.write(f"    Comp time: {data['avg_comp_time']:.3f}s\n")
                    f.write(f"    Success rate: {data['success_rate']:.2f}\n")
                    f.write(f"    Runs: {data['num_runs']}\n\n")
            
            f.write("\n")
    
    print(f"Benchmark summary saved to {summary_path}")

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    gpu_config(config)

    import jax
    import jax.numpy as jnp
    from model import model 

    mode = 'sim'

    # --- Output directory ---
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_base = os.path.splitext(os.path.basename(config_path))[0]
    base_dir = os.path.join("results", "benchmark_results", f"{config_base}_{now_str}")
    os.makedirs(base_dir, exist_ok=True)

    # --- Model parameters ---
    model_params = config["model_params"]

    G = model_params.get("G", 1.0)
    length = model_params.get("length", 64)
    softening = model_params.get("softening", 1.0)
    t_f = model_params.get("t_f", 1.0)

    data_seed = model_params.get("data_seed", 0)
    data_key = jax.random.PRNGKey(data_seed)
    
    density_scaling = model_params.get("density_scaling", "none")
    velocity_scaling = model_params.get("velocity_scaling", "none")
    
    blobs_params = model_params.get("blobs_params", [])
    if not blobs_params:
        raise ValueError("Blob parameters 'blobs_params' must be specified in the configuration file.")
    
    n_part = sum(blob['n_part'] for blob in blobs_params)
    total_mass = sum(blob['n_part'] * blob.get('m_part', 1.0) for blob in blobs_params)

    initial_position = None
    prior_params = None

    from utils import blobs_params_init
    data_params, fixed_params, other_params, params_infos = blobs_params_init(blobs_params, prior_params, initial_position, mode)

    observable = model_params.get("observable", ["density"])  # Use minimal observable for speed

    # Create model function with fixed parameters
    def model_fn(params, dt, key, solver):
        return model(
            params,
            fixed_params,
            other_params,
            params_infos,
            G=G,
            length=length,
            softening=softening,
            t_f=t_f,
            dt=dt,
            key=key,
            density_scaling=density_scaling,
            velocity_scaling=velocity_scaling,
            solver=solver,
            observable=observable
        )

    # First run to compile and get masses
    print("Running initial compilation...")
    result = model_fn(data_params, 0.01, data_key, "LeapfrogMidpoint")
    initial_field, final_field, final_observable_field, sol_ts, sol_ys, masses = result
    
    print(f"Compilation completed.")
    print(f"Total particles: {n_part}, Total mass: {total_mass:.2f}")

    # Get benchmark parameters
    benchmark_params = config["benchmark_params"]
    solvers = benchmark_params.get("solvers", ["LeapfrogMidpoint"])
    timesteps = benchmark_params.get("timesteps", [0.01])
    num_runs = benchmark_params.get("num_runs", 3)

    print(f"\nStarting benchmark with {len(solvers)} solvers and {len(timesteps)} timesteps")
    print(f"Running {num_runs} trials per configuration")

    # Initialize benchmark results dictionary
    benchmark_results = {}
    
    # Run benchmark for each solver and timestep
    for solver in solvers:
        print(f"\nBenchmarking solver: {solver}")
        benchmark_results[solver] = {}
        
        for dt in timesteps:
            print(f"  Testing timestep: {dt}")
            
            energy_drifts = []
            comp_times = []
            successes = []
            
            for run in range(num_runs):
                # Use different random key for each run
                run_key = jax.random.PRNGKey(data_seed + run * 1000)
                
                energy_drift, comp_time, success = run_single_benchmark(
                    model_fn, data_params, masses, G, length, softening,
                    solver, dt, run_key
                )
                
                if success:
                    energy_drifts.append(energy_drift)
                    comp_times.append(comp_time)
                
                successes.append(success)
            
            # Store results for this solver-timestep combination
            if energy_drifts:  # If at least one run succeeded
                benchmark_results[solver][dt] = {
                    'avg_energy_drift': np.mean(energy_drifts),
                    'std_energy_drift': np.std(energy_drifts),
                    'avg_comp_time': np.mean(comp_times),
                    'std_comp_time': np.std(comp_times),
                    'success_rate': np.mean(successes),
                    'num_runs': len(energy_drifts),
                    'all_energy_drifts': energy_drifts,
                    'all_comp_times': comp_times
                }
                
                print(f"    Average energy drift: {np.mean(energy_drifts):.2e}")
                print(f"    Average comp time: {np.mean(comp_times):.3f}s")
                print(f"    Success rate: {np.mean(successes):.2f}")
            else:
                print(f"    All runs failed for {solver} with dt={dt}")
                benchmark_results[solver][dt] = {
                    'avg_energy_drift': float('inf'),
                    'std_energy_drift': 0.0,
                    'avg_comp_time': float('inf'),
                    'std_comp_time': 0.0,
                    'success_rate': 0.0,
                    'num_runs': 0,
                    'all_energy_drifts': [],
                    'all_comp_times': []
                }

    # Create plots
    print("\nCreating benchmark plots...")
    create_benchmark_plots(benchmark_results, timesteps, base_dir)
    
    # Save numerical results
    print("Saving benchmark summary...")
    save_benchmark_summary(benchmark_results, timesteps, base_dir)
    
    # Save detailed results as numpy file
    np.save(os.path.join(base_dir, 'benchmark_results.npy'), benchmark_results)
    
    # Copy config file
    shutil.copy(config_path, os.path.join(base_dir, "config.yaml"))
    
    print(f"\nBenchmark completed successfully!")
    print(f"Results saved to: {base_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark N-body solvers")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
