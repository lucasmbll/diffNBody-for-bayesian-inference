import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import time
import datetime
from pathlib import Path

# Add src directory to path to import modules
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(src_dir)

from model import model
from utils import calculate_energy
def create_performance_comparison_plots(results, output_dir):
    """Create improved plots comparing solver performance."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Extract data for plotting
    solvers = list(results.keys())
    dt_values = []
    for solver in solvers:
        dt_values.extend(list(results[solver].keys()))
    dt_values = sorted(list(set(dt_values)))
    if not dt_values:
        print("No dt values found for plotting.")
        return

    # Find smallest and largest dt
    dt_min = min(dt_values)
    dt_max = max(dt_values)

    # Create figure with subplots (2x2, but only 3 used)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Solver Performance Comparison', fontsize=16)

    # Computation time vs time step
    ax1 = axes[0, 0]
    for solver in solvers:
        times = []
        dts = []
        for dt in dt_values:
            if dt in results[solver] and results[solver][dt]['success']:
                times.append(results[solver][dt]['computation_time'])
                dts.append(dt)
        if times:
            ax1.plot(dts, times, 'o-', label=solver, linewidth=2, markersize=6)
    ax1.set_xlabel('Time Step (dt)')
    ax1.set_ylabel('Computation Time (s)')
    ax1.set_title('Computation Time vs Time Step')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Energy drift vs time step
    ax2 = axes[0, 1]
    for solver in solvers:
        drifts = []
        dts = []
        for dt in dt_values:
            if dt in results[solver] and results[solver][dt]['success']:
                drifts.append(results[solver][dt]['energy_drift'])
                dts.append(dt)
        if drifts:
            ax2.plot(dts, drifts, 's-', label=solver, linewidth=2, markersize=6)
    ax2.set_xlabel('Time Step (dt)')
    ax2.set_ylabel('Relative Energy Drift')
    ax2.set_title('Energy Conservation vs Time Step')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    # Energy drift vs computation time for smallest dt
    ax3 = axes[1, 0]
    for solver in solvers:
        if dt_min in results[solver] and results[solver][dt_min]['success']:
            time = results[solver][dt_min]['computation_time']
            drift = results[solver][dt_min]['energy_drift']
            ax3.scatter(time, drift, label=solver, s=80, alpha=0.7)
    ax3.set_xlabel('Computation Time (s)')
    ax3.set_ylabel('Relative Energy Drift')
    ax3.set_title(f'Energy Drift vs Time (dt={dt_min})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_yscale('log')

    # Energy drift vs computation time for largest dt
    ax4 = axes[1, 1]
    for solver in solvers:
        if dt_max in results[solver] and results[solver][dt_max]['success']:
            time = results[solver][dt_max]['computation_time']
            drift = results[solver][dt_max]['energy_drift']
            ax4.scatter(time, drift, label=solver, s=80, alpha=0.7)
    ax4.set_xlabel('Computation Time (s)')
    ax4.set_ylabel('Relative Energy Drift')
    ax4.set_title(f'Energy Drift vs Time (dt={dt_max})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    ax4.set_yscale('log')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'performance_comparison.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Performance comparison plot saved to: {save_path}")
    return

def create_energy_conservation_plots(results, output_dir):
    """Create detailed energy conservation analysis plots."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    solvers = list(results.keys())
    dt_values = []
    for solver in solvers:
        dt_values.extend(list(results[solver].keys()))
    dt_values = sorted(list(set(dt_values)))
    
    # Create figure with energy conservation metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Energy Conservation Analysis', fontsize=16)
    
    # Energy drift
    ax1 = axes[0, 0]
    for solver in solvers:
        drifts = []
        dts = []
        for dt in dt_values:
            if dt in results[solver] and results[solver][dt]['success']:
                drifts.append(results[solver][dt]['energy_drift'])
                dts.append(dt)
        if drifts:
            ax1.loglog(dts, drifts, 'o-', label=solver, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Time Step (dt)')
    ax1.set_ylabel('Relative Energy Drift')
    ax1.set_title('Energy Drift vs Time Step')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Energy standard deviation
    ax2 = axes[0, 1]
    for solver in solvers:
        stds = []
        dts = []
        for dt in dt_values:
            if dt in results[solver] and results[solver][dt]['success']:
                stds.append(results[solver][dt]['energy_std'])
                dts.append(dt)
        if stds:
            ax2.loglog(dts, stds, 's-', label=solver, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Time Step (dt)')
    ax2.set_ylabel('Relative Energy Standard Deviation')
    ax2.set_title('Energy Fluctuations vs Time Step')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Maximum energy deviation
    ax3 = axes[1, 0]
    for solver in solvers:
        max_devs = []
        dts = []
        for dt in dt_values:
            if dt in results[solver] and results[solver][dt]['success']:
                max_devs.append(results[solver][dt]['max_energy_deviation'])
                dts.append(dt)
        if max_devs:
            ax3.loglog(dts, max_devs, '^-', label=solver, linewidth=2, markersize=6)
    
    ax3.set_xlabel('Time Step (dt)')
    ax3.set_ylabel('Maximum Relative Energy Deviation')
    ax3.set_title('Maximum Energy Deviation vs Time Step')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Success rate
    ax4 = axes[1, 1]
    solver_names = []
    success_rates = []
    
    for solver in solvers:
        total_runs = len(dt_values)
        successful_runs = sum(1 for dt in dt_values 
                            if dt in results[solver] and results[solver][dt]['success'])
        success_rate = successful_runs / total_runs * 100
        solver_names.append(solver)
        success_rates.append(success_rate)
    
    bars = ax4.bar(solver_names, success_rates, alpha=0.7)
    ax4.set_ylabel('Success Rate (%)')
    ax4.set_title('Solver Success Rate')
    ax4.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'energy_conservation.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Energy conservation plot saved to: {save_path}")

def create_energy_evolution_plots(results, energy_timesteps, output_dir):
    """Create detailed energy evolution plots for specific timesteps."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    solvers = list(results.keys())
    
    # Create a plot for each solver showing energy evolution
    for solver in solvers:
        dt_values = [dt for dt in results[solver].keys() if results[solver][dt]['success']]
        if not dt_values:
            continue
            
        # Create subplot grid
        n_plots = len(dt_values)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows), squeeze=False)
        fig.suptitle(f'Energy Evolution - {solver}', fontsize=16)
        
        for idx, dt in enumerate(sorted(dt_values)):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            result = results[solver][dt]
            times = result['times']
            ke = result['kinetic']
            pe = result['potential']
            te = result['total']
            
            ax.plot(times, ke, 'b-', label='Kinetic', linewidth=1.5)
            ax.plot(times, pe, 'r-', label='Potential', linewidth=1.5)
            ax.plot(times, te, 'k-', label='Total', linewidth=2)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Energy')
            ax.set_title(f'dt = {dt}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add energy drift annotation
            drift = result['energy_drift']
            ax.text(0.02, 0.98, f'Energy drift: {drift:.2e}', 
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top')
        
        # Hide unused subplots
        for idx in range(n_plots, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'energy_evolution_{solver}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Energy evolution plot for {solver} saved to: {save_path}")

def run_solver_test(config_path):
    """
    Run solver performance test comparing different solvers and time steps.
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set CUDA device
    cuda_num = config.get("cuda_visible_devices", None)
    if cuda_num is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_num)
        print(f"CUDA device set to: {cuda_num}")

    # Extract parameters
    model_params = config["model_params"]
    test_params = config["test_params"]
    
    # Common parameters
    G = model_params.get("G", 5.0)
    length = model_params.get("length", 64)
    softening = model_params.get("softening", 1.0)
    t_f = model_params.get("t_f", 3.0)
    m_part = model_params.get("m_part", 1.0)
    data_seed = model_params.get("data_seed", 42)
    density_scaling = model_params.get("density_scaling", "none")
    
    # Test parameters
    solvers = test_params["solvers_to_test"]
    dt_values = test_params["dt_values"]
    output_dir = test_params.get("output_dir", "results/solver_test")
    energy_timesteps = test_params.get("energy_comparison_timesteps", [0, 10, 20, 30, 50])
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_output_dir = os.path.join(output_dir, f"solver_test_{timestamp}")
    os.makedirs(full_output_dir, exist_ok=True)
    
    print(f"Starting solver test with {len(solvers)} solvers and {len(dt_values)} dt values")
    print(f"Solvers: {solvers}")
    print(f"dt values: {dt_values}")
    print(f"Output directory: {full_output_dir}")
    
    # Base blob parameters
    base_blobs = config["model_params"]["base_blobs_params"]
    
    # Storage for results
    results = {}
    
    # Total number of runs
    total_runs = len(solvers) * len(dt_values)
    run_count = 0
    
    # Run tests for each solver and dt combination
    for solver in solvers:
        results[solver] = {}
        
        for dt in dt_values:
            run_count += 1
            print(f"\n--- Run {run_count}/{total_runs}: {solver} with dt={dt} ---")
            
            # Create random key (same for all runs for fair comparison)
            key = jax.random.PRNGKey(data_seed)
            
            # Measure computation time
            start_time = time.time()
            
            try:
                # Run simulation
                input_field, init_pos, final_pos, output_field, sol = model(
                    base_blobs,
                    G=G,
                    length=length,
                    softening=softening,
                    t_f=t_f,
                    dt=dt,
                    m_part=m_part,
                    key=key,
                    density_scaling=density_scaling,
                    solver=solver
                )
                
                end_time = time.time()
                computation_time = end_time - start_time
                
                # Calculate energy evolution
                print(f"Calculating energy evolution for {len(sol.ts)} timesteps...")
                times = sol.ts
                ke_values = []
                pe_values = []
                te_values = []
                
                for t_idx in range(len(times)):
                    pos_t = sol.ys[t_idx, 0]
                    vel_t = sol.ys[t_idx, 1]
                    ke, pe, te = calculate_energy(pos_t, vel_t, G, length, softening, m_part)
                    ke_values.append(ke)
                    pe_values.append(pe)
                    te_values.append(te)
                
                # Calculate energy conservation metrics
                te_values = jnp.array(te_values)
                initial_energy = te_values[0]
                final_energy = te_values[-1]
                energy_drift = jnp.abs(final_energy - initial_energy) / jnp.abs(initial_energy)
                energy_std = jnp.std(te_values) / jnp.abs(initial_energy)
                max_energy_deviation = jnp.max(jnp.abs(te_values - initial_energy)) / jnp.abs(initial_energy)
                
                # Store results
                results[solver][dt] = {
                    'times': times,
                    'kinetic': jnp.array(ke_values),
                    'potential': jnp.array(pe_values),
                    'total': te_values,
                    'computation_time': computation_time,
                    'energy_drift': energy_drift,
                    'energy_std': energy_std,
                    'max_energy_deviation': max_energy_deviation,
                    'n_timesteps': len(times),
                    'success': True,
                    'solution': sol
                }
                
                print(f"Success! Time: {computation_time:.2f}s, Energy drift: {energy_drift:.2e}")
                
            except Exception as e:
                end_time = time.time()
                computation_time = end_time - start_time
                print(f"Failed! Time: {computation_time:.2f}s, Error: {str(e)}")
                
                # Store failure result
                results[solver][dt] = {
                    'computation_time': computation_time,
                    'success': False,
                    'error': str(e)
                }
    
    print("\n--- Creating plots ---")
    
    # Create performance comparison plots
    create_performance_comparison_plots(results, full_output_dir)
    
    # Create energy conservation plots
    create_energy_conservation_plots(results, full_output_dir)
    
    # Create detailed energy evolution plots
    create_energy_evolution_plots(results, energy_timesteps, full_output_dir)
    
    # Save configuration and results summary
    save_results_summary(results, config, full_output_dir)
    
    print(f"\nSolver test completed. Results saved to: {full_output_dir}")
    
    return results, full_output_dir

def main():
    """Main function for solver test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run solver performance test")
    parser.add_argument("--config", type=str, 
                       default="configs/test/solver_test_config.yaml",
                       help="Path to configuration file")
    args = parser.parse_args()
    
    # Adjust config path to be relative to project root
    if not os.path.isabs(args.config):
        # Get project root directory (parent of test directory)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, args.config)
    else:
        config_path = args.config
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file {config_path} not found")
        sys.exit(1)
    
    print("="*60)
    print("SOLVER PERFORMANCE TEST")
    print("="*60)
    print(f"Configuration file: {config_path}")
    
    # Run the test
    results, output_dir = run_solver_test(config_path)
    
    print(f"\nSolver test completed successfully!")
    print(f"Results directory: {output_dir}")
    
    # Additional performance analysis
    print("\n" + "="*60)
    print("DETAILED PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Collect all successful results for analysis
    successful_results = []
    for solver in results:
        for dt, result in results[solver].items():
            if result['success']:
                successful_results.append({
                    'solver': solver,
                    'dt': dt,
                    'energy_drift': result['energy_drift'],
                    'computation_time': result['computation_time'],
                    'energy_std': result['energy_std'],
                    'max_energy_deviation': result['max_energy_deviation'],
                    'n_timesteps': result['n_timesteps']
                })
    
    if successful_results:
        # Sort by different criteria
        by_energy_drift = sorted(successful_results, key=lambda x: x['energy_drift'])
        by_computation_time = sorted(successful_results, key=lambda x: x['computation_time'])
        by_combined_score = sorted(successful_results, key=lambda x: x['energy_drift'] * x['computation_time'])
        
        print("\nTop 3 by Energy Conservation:")
        for i, result in enumerate(by_energy_drift[:3]):
            print(f"  {i+1}. {result['solver']} (dt={result['dt']}) - Energy drift: {result['energy_drift']:.2e}")
        
        print("\nTop 3 by Speed:")
        for i, result in enumerate(by_computation_time[:3]):
            print(f"  {i+1}. {result['solver']} (dt={result['dt']}) - Time: {result['computation_time']:.2f}s")
        
        print("\nTop 3 by Overall Performance (energy_drift × time):")
        for i, result in enumerate(by_combined_score[:3]):
            score = result['energy_drift'] * result['computation_time']
            print(f"  {i+1}. {result['solver']} (dt={result['dt']}) - Score: {score:.2e}")
        
        # Success rate summary
        print(f"\nSuccess Rate Summary:")
        for solver in results.keys():
            total_runs = len(results[solver])
            successful_runs = sum(1 for r in results[solver].values() if r['success'])
            success_rate = successful_runs / total_runs * 100
            print(f"  {solver}: {successful_runs}/{total_runs} ({success_rate:.1f}%)")
    
    print("\n" + "="*60)
    print("Files saved:")
    print("="*60)
    print(f"  - Performance comparison: {output_dir}/performance_comparison.png")
    print(f"  - Energy conservation: {output_dir}/energy_conservation.png")
    print(f"  - Energy evolution plots: {output_dir}/energy_evolution_*.png")
    print(f"  - Results summary: {output_dir}/results_summary.yaml")
    print(f"  - Configuration: {output_dir}/config.yaml")
    print("="*60)

if __name__ == "__main__":
    main()
