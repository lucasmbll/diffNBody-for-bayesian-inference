import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import jax
import jax.numpy as jnp
from jax.experimental import ode
from diffrax import diffeqsolve, ODETerm, SaveAt, LeapfrogMidpoint, Tsit5
from scipy.integrate import solve_ivp
from jaxpm.painting import cic_paint

# Import local modules
from initialization import initialize_gaussian_positions, initialize_cold_velocities

def jax_pairwise_forces(pos, masses, G, softening, length):
    """JAX implementation of pairwise forces."""
    dx = pos[:, None, :] - pos[None, :, :]
    dx = dx - length * jnp.round(dx / length)
    r2 = jnp.sum(dx**2, axis=-1) + softening**2
    inv_r3 = jnp.where(jnp.eye(pos.shape[0]), 0., r2**-1.5)
    
    mass_matrix = masses[:, None] * masses[None, :]
    F = jnp.einsum('ij,ijc->ic', -G * mass_matrix * inv_r3, dx)
    return F

def jax_nbody_ode(y, t, G, masses, softening, length):
    """JAX N-body system for jax.experimental.ode."""
    N = len(masses)
    pos = y[:3*N].reshape(N, 3)
    vel = y[3*N:].reshape(N, 3)
    
    forces = jax_pairwise_forces(pos, masses, G, softening, length)
    acc = forces / masses[:, None]
    
    dydt = jnp.concatenate([vel.flatten(), acc.flatten()])
    return dydt

def diffrax_nbody_ode(t, y, args):
    """Diffrax N-body system."""
    G, masses, softening, length = args
    N = len(masses)
    pos = y[:3*N].reshape(N, 3)
    vel = y[3*N:].reshape(N, 3)
    
    forces = jax_pairwise_forces(pos, masses, G, softening, length)
    acc = forces / masses[:, None]
    
    dydt = jnp.concatenate([vel.flatten(), acc.flatten()])
    return dydt

def diffrax_nbody_ode_stacked(t, state, args):
    """Diffrax N-body system with stacked state (pos, vel)."""
    G, masses, softening, length = args
    pos, vel = state
    
    forces = jax_pairwise_forces(pos, masses, G, softening, length)
    acc = forces / masses[:, None]
    
    return jnp.stack([vel, acc])

def benchmark_jax_experimental_ode(n_part, t_f, dt, G, softening, length):
    """Benchmark JAX + experimental ODE."""
    print(f"  Running JAX+experimental.ode (N={n_part})...")
    
    # Initialize system
    key = jax.random.PRNGKey(42)
    pos_key, vel_key = jax.random.split(key)
    
    pos_params = {'sigma': 5.0, 'center': [length/2, length/2, length/2]}
    vel_params = {'vel_dispersion': 0.1}
    
    pos = initialize_gaussian_positions(n_part, pos_params, length, pos_key)
    vel = initialize_cold_velocities(pos, vel_params)
    masses = jnp.ones(n_part)
    
    # Flatten state
    y0 = jnp.concatenate([pos.flatten(), vel.flatten()])
    
    # Time points
    t_eval = jnp.arange(0, t_f + dt, dt)
    
    start_time = time.time()
    
    try:
        sol = ode.odeint(
            lambda y, t: jax_nbody_ode(y, t, G, masses, softening, length),
            y0, t_eval, rtol=1e-6, atol=1e-8
        )
        success = True
    except Exception as e:
        print(f"    JAX experimental.ode failed: {e}")
        success = False
    
    end_time = time.time()
    
    return end_time - start_time, success

def benchmark_diffrax(n_part, t_f, dt, G, softening, length, jit=False):
    """Benchmark JAX + Diffrax."""
    jit_str = "JIT" if jit else "non-JIT"
    print(f"  Running JAX+Diffrax {jit_str} (N={n_part})...")
    
    # Initialize system
    key = jax.random.PRNGKey(42)
    pos_key, vel_key = jax.random.split(key)
    
    pos_params = {'sigma': 5.0, 'center': [length/2, length/2, length/2]}
    vel_params = {'vel_dispersion': 0.1}
    
    pos = initialize_gaussian_positions(n_part, pos_params, length, pos_key)
    vel = initialize_cold_velocities(pos, vel_params)
    masses = jnp.ones(n_part)
    
    # State for diffrax (stacked format)
    y0 = jnp.stack([pos, vel])
    
    # Time points
    ts = jnp.arange(0, t_f + dt, dt)
    
    # Create ODE term
    args = (G, masses, softening, length)
    term = ODETerm(diffrax_nbody_ode_stacked)
    solver = LeapfrogMidpoint()
    
    # Define integration function
    def integrate():
        return diffeqsolve(
            term, solver, t0=0.0, t1=t_f, dt0=dt, y0=y0,
            args=args, saveat=SaveAt(ts=ts),
            max_steps=max(int(t_f/dt)*2, 1000)
        )
    
    # JIT compilation if requested
    if jit:
        integrate = jax.jit(integrate)
        # Warm up JIT
        try:
            _ = integrate()
        except:
            pass
    
    start_time = time.time()
    
    try:
        sol = integrate()
        success = True
    except Exception as e:
        print(f"    JAX+Diffrax {jit_str} failed: {e}")
        success = False
    
    end_time = time.time()
    
    return end_time - start_time, success

def benchmark_diffrax_flat(n_part, t_f, dt, G, softening, length, jit=False):
    """Benchmark JAX + Diffrax with flat state representation."""
    jit_str = "JIT" if jit else "non-JIT"
    print(f"  Running JAX+Diffrax-flat {jit_str} (N={n_part})...")
    
    # Initialize system
    key = jax.random.PRNGKey(42)
    pos_key, vel_key = jax.random.split(key)
    
    pos_params = {'sigma': 5.0, 'center': [length/2, length/2, length/2]}
    vel_params = {'vel_dispersion': 0.1}
    
    pos = initialize_gaussian_positions(n_part, pos_params, length, pos_key)
    vel = initialize_cold_velocities(pos, vel_params)
    masses = jnp.ones(n_part)
    
    # Flatten state
    y0 = jnp.concatenate([pos.flatten(), vel.flatten()])
    
    # Time points
    ts = jnp.arange(0, t_f + dt, dt)
    
    # Create ODE term
    args = (G, masses, softening, length)
    term = ODETerm(diffrax_nbody_ode)
    solver = Tsit5()  # Use adaptive solver for flat representation
    
    # Define integration function
    def integrate():
        return diffeqsolve(
            term, solver, t0=0.0, t1=t_f, dt0=dt, y0=y0,
            args=args, saveat=SaveAt(ts=ts),
            max_steps=max(int(t_f/dt)*5, 2000)  # More steps for adaptive solver
        )
    
    # JIT compilation if requested
    if jit:
        integrate = jax.jit(integrate)
        # Warm up JIT
        try:
            _ = integrate()
        except:
            pass
    
    start_time = time.time()
    
    try:
        sol = integrate()
        success = True
    except Exception as e:
        print(f"    JAX+Diffrax-flat {jit_str} failed: {e}")
        success = False
    
    end_time = time.time()
    
    return end_time - start_time, success

def run_performance_comparison():
    """Run comprehensive performance comparison."""
    # Create output directory
    script_dir = Path(__file__).parent
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = script_dir / 'test_outputs' / 'performance' / f'comparison_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running performance comparison...")
    print(f"Output directory: {output_dir}")
    
    # Simulation parameters
    t_f = 0.1  # Short simulation for benchmarking
    dt = 0.01
    G = 5.0
    softening = 0.1
    length = 32  # Smaller box for faster computation
    
    # Particle count range
    n_part_range = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    
    # Initialize results storage
    results = {
        'n_particles': n_part_range,
        'jax_experimental_ode': [],
        'jax_diffrax_nojit': [],
        'jax_diffrax_jit': [],
        'jax_diffrax_flat_nojit': [],
        'jax_diffrax_flat_jit': []
    }
    
    success_counts = {key: 0 for key in results.keys() if key != 'n_particles'}
    
    print(f"Testing particle counts: {n_part_range}")
    print(f"Simulation parameters: t_f={t_f}, dt={dt}, G={G}, softening={softening}, length={length}")
    print("="*80)
    
    for n_part in n_part_range:
        print(f"\nTesting with {n_part} particles:")
        
        # 1. JAX + experimental.ode
        try:
            time_taken, success = benchmark_jax_experimental_ode(n_part, t_f, dt, G, softening, length)
            results['jax_experimental_ode'].append(time_taken if success else np.nan)
            if success:
                success_counts['jax_experimental_ode'] += 1
                print(f"    JAX+experimental.ode: {time_taken:.3f} seconds")
            else:
                print(f"    JAX+experimental.ode: FAILED")
        except Exception as e:
            print(f"    JAX+experimental.ode: ERROR - {e}")
            results['jax_experimental_ode'].append(np.nan)
        
        # 2. JAX + Diffrax (no JIT)
        try:
            time_taken, success = benchmark_diffrax(n_part, t_f, dt, G, softening, length, jit=False)
            results['jax_diffrax_nojit'].append(time_taken if success else np.nan)
            if success:
                success_counts['jax_diffrax_nojit'] += 1
                print(f"    JAX+Diffrax (no JIT): {time_taken:.3f} seconds")
            else:
                print(f"    JAX+Diffrax (no JIT): FAILED")
        except Exception as e:
            print(f"    JAX+Diffrax (no JIT): ERROR - {e}")
            results['jax_diffrax_nojit'].append(np.nan)
        
        # 3. JAX + Diffrax (JIT)
        try:
            time_taken, success = benchmark_diffrax(n_part, t_f, dt, G, softening, length, jit=True)
            results['jax_diffrax_jit'].append(time_taken if success else np.nan)
            if success:
                success_counts['jax_diffrax_jit'] += 1
                print(f"    JAX+Diffrax (JIT): {time_taken:.3f} seconds")
            else:
                print(f"    JAX+Diffrax (JIT): FAILED")
        except Exception as e:
            print(f"    JAX+Diffrax (JIT): ERROR - {e}")
            results['jax_diffrax_jit'].append(np.nan)
        
        # 4. JAX + Diffrax flat (no JIT)
        try:
            time_taken, success = benchmark_diffrax_flat(n_part, t_f, dt, G, softening, length, jit=False)
            results['jax_diffrax_flat_nojit'].append(time_taken if success else np.nan)
            if success:
                success_counts['jax_diffrax_flat_nojit'] += 1
                print(f"    JAX+Diffrax-flat (no JIT): {time_taken:.3f} seconds")
            else:
                print(f"    JAX+Diffrax-flat (no JIT): FAILED")
        except Exception as e:
            print(f"    JAX+Diffrax-flat (no JIT): ERROR - {e}")
            results['jax_diffrax_flat_nojit'].append(np.nan)
        
        # 5. JAX + Diffrax flat (JIT)
        try:
            time_taken, success = benchmark_diffrax_flat(n_part, t_f, dt, G, softening, length, jit=True)
            results['jax_diffrax_flat_jit'].append(time_taken if success else np.nan)
            if success:
                success_counts['jax_diffrax_flat_jit'] += 1
                print(f"    JAX+Diffrax-flat (JIT): {time_taken:.3f} seconds")
            else:
                print(f"    JAX+Diffrax-flat (JIT): FAILED")
        except Exception as e:
            print(f"    JAX+Diffrax-flat (JIT): ERROR - {e}")
            results['jax_diffrax_flat_jit'].append(np.nan)
    
    print("\n" + "="*80)
    print("Performance comparison completed!")
    
    # Print success summary
    print("\nSuccess rates:")
    for method, count in success_counts.items():
        rate = count / len(n_part_range) * 100
        print(f"  {method}: {count}/{len(n_part_range)} ({rate:.1f}%)")
    
    # Create performance plots
    create_performance_plots(results, success_counts, output_dir, t_f, dt, G, softening, length)
    
    # Save results
    save_results(results, success_counts, output_dir, t_f, dt, G, softening, length)
    
    print(f"\nResults saved to: {output_dir}")
    
    return results, output_dir

def create_performance_plots(results, success_counts, output_dir, t_f, dt, G, softening, length):
    """Create performance comparison plots."""
    print("\nCreating performance plots...")
    
    n_particles = np.array(results['n_particles'])
    
    # Define plot styles
    methods = [
        ('jax_experimental_ode', 'JAX + experimental.ode', 'blue', '--'),
        ('jax_diffrax_nojit', 'JAX + Diffrax (no JIT)', 'green', '-.'),
        ('jax_diffrax_jit', 'JAX + Diffrax (JIT)', 'orange', '-'),
        ('jax_diffrax_flat_nojit', 'JAX + Diffrax-flat (no JIT)', 'purple', ':'),
        ('jax_diffrax_flat_jit', 'JAX + Diffrax-flat (JIT)', 'brown', '-')
    ]
    
    # 1. Linear scale plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for method_key, method_name, color, linestyle in methods:
        times = np.array(results[method_key])
        valid_mask = ~np.isnan(times)
        
        if np.sum(valid_mask) > 0:
            ax.plot(n_particles[valid_mask], times[valid_mask], 
                   color=color, linestyle=linestyle, linewidth=2, 
                   marker='o', markersize=6, label=method_name)
    
    ax.set_xlabel('Number of Particles', fontsize=12)
    ax.set_ylabel('Computation Time (seconds)', fontsize=12)
    ax.set_title(f'N-Body Simulation Performance Comparison\n'
                f't_f={t_f}, dt={dt}, G={G}, softening={softening}, length={length}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison_linear.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Log scale plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for method_key, method_name, color, linestyle in methods:
        times = np.array(results[method_key])
        valid_mask = ~np.isnan(times) & (times > 0)
        
        if np.sum(valid_mask) > 0:
            ax.semilogy(n_particles[valid_mask], times[valid_mask], 
                       color=color, linestyle=linestyle, linewidth=2, 
                       marker='o', markersize=6, label=method_name)
    
    ax.set_xlabel('Number of Particles', fontsize=12)
    ax.set_ylabel('Computation Time (seconds, log scale)', fontsize=12)
    ax.set_title(f'N-Body Simulation Performance Comparison (Log Scale)\n'
                f't_f={t_f}, dt={dt}, G={G}, softening={softening}, length={length}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison_log.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Speedup comparison (relative to fastest method)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Find fastest method across all particle counts to use as baseline
    fastest_times = np.nanmin([np.array(results[method_key]) for method_key, _, _, _ in methods], axis=0)
    
    for method_key, method_name, color, linestyle in methods:
        times = np.array(results[method_key])
        valid_mask = ~np.isnan(times) & ~np.isnan(fastest_times) & (times > 0) & (fastest_times > 0)
        
        if np.sum(valid_mask) > 0:
            speedup = fastest_times[valid_mask] / times[valid_mask]
            ax.plot(n_particles[valid_mask], speedup, 
                   color=color, linestyle=linestyle, linewidth=2, 
                   marker='o', markersize=6, label=method_name)
    
    ax.axhline(y=1.0, color='black', linestyle='-', alpha=0.5, label='Fastest method (reference)')
    ax.set_xlabel('Number of Particles', fontsize=12)
    ax.set_ylabel('Speedup (relative to fastest method)', fontsize=12)
    ax.set_title(f'Speedup Comparison\n'
                f't_f={t_f}, dt={dt}, G={G}, softening={softening}, length={length}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'speedup_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Success rate bar chart
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    method_names = [name for _, name, _, _ in methods]
    success_rates = [success_counts[key] / len(results['n_particles']) * 100 
                    for key, _, _, _ in methods]
    colors = [color for _, _, color, _ in methods]
    
    bars = ax.bar(method_names, success_rates, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Integration Success Rate by Method', fontsize=14)
    ax.set_ylim(0, 105)
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'success_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Performance plots created successfully!")

def save_results(results, success_counts, output_dir, t_f, dt, G, softening, length):
    """Save benchmark results to files."""
    print("Saving results...")
    
    # Save raw data
    np.savez(output_dir / 'benchmark_results.npz', **results)
    
    # Save summary report
    with open(output_dir / 'performance_summary.txt', 'w') as f:
        f.write("=== N-BODY SIMULATION PERFORMANCE COMPARISON ===\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output directory: {output_dir}\n\n")
        
        f.write("=== SIMULATION PARAMETERS ===\n")
        f.write(f"Final time: {t_f}\n")
        f.write(f"Time step: {dt}\n")
        f.write(f"Gravitational constant: {G}\n")
        f.write(f"Softening: {softening}\n")
        f.write(f"Box length: {length}\n")
        f.write(f"Particle counts tested: {results['n_particles']}\n\n")
        
        f.write("=== SUCCESS RATES ===\n")
        for method_key, count in success_counts.items():
            rate = count / len(results['n_particles']) * 100
            f.write(f"{method_key}: {count}/{len(results['n_particles'])} ({rate:.1f}%)\n")
        f.write("\n")
        
        f.write("=== TIMING RESULTS (seconds) ===\n")
        f.write("N_particles")
        methods = ['jax_experimental_ode', 'jax_diffrax_nojit', 
                  'jax_diffrax_jit', 'jax_diffrax_flat_nojit', 'jax_diffrax_flat_jit']
        for method in methods:
            f.write(f"\t{method}")
        f.write("\n")
        
        for i, n_part in enumerate(results['n_particles']):
            f.write(f"{n_part}")
            for method in methods:
                time_val = results[method][i]
                if np.isnan(time_val):
                    f.write("\tFAILED")
                else:
                    f.write(f"\t{time_val:.3f}")
            f.write("\n")
        
        f.write("\n=== PERFORMANCE INSIGHTS ===\n")
        
        # Find fastest method for largest N
        largest_n_idx = -1
        fastest_method = None
        fastest_time = float('inf')
        
        for method in methods:
            time_val = results[method][largest_n_idx]
            if not np.isnan(time_val) and time_val < fastest_time:
                fastest_time = time_val
                fastest_method = method
        
        if fastest_method:
            f.write(f"Fastest method for N={results['n_particles'][largest_n_idx]}: {fastest_method} ({fastest_time:.3f}s)\n")
        
       
        
        # Scaling analysis
        f.write(f"\n=== SCALING ANALYSIS ===\n")
        for method in methods:
            times = np.array(results[method])
            particles = np.array(results['n_particles'])
            valid_mask = ~np.isnan(times) & (times > 0)
            
            if np.sum(valid_mask) >= 3:  # Need at least 3 points for trend
                valid_times = times[valid_mask]
                valid_particles = particles[valid_mask]
                
                # Fit power law: t = a * N^b
                log_times = np.log(valid_times)
                log_particles = np.log(valid_particles)
                
                # Linear regression in log space
                coeffs = np.polyfit(log_particles, log_times, 1)
                scaling_exponent = coeffs[0]
                
                f.write(f"{method}: O(N^{scaling_exponent:.2f})\n")
        
        f.write("\n=== RECOMMENDATIONS ===\n")
        
        # Best method for different use cases
        best_for_small = None
        best_for_large = None
        most_reliable = None
        
        small_n_idx = 0  # First particle count
        large_n_idx = -1  # Last particle count
        
        # Find best for small N
        best_small_time = float('inf')
        for method in methods:
            time_val = results[method][small_n_idx]
            if not np.isnan(time_val) and time_val < best_small_time:
                best_small_time = time_val
                best_for_small = method
        
        # Find best for large N
        best_large_time = float('inf')
        for method in methods:
            time_val = results[method][large_n_idx]
            if not np.isnan(time_val) and time_val < best_large_time:
                best_large_time = time_val
                best_for_large = method
        
        # Find most reliable (highest success rate)
        best_success_rate = 0
        for method, count in success_counts.items():
            rate = count / len(results['n_particles'])
            if rate > best_success_rate:
                best_success_rate = rate
                most_reliable = method
        
        f.write(f"Best for small systems (N={results['n_particles'][small_n_idx]}): {best_for_small}\n")
        f.write(f"Best for large systems (N={results['n_particles'][large_n_idx]}): {best_for_large}\n")
        f.write(f"Most reliable (success rate): {most_reliable} ({best_success_rate*100:.1f}%)\n")
        
        f.write("\n=== USAGE GUIDELINES ===\n")
        f.write("- For production runs: Use JAX+Diffrax (JIT) for best performance\n")
        f.write("- For prototyping: Use JAX+Diffrax (no JIT) for good balance\n")
        f.write("- For very large systems: Consider JAX+Diffrax with adaptive solvers\n")
        
        f.write("\n=== FILES GENERATED ===\n")
        f.write("- performance_comparison_linear.png\n")
        f.write("- performance_comparison_log.png\n")
        f.write("- speedup_comparison.png\n")
        f.write("- success_rates.png\n")
        f.write("- benchmark_results.npz\n")
        f.write("- performance_summary.txt\n")
    
    print(f"Results saved to: {output_dir}")

def load_config(config_path):
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_performance_comparison_from_config(config):
    """Run performance comparison using configuration file."""
    print("Running performance comparison from configuration...")
    
    # Create output directory
    script_dir = Path(__file__).parent
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = script_dir / 'test_outputs' / 'performance' / f'comparison_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Get parameters from config
    perf_params = config['performance_params']
    t_f = perf_params['t_f']
    dt = perf_params['dt']
    G = perf_params['G']
    softening = perf_params['softening']
    length = perf_params['length']
    n_part_range = perf_params['n_part_range']
    
    # Get methods to test
    methods_to_test = config.get('methods_to_test', {})
    
    print(f"Testing particle counts: {n_part_range}")
    print(f"Simulation parameters: t_f={t_f}, dt={dt}, G={G}, softening={softening}, length={length}")
    print(f"Methods to test: {list(methods_to_test.keys())}")
    print("="*80)
    
    # Initialize results storage
    results = {'n_particles': n_part_range}
    success_counts = {}
    
    # Add result arrays for each method
    for method_key in methods_to_test.keys():
        results[method_key] = []
        success_counts[method_key] = 0
    
    for n_part in n_part_range:
        print(f"\nTesting with {n_part} particles:")
        
        # Test each enabled method
    
        
        if methods_to_test.get('jax_experimental_ode', True):
            try:
                time_taken, success = benchmark_jax_experimental_ode(n_part, t_f, dt, G, softening, length)
                results['jax_experimental_ode'].append(time_taken if success else np.nan)
                if success:
                    success_counts['jax_experimental_ode'] += 1
                    print(f"    JAX+experimental.ode: {time_taken:.3f} seconds")
                else:
                    print(f"    JAX+experimental.ode: FAILED")
            except Exception as e:
                print(f"    JAX+experimental.ode: ERROR - {e}")
                results['jax_experimental_ode'].append(np.nan)
        
        if methods_to_test.get('jax_diffrax_nojit', True):
            try:
                time_taken, success = benchmark_diffrax(n_part, t_f, dt, G, softening, length, jit=False)
                results['jax_diffrax_nojit'].append(time_taken if success else np.nan)
                if success:
                    success_counts['jax_diffrax_nojit'] += 1
                    print(f"    JAX+Diffrax (no JIT): {time_taken:.3f} seconds")
                else:
                    print(f"    JAX+Diffrax (no JIT): FAILED")
            except Exception as e:
                print(f"    JAX+Diffrax (no JIT): ERROR - {e}")
                results['jax_diffrax_nojit'].append(np.nan)
        
        if methods_to_test.get('jax_diffrax_jit', True):
            try:
                time_taken, success = benchmark_diffrax(n_part, t_f, dt, G, softening, length, jit=True)
                results['jax_diffrax_jit'].append(time_taken if success else np.nan)
                if success:
                    success_counts['jax_diffrax_jit'] += 1
                    print(f"    JAX+Diffrax (JIT): {time_taken:.3f} seconds")
                else:
                    print(f"    JAX+Diffrax (JIT): FAILED")
            except Exception as e:
                print(f"    JAX+Diffrax (JIT): ERROR - {e}")
                results['jax_diffrax_jit'].append(np.nan)
        
        if methods_to_test.get('jax_diffrax_flat_nojit', True):
            try:
                time_taken, success = benchmark_diffrax_flat(n_part, t_f, dt, G, softening, length, jit=False)
                results['jax_diffrax_flat_nojit'].append(time_taken if success else np.nan)
                if success:
                    success_counts['jax_diffrax_flat_nojit'] += 1
                    print(f"    JAX+Diffrax-flat (no JIT): {time_taken:.3f} seconds")
                else:
                    print(f"    JAX+Diffrax-flat (no JIT): FAILED")
            except Exception as e:
                print(f"    JAX+Diffrax-flat (no JIT): ERROR - {e}")
                results['jax_diffrax_flat_nojit'].append(np.nan)
        
        if methods_to_test.get('jax_diffrax_flat_jit', True):
            try:
                time_taken, success = benchmark_diffrax_flat(n_part, t_f, dt, G, softening, length, jit=True)
                results['jax_diffrax_flat_jit'].append(time_taken if success else np.nan)
                if success:
                    success_counts['jax_diffrax_flat_jit'] += 1
                    print(f"    JAX+Diffrax-flat (JIT): {time_taken:.3f} seconds")
                else:
                    print(f"    JAX+Diffrax-flat (JIT): FAILED")
            except Exception as e:
                print(f"    JAX+Diffrax-flat (JIT): ERROR - {e}")
                results['jax_diffrax_flat_jit'].append(np.nan)
    
    print("\n" + "="*80)
    print("Performance comparison completed!")
    
    # Print success summary
    print("\nSuccess rates:")
    for method, count in success_counts.items():
        rate = count / len(n_part_range) * 100
        print(f"  {method}: {count}/{len(n_part_range)} ({rate:.1f}%)")
    
    # Create performance plots
    create_performance_plots(results, success_counts, output_dir, t_f, dt, G, softening, length)
    
    # Save results
    save_results(results, success_counts, output_dir, t_f, dt, G, softening, length)
    
    print(f"\nResults saved to: {output_dir}")
    
    return results, output_dir

def main():
    """Main function to run performance comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description='N-body simulation performance comparison')
    parser.add_argument('--config', type=str, help='Path to configuration file (optional)')
    args = parser.parse_args()
    
    print("N-Body Simulation Performance Comparison")
    print("="*50)
    
    # Set JAX platform
    try:
        jax.config.update("jax_platform_name", "gpu")
        print(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    except:
        print("JAX backend: CPU (GPU not available)")
    
    if args.config:
        # Run with configuration file
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Set CUDA device if specified
        cuda_num = config.get("cuda_visible_devices", None)
        if cuda_num is not None:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_num)
            print(f"CUDA device set to: {cuda_num}")
        
        results, output_dir = run_performance_comparison_from_config(config)
    else:
        # Run with default parameters
        print("Running with default parameters...")
        print("Available methods:")
        print("2. JAX + experimental.ode")
        print("3. JAX + Diffrax (no JIT)")
        print("4. JAX + Diffrax (JIT)")
        print("5. JAX + Diffrax-flat (no JIT)")
        print("6. JAX + Diffrax-flat (JIT)")
        print("="*50)
        
        results, output_dir = run_performance_comparison()
    
    print("\nPerformance comparison completed successfully!")
    print(f"Check the results in: {output_dir}")

if __name__ == "__main__":
    main()
