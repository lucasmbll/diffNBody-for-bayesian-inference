import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jaxpm.painting import cic_paint
import datetime
from pathlib import Path

#Use with : python test/radius_test.py --config configs/test/radius_test_config.yaml
# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import local modules
from model import model

def calculate_blob_radius(positions, center, percentile=90):
    """
    Calculate the radius containing a given percentile of particles.
    
    Parameters:
    -----------
    positions : jnp.array
        Particle positions (N, 3)
    center : jnp.array
        Center position (3,)
    percentile : float
        Percentile for radius calculation (default: 90th percentile)
        
    Returns:
    --------
    radius : float
        Radius containing the specified percentile of particles
    """
    # Calculate distances from center
    distances = jnp.linalg.norm(positions - center, axis=1)
    
    # Return the percentile radius
    return jnp.percentile(distances, percentile)

def calculate_rms_radius(positions, center):
    """
    Calculate the RMS radius of the blob.
    
    Parameters:
    -----------
    positions : jnp.array
        Particle positions (N, 3)
    center : jnp.array
        Center position (3,)
        
    Returns:
    --------
    rms_radius : float
        RMS radius of the blob
    """
    distances_squared = jnp.sum((positions - center)**2, axis=1)
    return jnp.sqrt(jnp.mean(distances_squared))

def run_radius_test(config_path):
    """
    Run radius test with different sigma values.
    
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
    plot_radius_log_y = test_params.get("plot_radius_log_y", False)

    
    # Common parameters
    G = model_params.get("G", 5.0)
    length = model_params.get("length", 64)
    softening = model_params.get("softening", 0.1)
    t_f = model_params.get("t_f", 1.0)
    dt = model_params.get("dt", 0.05)
    m_part = model_params.get("m_part", 1.0)
    data_seed = model_params.get("data_seed", 42)
    density_scaling = model_params.get("density_scaling", "none")
    scaling_kwargs = model_params.get("scaling_kwargs", {})
    
    # Test parameters
    sigma_range = test_params["sigma_range"]
    timesteps_to_plot = test_params["timesteps_to_plot"]
    output_dir = test_params.get("output_dir", "test/test_outputs/radius_test")  # Changed default path
        
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_output_dir = os.path.join(output_dir, f"radius_test_{timestamp}")
    os.makedirs(full_output_dir, exist_ok=True)
    
    print(f"Starting radius test with {len(sigma_range)} sigma values: {sigma_range}")
    print(f"Output directory: {full_output_dir}")
    
    # Storage for results
    results = {}
    
    # Base blob parameters
    base_blob = config["model_params"]["base_blob_params"]
    
    # Run simulations for each sigma value
    for i, sigma in enumerate(sigma_range):
        print(f"\n--- Running simulation {i+1}/{len(sigma_range)} with sigma = {sigma} ---")
        
        # Create blob parameters with current sigma
        blob_params = [dict(base_blob)]
        blob_params[0]["pos_params"]["sigma"] = sigma
        
        # Create random key
        key = jax.random.PRNGKey(data_seed + i)  # Different seed for each sigma
        
        # Run simulation
        input_field, init_pos, final_pos, output_field, sol = model(
            blob_params,
            G=G,
            length=length,
            softening=softening,
            t_f=t_f,
            dt=dt,
            m_part=m_part,
            key=key,
            density_scaling=density_scaling,
            **scaling_kwargs
        )
        
        # Calculate radius evolution
        center = jnp.array(base_blob["pos_params"]["center"])
        times = sol.ts
        percentile_radii = []
        rms_radii = []
        
        print(f"Calculating radius evolution for {len(times)} timesteps...")
        for t_idx in range(len(times)):
            pos_t = sol.ys[t_idx, 0]  # Positions at time t
            
            # Calculate radii
            perc_radius = calculate_blob_radius(pos_t, center, percentile=90)
            rms_radius = calculate_rms_radius(pos_t, center)
            
            percentile_radii.append(perc_radius)
            rms_radii.append(rms_radius)
        
        # Store results
        results[sigma] = {
            'times': times,
            'percentile_radii': jnp.array(percentile_radii),
            'rms_radii': jnp.array(rms_radii),
            'solution': sol,
            'center': center,
            'input_field': input_field,
            'output_field': output_field
        }
        
        print(f"Initial 90th percentile radius: {percentile_radii[0]:.2f}")
        print(f"Final 90th percentile radius: {percentile_radii[-1]:.2f}")
        print(f"Initial RMS radius: {rms_radii[0]:.2f}")
        print(f"Final RMS radius: {rms_radii[-1]:.2f}")
    
    print("\n--- Creating plots ---")
    
    # Create radius evolution plot
    create_radius_evolution_plot(results, full_output_dir, plot_radius_log_y)
    
    # Create density field comparison plot
    create_density_field_comparison_plot(results, timesteps_to_plot, length, full_output_dir)
    
    # Save configuration for reference
    with open(os.path.join(full_output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\nRadius test completed. Results saved to: {full_output_dir}")
    
    return results, full_output_dir

def create_radius_evolution_plot(results, output_dir, log_y=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for i, (sigma, data) in enumerate(results.items()):
        times = data['times']
        perc_radii = data['percentile_radii']
        rms_radii = data['rms_radii']
        ax1.plot(times, perc_radii, 'o-', color=colors[i], label=f'σ = {sigma}', linewidth=2, markersize=3)
        ax2.plot(times, rms_radii, 'o-', color=colors[i], label=f'σ = {sigma}', linewidth=2, markersize=3)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('90th Percentile Radius')
    ax1.set_title('90th Percentile Radius Evolution' + (' (log y)' if log_y else ''))
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('RMS Radius')
    ax2.set_title('RMS Radius Evolution' + (' (log y)' if log_y else ''))
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    if log_y:
        ax1.set_yscale('log')
        ax2.set_yscale('log')
    plt.suptitle('Blob Radius Evolution for Different Initial σ Values', fontsize=16)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "radius_evolution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Radius evolution plot saved: {plot_path}")


def create_density_field_comparison_plot(results, timesteps_to_plot, length, output_dir):
    """
    Create density field comparison plot showing XY projections at different timesteps.
    
    Parameters:
    -----------
    results : dict
        Results dictionary with sigma values as keys
    timesteps_to_plot : list
        List of timestep indices to plot
    length : int
        Box size for density field
    output_dir : str
        Output directory path
    """
    n_sigma = len(results)
    n_timesteps = len(timesteps_to_plot)
    
    fig, axes = plt.subplots(n_sigma, n_timesteps, figsize=(4*n_timesteps, 4*n_sigma))
    
    # Ensure axes is 2D
    if n_sigma == 1:
        axes = axes.reshape(1, -1)
    if n_timesteps == 1:
        axes = axes.reshape(-1, 1)
    
    sigma_values = list(results.keys())
    
    for i, sigma in enumerate(sigma_values):
        sol = results[sigma]['solution']
        times = results[sigma]['times']
        
        for j, t_idx in enumerate(timesteps_to_plot):
            if t_idx >= len(times):
                # Skip if timestep index is out of range
                axes[i, j].text(0.5, 0.5, 'N/A', ha='center', va='center', 
                               transform=axes[i, j].transAxes, fontsize=16)
                axes[i, j].set_title(f'σ={sigma}, t_idx={t_idx} (N/A)')
                continue
            
            # Get positions at this timestep
            pos_t = sol.ys[t_idx, 0]
            time_t = times[t_idx]
            
            # Create density field
            density_field = cic_paint(jnp.zeros((length, length, length)), pos_t)
            
            # Project onto XY plane
            density_xy = jnp.sum(density_field, axis=2)
            
            # Plot
            im = axes[i, j].imshow(density_xy, cmap='inferno', origin='lower',
                                  extent=[0, length, 0, length], aspect='auto')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)
            
            # Set title and labels
            if i == 0:  # Top row
                axes[i, j].set_title(f't = {time_t:.2f}')
            if j == 0:  # Left column
                axes[i, j].set_ylabel(f'σ = {sigma}\nY')
            if i == n_sigma - 1:  # Bottom row
                axes[i, j].set_xlabel('X')
            
            # Set axis limits
            axes[i, j].set_xlim(0, length)
            axes[i, j].set_ylim(0, length)
    
    plt.suptitle('Density Field Evolution (XY Projection) for Different σ Values', 
                 fontsize=16)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "density_field_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Density field comparison plot saved: {plot_path}")

def main():
    """Main function for radius test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run radius test for Gaussian blob simulations")
    parser.add_argument("--config", type=str, 
                       default="configs/radius_test_config.yaml",
                       help="Path to configuration file")
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file {args.config} not found")
        sys.exit(1)
    
    # Run the test
    results, output_dir = run_radius_test(args.config)
    
    print(f"\nRadius test completed successfully!")
    print(f"Results directory: {output_dir}")
    
    # Print summary
    print("\nSummary of results:")
    print("-" * 50)
    for sigma, data in results.items():
        initial_rms = data['rms_radii'][0]
        final_rms = data['rms_radii'][-1]
        growth_factor = final_rms / initial_rms
        print(f"σ = {sigma:5.1f}: Initial RMS radius = {initial_rms:6.2f}, "
              f"Final RMS radius = {final_rms:6.2f}, Growth factor = {growth_factor:.2f}")

if __name__ == "__main__":
    main()
