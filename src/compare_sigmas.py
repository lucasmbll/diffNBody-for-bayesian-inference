"""
Script to compare the evolution of RMS radius for different sigma values in N-body simulations.
"""

import os
import yaml
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from run import gpu_config, main
from model import model
from utils import blobs_params_init
import matplotlib as mpl

# Set up better plot styling
plt.style.use('seaborn-v0_8-darkgrid')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.titlesize'] = 18

def calculate_rms_radius(positions, masses, center=None):
    """Calculate the RMS radius of particles."""
    if center is None:
        # Calculate center of mass
        center = jnp.sum(positions * masses[:, None], axis=0) / jnp.sum(masses)
    
    # Calculate squared distance from center
    r_squared = jnp.sum((positions - center)**2, axis=1)
    
    # Calculate RMS radius
    rms_radius = jnp.sqrt(jnp.sum(r_squared * masses) / jnp.sum(masses))
    return rms_radius

def run_sigma_comparison(config_path, sigma_values, output_dir='results/sigma_comparison'):
    """
    Run simulations with different sigma values and plot RMS radius vs time.
    
    Args:
        config_path: Path to base configuration file
        sigma_values: List of sigma values to simulate
        output_dir: Directory to save the output plot
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the base configuration
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)
    
    # Configure GPU
    gpu_config(base_config)
    
    # Import necessary modules after GPU configuration
    import jax
    
    # Extract model parameters from config
    model_params = base_config["model_params"]
    G = model_params.get("G", 1.0)
    length = model_params.get("length", 64)
    softening = model_params.get("softening", 1.0)
    t_f = model_params.get("t_f", 1.0)
    dt = model_params.get("dt", 0.005)
    solver = model_params.get("solver", "LeapfrogMidpoint")
    
    # Extract unit conversion factors from config
    units = base_config.get("plot_settings", {}).get("units", {})
    kpc_per_pixel = units.get("kpc_per_pixel", 1.0)
    msun_per_mass_unit = units.get("msun_per_mass_unit", 1.0)
    gyr_per_time_unit = units.get("gyr_per_time_unit", 1.0)
    
    # Setup figure for plotting
    plt.figure(figsize=(12, 8))
    
    # List to store data for legend
    legend_entries = []
    
    # Run simulation for each sigma value
    for sigma in sigma_values:
        print(f"Running simulation with sigma = {sigma}")
        
        # Create a modified config with the current sigma
        config = base_config.copy()
        # Modify sigma in the configuration
        config["model_params"]["blobs_params"][0]["pos_params"]["sigma"] = sigma
        
    
        # Create model with modified params
        data_key = jax.random.PRNGKey(config["model_params"].get("data_seed", 0))
        blobs_params = config["model_params"]["blobs_params"]
        density_scaling = config["model_params"].get("density_scaling", "none")
        velocity_scaling = config["model_params"].get("velocity_scaling", "none")
        
        # Initialize parameters for the model
        data_params, fixed_params, other_params, params_infos = blobs_params_init(blobs_params, None, None, "sim")
        
        # Create model function
        def model_fn(params, key):
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
                observable=["density", "vx", "vy", "vz"]
            )
        
        # Run the model
        _, _, _, sol_ts, sol_ys, masses = model_fn(data_params, data_key)
        
        # Calculate RMS radius for each timestep
        rms_radii = []
        for i in range(len(sol_ts)):
            positions = sol_ys[i, 0]  # positions at timestep i
            rms_radius = calculate_rms_radius(positions, masses)
            rms_radii.append(rms_radius.item())  # Convert to Python scalar
        
        # Convert time to physical units but keep radius in code units
        time_physical = sol_ts * gyr_per_time_unit
        rms_code_units = np.array(rms_radii)
        
        # Plot the RMS radius vs time with time in physical units but radius in code units
        plt.plot(time_physical, rms_code_units, linewidth=2)
        
        # Format sigma value for legend - show in physical units (kpc)
        sigma_physical = sigma * kpc_per_pixel
        legend_entries.append(f"σ = {sigma_physical:.2f} kpc")
    
    # Finalize plot with proper units
    plt.xlabel('Time [Gyr]', fontsize=14)
    plt.ylabel('RMS Radius [code units]', fontsize=14)
    #plt.title('Evolution of RMS Radius for Different Initial Size Parameters (σ)', fontsize=16)
    
    # Add parameter information to the plot with both unit systems
    #param_info = (f'G={G}, tf={t_f*gyr_per_time_unit:.2f} Gyr, dt={dt*gyr_per_time_unit:.4f} Gyr, ' 
                 #f'L={length*kpc_per_pixel:.1f} kpc ({length} code units), Solver={solver}')
    #plt.figtext(0.5, 0.01, param_info, ha='center', fontsize=12)
    
    # Add unit conversion note to clarify the dual unit system
    unit_note = f'1 code unit = {kpc_per_pixel:.2f} kpc'
    plt.figtext(0.5, 0.04, unit_note, ha='center', fontsize=10, style='italic')
    
    plt.legend(legend_entries)
    plt.grid(True, alpha=0.3)
    
    # Create a subfolder with the current date and time
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    subfolder = os.path.join(output_dir, timestamp)
    os.makedirs(subfolder, exist_ok=True)
    
    # Save the plot in the subfolder
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])  # Make room for the parameter text and unit note
    plt.savefig(os.path.join(subfolder, 'rms_radius_comparison.png'), dpi=300)
    
    print(f"Plot saved to {subfolder}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Compare RMS radius evolution for different sigma values")
    parser.add_argument("--config", type=str, default="configs/report/1blob/sim/low_res_sim_1_blob.yaml", 
                        help="Path to base configuration file")
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0],
                        help="List of sigma values to simulate")
    parser.add_argument("--output", type=str, default="results/sigma_comparison",
                        help="Directory to save the output plot")
    
    args = parser.parse_args()
    run_sigma_comparison(args.config, args.sigmas, args.output)
