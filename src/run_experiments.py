# run_experiment.py

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0" # Set memory fraction to 100% for JAX

import yaml
import datetime  # Added for date-time based directory naming

# --- Set CUDA device before importing JAX ---
def set_cuda_device_from_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    cuda_num = config.get("cuda_visible_devices", None)
    if cuda_num is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_num)
        print(f"CUDA device set to: {cuda_num}")

import argparse
import jax
import shutil
import jax.numpy as jnp
import numpy as np
import pickle

from model import model 

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    mode = config.get("mode")
    if mode is None:
        raise ValueError("Please specify the mode in the configuration file under 'mode' key : 'sim' or 'sampling'.")
    if mode not in ["sim", "sampling"]:
        raise ValueError(f"Invalid mode: {mode}. Must be either 'sim' or 'sampling'")

    # --- Output directory ---
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_base = os.path.splitext(os.path.basename(config_path))[0]
    if mode == "sampling":
        base_dir = os.path.join("results", "sampling_results", f"{config_base}_{now_str}")
    else:  # mode == "sim"
        base_dir = os.path.join("results", "nbody_sim_results", f"{config_base}_{now_str}")
    os.makedirs(base_dir, exist_ok=True)

    # --- Model parameters ---
    model_params = config["model_params"]
    model_fn = model 
        
    # Extract common parameters
    G = model_params.get("G", 5.0)
    length = model_params.get("length", 64)
    softening = model_params.get("softening", 0.1)
    t_f = model_params.get("t_f", 1.0)
    dt = model_params.get("dt", 0.5)
    m_part = model_params.get("m_part", 1.0)

    # Random seed
    data_seed = model_params.get("data_seed", 0)
    data_key = jax.random.PRNGKey(data_seed)
    
    # Density scaling parameters
    density_scaling = model_params.get("density_scaling", "none")
    scaling_kwargs = model_params.get("scaling_kwargs", {})
    
    # Extract blobs parameters
    blobs_params = model_params.get("blobs_params", [])
    if not blobs_params:
        raise ValueError("Blob parameters 'blobs_params' must be specified in the configuration file.")
    
    n_part = sum(blob['n_part'] for blob in blobs_params)

    # Run model to generate mock data
    input_field, init_pos, final_pos, output_field, sol, init_params = model_fn(
        blobs_params,
        G=G,
        length=length,
        softening=softening,
        t_f=t_f,
        dt=dt,
        m_part=m_part,
        key=data_key,
        density_scaling=density_scaling,
        **scaling_kwargs
    )
    
    if mode == "sim":
        print(f"Simulation completed.")
    else:
        print(f"Mock data generated.")  
    if density_scaling != "none":
        print(f"Density scaling applied: {density_scaling} with parameters: {scaling_kwargs}")

    data = output_field  # This is the mock data we will use for inference

    # --- Plots and video ---
    plot_settings = config.get("plot_settings", {})
    
    # Pre-compute energy if needed for any plots
    enable_energy_tracking = plot_settings.get("enable_energy_tracking", True)
    energy_data = None

    if enable_energy_tracking:
            print("Pre-calculating energy for all timesteps...")
            from utils import calculate_energy
            all_times = []
            all_ke = []
            all_pe = []
            all_te = []
            
            for i in range(len(sol.ts)):
                pos_t = sol.ys[i, 0]
                vel_t = sol.ys[i, 1]
                ke, pe, te = calculate_energy(pos_t, vel_t, G, length, softening, m_part)
                all_times.append(sol.ts[i])
                all_ke.append(ke)
                all_pe.append(pe)
                all_te.append(te)
            
            energy_data = {
                'times': jnp.array(all_times),
                'kinetic': jnp.array(all_ke),
                'potential': jnp.array(all_pe),
                'total': jnp.array(all_te)
            }
            print("Energy calculation completed.")
    
    if plot_settings['density_field_plot'].get("do"):
        from plotting import plot_density_fields_and_positions
        print("Creating density fields and positions plot...")
        fig = plot_density_fields_and_positions(
            G, t_f, dt, length, n_part, input_field, init_pos, final_pos, output_field, 
            density_scaling=density_scaling)
        fig.savefig(os.path.join(base_dir, "density_fields_and_positions.png"))
        print("Density fields and positions plots saved successfully")
    
    if plot_settings['timeseries_plot'].get("do"):
        from plotting import plot_timesteps
        print("Creating timesteps plot...")
        plot_timesteps_num = config.get("plot_timesteps", 10)
        fig, _ = plot_timesteps(sol, length, G, t_f, dt, n_part, num_timesteps=plot_timesteps_num, softening=softening, m_part=m_part,
                                enable_energy_tracking=enable_energy_tracking, density_scaling=density_scaling,
                                energy_data=energy_data)
        fig.savefig(os.path.join(base_dir, "timesteps.png"))
        print("Timesteps plot saved successfully")
    
    if plot_settings['trajectories_plot'].get("do"):
        from plotting import plot_trajectories
        print("Creating trajectories plot...")
        num_trajectories = plot_settings['trajectories_plot'].get("num_trajectories", 10)
        zoom = plot_settings['trajectories_plot'].get("zoom_for_trajectories", True)
        fig = plot_trajectories(sol, G, t_f, dt, length, n_part, num_trajectories=num_trajectories, 
                                zoom=zoom)
        fig.savefig(os.path.join(base_dir, "trajectories.png"))
        print("Trajectories plot saved successfully")
    
    if plot_settings['velocity_plot'].get("do"):
        from plotting import plot_velocity_distributions
        print("Creating velocity distributions plot...")
        fig, _ = plot_velocity_distributions(sol, G, t_f, dt, length, n_part)
        fig.savefig(os.path.join(base_dir, "velocity_distributions.png"))
        print("Velocity distributions plot saved successfully")
 
    if plot_settings['generate_video'].get("do"):
        print("Creating simulation video...")
        from plotting import create_video
        video_path = os.path.join(base_dir, "simulation_video.mp4")
        fps = plot_settings['generate_video'].get("video_fps", 10)
        dpi = plot_settings['generate_video'].get("video_dpi", 100)
        
        create_video(sol, length, G, t_f, dt, n_part, 
                    save_path=video_path, fps=fps, dpi=dpi, density_scaling=density_scaling, 
                    softening=softening, m_part=m_part,
                    enable_energy_tracking=enable_energy_tracking,
                    energy_data=energy_data)
        print("Simulation video saved successfully")
        
    # Save init_params for reference
    # Save a copy of the config file in the result directory
    shutil.copy(config_path, os.path.join(base_dir, "config.yaml"))
    
    if mode == "sim":
        print("Simulation completed.")
        return

    # --- Sampling part ---
    from likelihood import get_log_posterior
    from sampling import run_hmc, run_nuts, extract_params_to_infer

    prior_type = config.get("prior_type", None)
    prior_params = config.get("prior_params", None)
    if prior_params is None or prior_type is None:
        raise ValueError("No prior specified in config file. Please provide 'prior_params' and 'prior_type' in your configuration.")
    
    # Prepare model arguments for likelihood
    model_kwargs = {
        "G": G,
        "length": length,
        "softening": softening,
        "t_f": t_f,
        "dt": dt,
        "m_part": m_part,
        "density_scaling": density_scaling,
        **scaling_kwargs
    }
    
    log_posterior = get_log_posterior(
        config["likelihood_type"], 
        data, 
        prior_params=prior_params,
        prior_type=prior_type,
        model_fn=model_fn,
        init_params=init_params,
        **model_kwargs,
        **config.get("likelihood_kwargs", {})
    )

    # Initialize parameters for sampling
    initial_position = config.get("initial_position", None)
    if initial_position is None:
        print("No initial position specified. Using default values from init_params...")
        # Only include parameters that have priors defined
        available_params = extract_params_to_infer(init_params)
        initial_position = {k: v for k, v in available_params.items() if k in prior_params}
        print(f"Auto-generated initial position: {list(initial_position.keys())}")
    
        # Add after initial_position is created
    print("="*50)
    print("PARAMETER SAMPLING VERIFICATION")
    print("="*50)
    all_possible_params = extract_params_to_infer(init_params)
    print(f"All possible parameters: {list(all_possible_params.keys())}")
    print(f"Parameters with priors: {list(prior_params.keys())}")
    print(f"Parameters being sampled: {list(initial_position.keys())}")
    print(f"Fixed parameters: {set(all_possible_params.keys()) - set(initial_position.keys())}")

    rng_key = jax.random.PRNGKey(config.get("sampling_seed", 12345))
    num_samples = config.get("num_samples", 500)
    progress_bar = config.get("progress_bar", False)
    if config["sampler"] == "hmc":
        inv_mass_matrix = np.array(config["inv_mass_matrix"])
        step_size = config.get("step_size", 1e-3)
        num_integration_steps = config.get("num_integration_steps", 50)
        print(f"HMC parameters: step_size={step_size}, num_integration_steps={num_integration_steps}")
        samples = run_hmc(
            log_posterior,
            initial_position,
            inv_mass_matrix,
            step_size,
            num_integration_steps,
            rng_key,
            num_samples,
            progress_bar=progress_bar  # Pass progress_bar
        )
    elif config["sampler"] == "nuts":
        num_warmup = config.get("num_warmup", 1000)
        samples = run_nuts(
            log_posterior,
            initial_position,
            rng_key,
            num_samples,
            num_warmup,
            progress_bar=progress_bar  # Pass progress_bar
        )
    else:
        raise ValueError("Unknown sampler: should be 'hmc' or 'nuts'")
    print("Sampling finished.")

    #  Save sampling results
    if not isinstance(samples, dict):
        raise ValueError("Expected sampling result to be a dict of parameter arrays.")

    samples_dict = {k: np.array(v) for k, v in samples.items()}
    inferred_keys = list(samples_dict.keys())

    # Build theta dict with true values extracted from blobs_params
    theta = {}
    blobs_params = model_params.get("blobs_params", [])

    def extract_true_values_from_blobs(blobs_params):
        """Extract true parameter values from blobs_params configuration."""
        true_values = {}
        
        for blob_idx, blob in enumerate(blobs_params):
            # Extract position parameters
            if blob['pos_type'] == 'gaussian':
                true_values[f"blob{blob_idx}_sigma"] = blob['pos_params']['sigma']
                true_values[f"blob{blob_idx}_center"] = blob['pos_params']['center']
            elif blob['pos_type'] == 'nfw':
                true_values[f"blob{blob_idx}_rs"] = blob['pos_params']['rs']
                true_values[f"blob{blob_idx}_c"] = blob['pos_params']['c']
                true_values[f"blob{blob_idx}_center"] = blob['pos_params']['center']
            
            # Extract velocity parameters
            if blob['vel_type'] == 'cold':
                if 'vel_dispersion' in blob['vel_params']:
                    true_values[f"blob{blob_idx}_vel_dispersion"] = blob['vel_params']['vel_dispersion']
            elif blob['vel_type'] == 'virial':
                if 'virial_ratio' in blob['vel_params']:
                    true_values[f"blob{blob_idx}_virial_ratio"] = blob['vel_params']['virial_ratio']
            elif blob['vel_type'] == 'circular':
                if 'vel_factor' in blob['vel_params']:
                    true_values[f"blob{blob_idx}_vel_factor"] = blob['vel_params']['vel_factor']
        
        return true_values

    # Extract true values
    all_true_values = extract_true_values_from_blobs(blobs_params)

    # Build theta dict for only the parameters that were actually sampled
    for key in inferred_keys:
        if key in all_true_values:
            theta[key] = all_true_values[key]
            print(f"✅ True value for '{key}': {all_true_values[key]}")
        else:
            theta[key] = None
            print(f"⚠️  Warning: No true value found for '{key}' in blobs_params.")

    param_order = tuple(inferred_keys)

    if isinstance(samples, dict):
        np.savez(os.path.join(base_dir, "samples.npz"), **{k: np.array(v) for k, v in samples.items()})
    else:
        np.savez(os.path.join(base_dir, "samples.npz"), samples=np.array(samples))
    print(f"Results saved to {os.path.join(base_dir, 'samples.npz')}")

    # Generate plots
    from plotting import plot_trace_subplots, plot_corner_after_burnin

    fig, _ = plot_trace_subplots(
        samples_dict,
        theta=theta,
        G=G, t_f=t_f, dt=dt, softening=softening, length=length, n_part=n_part,
        method=config["sampler"],
        param_order=param_order
    )
    fig.savefig(os.path.join(base_dir, "trace_sampling.png"))
    print("Trace plot saved.")

    fig = plot_corner_after_burnin(
        samples_dict,
        theta=theta,
        G=G, t_f=t_f, dt=dt, softening=softening, length=length, n_part=n_part,
        method=config["sampler"],
        burnin=config.get("num_warmup", 1000),
        param_order=param_order
    )
    fig.savefig(os.path.join(base_dir, "corner_sampling.png"))
    print("Corner plot saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCMC sampling for N-body initial distribution parameters inference")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    set_cuda_device_from_config(args.config)
    main(args.config)