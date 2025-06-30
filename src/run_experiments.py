# run_experiment.py

import os
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0" # Set memory fraction to 100% for JAX

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
    solver = model_params.get("solver", "LeapfrogMidpoint") 


    # Random seed for data generation
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
    total_mass = sum(blob['n_part'] * blob.get('m_part', 1.0) for blob in blobs_params)

    # Run model to generate mock data
    result = model_fn(
        blobs_params,
        G=G,
        length=length,
        softening=softening,
        t_f=t_f,
        dt=dt,
        key=data_key,
        density_scaling=density_scaling,
        solver=solver,
        **scaling_kwargs
    )
    
    # Unpack results (now includes masses)
    if len(result) == 6:
        input_field, init_pos, final_pos, output_field, sol, masses = result
    else:
        # Backward compatibility
        input_field, init_pos, final_pos, output_field, sol = result
        masses = jnp.ones(n_part)  # Default unit masses
    
    if mode == "sim":
        print(f"Simulation completed.")
        print(f"Total particles: {n_part}, Total mass: {total_mass:.2f}")
    else:
        print(f"Mock data generated.")
        print(f"Total particles: {n_part}, Total mass: {total_mass:.2f}")
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
            from utils import calculate_energy_variable_mass
            all_times = []
            all_ke = []
            all_pe = []
            all_te = []
            
            for i in range(len(sol.ts)):
                pos_t = sol.ys[i, 0]
                vel_t = sol.ys[i, 1]
                ke, pe, te = calculate_energy_variable_mass(pos_t, vel_t, masses, G, length, softening)
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
        from plotting import plot_density_fields_and_positions, plot_position_vs_radius_blobs
        print("Creating density fields and positions plot...")
        fig = plot_density_fields_and_positions(
            G, t_f, dt, length, n_part, input_field, init_pos, final_pos, output_field, 
            density_scaling=density_scaling, solver=solver,)
        fig.savefig(os.path.join(base_dir, "density_fields_and_positions.png"))
        fig2 = plot_position_vs_radius_blobs(sol, blobs_params, length, time_idx=0)
        fig2.savefig(os.path.join(base_dir, "position_vs_radius_blobs.png"))
        print("Density fields and positions plots saved successfully")
    
    if plot_settings['timeseries_plot'].get("do"):
        from plotting import plot_timesteps
        print("Creating timesteps plot...")
        plot_timesteps_num = config.get("plot_timesteps", 10)
        fig, _ = plot_timesteps(sol, length, G, t_f, dt, n_part, num_timesteps=plot_timesteps_num, softening=softening, masses=masses, solver=solver,
                                enable_energy_tracking=enable_energy_tracking, density_scaling=density_scaling,
                                energy_data=energy_data)
        fig.savefig(os.path.join(base_dir, "timesteps.png"))
        print("Timesteps plot saved successfully")
    
    if plot_settings['trajectories_plot'].get("do"):
        from plotting import plot_trajectories
        print("Creating trajectories plot...")
        num_trajectories = plot_settings['trajectories_plot'].get("num_trajectories", 10)
        zoom = plot_settings['trajectories_plot'].get("zoom_for_trajectories", True)
        fig = plot_trajectories(sol, G, t_f, dt, length, n_part, solver, num_trajectories=num_trajectories, 
                                zoom=zoom)
        fig.savefig(os.path.join(base_dir, "trajectories.png"))
        print("Trajectories plot saved successfully")
    
    if plot_settings['velocity_plot'].get("do"):
        from plotting import plot_velocity_distributions, plot_velocity_vs_radius_blobs
        print("Creating velocity distributions plot...")
        fig, _ = plot_velocity_distributions(sol, G, t_f, dt, length, n_part, solver)
        fig.savefig(os.path.join(base_dir, "velocity_distributions.png"))
        fig2 = plot_velocity_vs_radius_blobs(sol, blobs_params, G, masses, softening)
        fig2.savefig(os.path.join(base_dir, "velocity_wrt_radius.png"))
        print("Velocity distributions plot saved successfully")
 
    if plot_settings['generate_video'].get("do"):
        print("Creating simulation video...")
        from plotting import create_video
        video_path = os.path.join(base_dir, "simulation_video.mp4")
        fps = plot_settings['generate_video'].get("video_fps", 10)
        dpi = plot_settings['generate_video'].get("video_dpi", 100)
        
        create_video(sol, length, G, t_f, dt, n_part, 
                    save_path=video_path, fps=fps, dpi=dpi, density_scaling=density_scaling, solver=solver,
                    softening=softening, masses=masses,
                    enable_energy_tracking=enable_energy_tracking,
                    energy_data=energy_data)
        print("Simulation video saved successfully")
        
    # Save init_params for reference
    # Save a copy of the config file in the result directory
    shutil.copy(config_path, os.path.join(base_dir, "config.yaml"))
    
    if mode == "sim":
        print("Simulation completed.")
        return

    # --- SAMPLING ---
    from likelihood import get_log_posterior
    from sampling import extract_params_to_infer
    from utils import extract_true_values_from_blobs

    print('Starting sampling process...')
    # Prior
    prior_type = config.get("prior_type", None)
    prior_params = config.get("prior_params", None)
    if prior_params is None or prior_type is None:
        raise ValueError("No prior specified in config file. Please provide 'prior_params' and 'prior_type' in your configuration.")
    
    # Likelihood
    likelihood_type = config.get("likelihood_type", None)
    likelihood_kwargs = config.get("likelihood_kwargs", {})
    if likelihood_type is None:
        raise ValueError("No likelihood type specified in config file. Please provide 'likelihood_type' in your configuration.")
    model_kwargs = {
        "G": G,
        "length": length,
        "softening": softening,
        "t_f": t_f,
        "dt": dt,
        "m_part": m_part,
        "density_scaling": density_scaling,
        "solver": solver,
        **scaling_kwargs
    }
    
    # Posterior
    log_posterior = get_log_posterior(
        likelihood_type, 
        data, 
        prior_params=prior_params,
        prior_type=prior_type,
        model_fn=model_fn,
        init_params=blobs_params,
        **model_kwargs,
        **likelihood_kwargs
    )

    # Sampling initialization
    initial_position = config.get("initial_position", None)
    if initial_position is None:
        raise ValueError("No initial position specified in config file. Please provide 'initial_position' in your configuration.")
    rng_key = jax.random.PRNGKey(config.get("sampling_seed", 12345))
    num_samples = config.get("num_samples", 1000)
    progress_bar = config.get("progress_bar", False)

    # Sanity check
    print("="*50)
    print("PARAMETER SAMPLING VERIFICATION")
    print("="*50)
    all_possible_params = extract_params_to_infer(blobs_params)
    print(f"All possible parameters: {list(all_possible_params.keys())}")
    print(f"Parameters with priors: {list(prior_params.keys())}")
    print(f"Parameters being sampled: {list(initial_position.keys())}")
    print(f"Fixed parameters: {set(all_possible_params.keys()) - set(initial_position.keys())}")

    # Run the sampler
    if config["sampler"] == "hmc":
        from sampling import run_hmc
        print("Running HMC sampler...")
        inv_mass_matrix = np.array(config["inv_mass_matrix"])
        step_size = config.get("step_size", 1e-3)
        num_integration_steps = config.get("num_integration_steps", 50)
        num_warmup = config.get("num_warmup", 1000)
        samples = run_hmc(
            log_posterior,
            initial_position,
            inv_mass_matrix,
            step_size,
            num_integration_steps,
            rng_key,
            num_samples,
            num_warmup,  
            progress_bar
        )

    elif config["sampler"] == "nuts":
        print("Running NUTS sampler...")
        from sampling import run_nuts
        num_warmup = config.get("num_warmup", 1000)
        samples = run_nuts(
            log_posterior,
            initial_position,
            rng_key,
            num_samples,
            num_warmup,
            progress_bar=progress_bar  
        )
    
    elif config["sampler"] == "rwm":
        print("Running Random Walk Metropolis sampler...")
        from sampling import run_rwm
        step_size = config.get("step_size", 0.1)
        samples = run_rwm(
            log_posterior,
            initial_position,
            step_size,
            rng_key,
            num_samples,
            progress_bar=progress_bar
        )
    
    elif config["sampler"] == "mala":
        print("Running MALA sampler...")
        from sampling import run_mala
        step_size = config.get("step_size", 0.01)
        num_warmup = config.get("num_warmup", 0)  
        autotuning = config.get("autotuning", False)
        samples = run_mala(
            log_posterior,
            initial_position,
            step_size,
            rng_key,
            num_samples,
            num_warmup=num_warmup,  
            autotuning=autotuning,
            progress_bar=progress_bar
        )
    
    else:
        raise ValueError("Unknown sampler: should be 'hmc', 'nuts', 'rwm', or 'mala'")
    
    print("Sampling finished.")
    
    samples_dict = {}
    for key, value in samples.items():
        if isinstance(value, list) and all(hasattr(v, 'ndim') for v in value):
            # Stack list of arrays into a single 2D array
            samples_dict[key] = jnp.stack(value, axis=1)
        else:
            # Keep as is for scalar parameters
            samples_dict[key] = value
    
    np.savez(os.path.join(base_dir, "samples.npz"), **{k: np.array(v) for k, v in samples_dict.items()})
    print(f"Results saved to {os.path.join(base_dir, 'samples.npz')}")   

    inferred_keys = list(samples_dict.keys())

    theta = {}
    all_true_values = extract_true_values_from_blobs(blobs_params)
    # Build theta dict for only the parameters that were actually sampled
    for key in inferred_keys:
        if key in all_true_values:
            theta[key] = all_true_values[key]
        else:
            theta[key] = None
            print(f"⚠️  Warning: No true value found for '{key}' in blobs_params.")

    param_order = tuple(inferred_keys)

    # Sampling plots
    from plotting import plot_trace_subplots, plot_corner_after_burnin
    from utils import format_prior_info, format_initial_pos

    # Build info strings for each parameter
    param_info = {}
    for param_name in inferred_keys:
        prior_str = format_prior_info(param_name, prior_params, prior_type)
        init_str = format_initial_pos(param_name, initial_position)
        param_info[param_name] = f"{prior_str} | {init_str}"

    # Create a list of strings for the suptitle
    info_lines = [f"{param}: {info}" for param, info in param_info.items()]
    max_per_line = 1
    lines = [" | ".join(info_lines[i:i+max_per_line]) for i in range(0, len(info_lines), max_per_line)]

    fig, _ = plot_trace_subplots(
        samples_dict,
        theta=theta,
        G=G, t_f=t_f, dt=dt, softening=softening, length=length, n_part=n_part,
        method=config["sampler"],
        param_order=param_order
    )
    existing_suptitle = fig._suptitle.get_text() if fig._suptitle else "Trace Plot"
    new_suptitle = existing_suptitle + "\n" + "\n".join(lines)
    fig.suptitle(new_suptitle, fontsize=10)
    fig.savefig(os.path.join(base_dir, "trace_sampling.png"))
    print("Trace plot saved.")

    burnin = num_warmup if config["sampler"] in ["nuts", "hmc", "mala"] else 0
    fig = plot_corner_after_burnin(
        samples_dict,
        theta=theta,
        G=G, t_f=t_f, dt=dt, softening=softening, length=length, n_part=n_part,
        method=config["sampler"],
        param_order=param_order,
        burnin=burnin
    )
    existing_suptitle = fig._suptitle.get_text() if fig._suptitle else "Corner Plot"
    new_suptitle = existing_suptitle + "\n" + "\n".join(lines)
    fig.suptitle(new_suptitle, fontsize=10)
    fig.savefig(os.path.join(base_dir, "corner_sampling.png"))
    print("Corner plot saved.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCMC sampling for N-body initial distribution parameters inference")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    set_cuda_device_from_config(args.config)
    main(args.config)