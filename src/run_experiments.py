# run_experiment.py

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0" # Set memory fraction to 100% for JAX

import sys
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
import jax.numpy as jnp
import numpy as np
from model import gaussian_model, gaussian_2blobs  # Import the renamed model
from likelihood import get_log_posterior
from sampling import run_hmc, run_nuts
from plotting import (
    plot_density_fields_and_positions,
    plot_timesteps,
    plot_trajectories,
    plot_velocity_distributions,
    plot_trace_subplots,
    plot_corner_after_burnin,
)

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    mode = config.get("mode", "sampling")  # Default to "sampling" for backward compatibility
    
    # Validate mode parameter
    if mode not in ["sim", "sampling"]:
        raise ValueError(f"Invalid mode: {mode}. Must be either 'sim' or 'sampling'")

    # --- Output directory logic ---
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_base = os.path.splitext(os.path.basename(config_path))[0]
    if mode == "sampling":
        base_dir = os.path.join("results", "sampling_results", f"{config_base}_{now_str}")
    else:  # mode == "sim"
        base_dir = os.path.join("results", "nbody_sim_results", f"{config_base}_{now_str}")
    os.makedirs(base_dir, exist_ok=True)

    # --- Model selection ---
    model_params = config["model_params"]
    model_type = model_params.get("model_type", "gaussian")  # Default to "gaussian"
    model_registry = {
        "gaussian": gaussian_model,
        "gaussian_2blobs": gaussian_2blobs,
        # Add more models here as needed
    }
    if model_type not in model_registry:
        raise ValueError(f"Unknown model_type: {model_type}. Available: {list(model_registry.keys())}")
    model_fn = model_registry[model_type]

    # --- Generate data using model and model_params ---
    if mode == "sim":
        print(f"Running simulation using {model_type}_model...")
    else:
        print(f"Generating mock data using {model_type}_model...")
    # Extract parameters for the selected model
    params = model_params.get("params", None)
    if params is None:
        raise ValueError("Model parameters 'params' must be specified in the configuration file.")
    params = jnp.array(params)
    n_part = model_params.get("n_part", 1000)
    G = model_params.get("G", 5.0)
    length = model_params.get("length", 64)
    softening = model_params.get("softening", 0.1)
    t_f = model_params.get("t_f", 1.0)
    dt = model_params.get("dt", 0.5)
    m_part = model_params.get("m_part", 1.0)  # Default mass per particle
    random_vel = model_params.get("random_vel", True)  # Default to True for backward compatibility
    
    # Extract density scaling parameters
    density_scaling = model_params.get("density_scaling", "none")
    scaling_kwargs = model_params.get("scaling_kwargs", {})
    
    # Optionally: allow for random seed
    data_seed = model_params.get("data_seed", 0)
    data_key = jax.random.PRNGKey(data_seed)
    
    # Run model to generate mock data (only the output field is needed)
    input_field, init_pos, final_pos, output_field, sol = model_fn(
        params,
        n_part=n_part,
        G=G,
        length=length,
        softening=softening,
        t_f=t_f,
        dt=dt,
        m_part=m_part,
        key=data_key,
        random_vel=random_vel,
        density_scaling=density_scaling,
        **scaling_kwargs
    )
    if mode == "sim":
        print(f"Simulation completed.")
        print(f"Density scaling applied: {density_scaling}")
        if density_scaling != "none":
            print(f"Scaling parameters: {scaling_kwargs}")
    else:
        print(f"Mock data generated.")
        print(f"Density scaling applied: {density_scaling}")

    data = output_field  # This is the mock data we will use for inference

    # --- Save plots for generated mock data ---
    try:
        print("Creating density fields and positions plot...")
        # 1. Density fields and positions
        fig = plot_density_fields_and_positions(
            G, t_f, dt, length, n_part, input_field, init_pos, final_pos, output_field, density_scaling=density_scaling, random_vel=random_vel)
        fig.savefig(os.path.join(base_dir, "density_fields_and_positions.png"))
        print("Density fields and positions plots saved successfully")
        
        # Pre-compute energy if needed for any plots
        enable_energy_tracking = config.get("enable_energy_tracking", True)
        energy_data = None
        
        if enable_energy_tracking:
            print("Pre-calculating energy for all timesteps...")
            from plotting import calculate_energy
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
        
        print("Creating timesteps plot...")
        # 2. All timesteps
        plot_timesteps_num = config.get("plot_timesteps", 10)
        fig, _ = plot_timesteps(sol, length, G, t_f, dt, n_part, num_timesteps=plot_timesteps_num, 
                               random_vel=random_vel, softening=softening, m_part=m_part,
                               enable_energy_tracking=enable_energy_tracking, density_scaling=density_scaling,
                               energy_data=energy_data)
        fig.savefig(os.path.join(base_dir, "timesteps.png"))
        print("Timesteps plot saved successfully")
        
        print("Creating trajectories plot...")
        # 3. Trajectories
        num_trajectories = config.get("num_trajectories", 10)
        zoom = config.get("zoom", True)
        fig = plot_trajectories(sol, G, t_f, dt, length, n_part, num_trajectories=num_trajectories, zoom=zoom, random_vel=random_vel)
        fig.savefig(os.path.join(base_dir, "trajectories.png"))
        print("Trajectories plot saved successfully")
        
        print("Creating velocity distributions plot...")
        # 4. Velocity distributions
        fig, _ = plot_velocity_distributions(sol, G, t_f, dt, length, n_part, random_vel=random_vel)
        fig.savefig(os.path.join(base_dir, "velocity_distributions.png"))
        print("Velocity distributions plot saved successfully")
    
        # 5. Video generation (if enabled)
        generate_video = config.get("generate_video", False)
        if generate_video:
            print("Creating simulation video...")
            try:
                from plotting import create_video
                video_path = os.path.join(base_dir, "simulation_video.mp4")
                fps = config.get("video_fps", 10)
                dpi = config.get("video_dpi", 100)
                
                create_video(sol, length, G, t_f, dt, n_part, 
                            save_path=video_path, fps=fps, dpi=dpi, density_scaling=density_scaling, random_vel=random_vel,
                            softening=softening, m_part=m_part,
                            enable_energy_tracking=enable_energy_tracking,
                            energy_data=energy_data)
                print("Simulation video saved successfully")
            except ImportError as e:
                print(f"Warning: Could not create video. Missing dependencies: {e}")
                print("Install ffmpeg and matplotlib to enable video generation")
            except Exception as e:
                print(f"Error creating video: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"Error creating data plots: {e}")
        import traceback
        traceback.print_exc()

    if mode == "sim":
        print("Running in simulation mode - skipping sampling.")
        return

    # --- Pass model_fn to likelihood ---
    prior_type = config.get("prior_type", "gaussian")
    
    log_posterior = get_log_posterior(
        config["likelihood_type"], 
        data, 
        prior_params=config.get("prior_params", None),
        prior_type=prior_type,
        model_fn=model_fn,  # Pass the selected model function
        # Pass model parameters to likelihood functions
        n_part=n_part,
        G=G,
        length=length,
        softening=softening,
        t_f=t_f,
        dt=dt,
        m_part=m_part,
        random_vel=random_vel,  # Pass random_vel as fixed parameter
        density_scaling=density_scaling,  # Pass density scaling
        **scaling_kwargs,  # Pass scaling parameters
        **config.get("likelihood_kwargs", {})
    )

    # 3. Initialisation des paramètres
    initial_position = config["initial_position"]
    
    # Check if we should infer vel_sigma based on random_vel
    infer_vel_sigma = random_vel
    
    # Remove vel_sigma from initial_position if not inferring it
    if not infer_vel_sigma:
        initial_position = {k: v for k, v in initial_position.items() if k != "vel_sigma"}
        print("Note: vel_sigma excluded from inference since random_vel=False")
    
    rng_key = jax.random.PRNGKey(config.get("sampling_seed", 12345))
    num_samples = config.get("num_samples", 5000)
    progress_bar = config.get("progress_bar", False)
    
    # 4. Lancer le sampler
    print(f"Starting sampling with {config['sampler'].upper()} sampler...")
    print(f"Number of samples: {num_samples}")
    print(f"Parameters to infer: {list(initial_position.keys())}")
    
    if config["sampler"] == "hmc":
        inv_mass_matrix = np.array(config["inv_mass_matrix"])
        
        # Adjust inv_mass_matrix if not inferring vel_sigma
        if not infer_vel_sigma and len(inv_mass_matrix) == 3:
            inv_mass_matrix = inv_mass_matrix[:2]  # Only sigma and mean
            print(f"Adjusted inv_mass_matrix for 2 parameters: {inv_mass_matrix}")
        
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

    # --- Save plots for sampling results ---
    # Convert samples to dict if needed and handle missing vel_sigma
    if isinstance(samples, dict):
        samples_dict = samples
    else:
        if infer_vel_sigma:
            samples_dict = {
                "sigma": samples[:, 0],
                "mean": samples[:, 1],
                "vel_sigma": samples[:, 2],
            }
        else:
            samples_dict = {
                "sigma": samples[:, 0],
                "mean": samples[:, 1],
                "vel_sigma": np.zeros(len(samples[:, 0])),  # Dummy values for plotting
            }
    
    # Extract true parameter values for plotting
    true_params = model_params.get("params", [10.0, 30.0, 1.0])
    sigma = true_params[0] if len(true_params) > 0 else 10.0
    mean = true_params[1] if len(true_params) > 1 else 30.0
    vel_sigma = true_params[2] if len(true_params) > 2 else 1.0
    
    # Determine which parameters to plot
    if infer_vel_sigma:
        param_order = ("sigma", "mean", "vel_sigma")
        theta = {"pos_std": sigma, "pos_mean": mean, "vel_std": vel_sigma}
    else:
        param_order = ("sigma", "mean")
        theta = {"pos_std": sigma, "pos_mean": mean}
    
    # Trace plots
    fig, _ = plot_trace_subplots(
        samples_dict,
        theta=theta,
        G=G, t_f=t_f, dt=dt, softening=softening, length=length, n_part=n_part,
        method=config.get("sampler", "sampler").upper(),
        random_vel=random_vel,
        param_order=param_order,
        infer_vel_sigma=infer_vel_sigma
    )
    fig.savefig(os.path.join(base_dir, "trace_sampling.png"))
    
    # Corner plot
    fig = plot_corner_after_burnin(
        samples_dict,
        theta=theta,
        burnin=config.get("num_warmup", 1000),
        param_order=param_order,
        infer_vel_sigma=infer_vel_sigma
    )
    fig.savefig(os.path.join(base_dir, "corner_sampling.png"))

    # 5. Sauvegarde des résultats
    if isinstance(samples, dict):
        np.savez(os.path.join(base_dir, "samples.npz"), **{k: np.array(v) for k, v in samples.items()})
    else:
        np.savez(os.path.join(base_dir, "samples.npz"), samples=np.array(samples))
    print(f"Results saved to {os.path.join(base_dir, 'samples.npz')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCMC sampling for N-body initial distribution parameters inference")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    set_cuda_device_from_config(args.config)
    main(args.config)
