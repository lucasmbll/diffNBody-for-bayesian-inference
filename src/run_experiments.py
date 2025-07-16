# run_experiment.py

import os
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0" # Set memory fraction for JAX
import yaml
import datetime 
import argparse
import shutil
import numpy as np

def gpu_config(config):
    cuda_num = config.get("cuda_visible_devices", None)
    if cuda_num is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_num)
        print(f"CUDA device set to: {cuda_num}")
        
# Main function to run the experiment
def main(config_path):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    gpu_config(config)

    import jax
    import jax.numpy as jnp
    from model import model 

    mode = config.get("mode")
    if mode is None:
        raise ValueError("Please specify the mode in the configuration file under 'mode' key : 'sim' or 'sampling'.")
    if mode not in ["sim", "sampling", "mle"]:
        raise ValueError(f"Invalid mode: {mode}. Must be either 'sim', 'sampling', or 'mle'.")

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

    G = model_params.get("G", 1.0)
    length = model_params.get("length", 64)
    softening = model_params.get("softening", 1.0)
    t_f = model_params.get("t_f", 1.0)
    dt = model_params.get("dt", 0.5)
    solver = model_params.get("solver", "LeapfrogMidpoint") 

    data_seed = model_params.get("data_seed", 0) # Random seed for data generation
    data_key = jax.random.PRNGKey(data_seed)
    
    density_scaling = model_params.get("density_scaling", "none")
    velocity_scaling = model_params.get("velocity_scaling", "none")
    
    blobs_params = model_params.get("blobs_params", [])
    if not blobs_params:
        raise ValueError("Blob parameters 'blobs_params' must be specified in the configuration file.")
    
    n_part = sum(blob['n_part'] for blob in blobs_params)
    total_mass = sum(blob['n_part'] * blob.get('m_part', 1.0) for blob in blobs_params)

    if mode == "sampling":
        # Extract params to infer 
        initial_position = config.get("initial_position", None)
        if initial_position is None:
            raise ValueError("No initial position specified in config file. Please provide 'initial_position' in your configuration.")
        prior_type = config.get("prior_type", None)
        prior_params = config.get("prior_params", None)
        if prior_params is None or prior_type is None:
            raise ValueError("No prior specified in config file. Please provide 'prior_params' and 'prior_type' in your configuration.")
    elif mode == "mle":
        initial_position = config.get("initial_position", None)
        if initial_position is None:
            raise ValueError("No initial position specified in config file. Please provide 'initial_position' in your configuration.")
        prior_params = None
    else:
        initial_position = None
        prior_params = None

    from utils import blobs_params_init
    data_params, fixed_params, other_params, params_infos = blobs_params_init(blobs_params, prior_params, initial_position, mode)


    # Create model function with fixed parameters
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
        )
    result = model_fn(
        data_params,
        data_key
    )
    
    initial_field, final_field, sol_ts, sol_ys, masses = result

    if mode == "sim":
        print(f"Simulation completed.")
        print(f"Total particles: {n_part}, Total mass: {total_mass:.2f}")
    else:
        print(f"Mock data generated.")
        print(f"Total particles: {n_part}, Total mass: {total_mass:.2f}") 
    
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
            
            for i in range(len(sol_ts)):
                pos_t = sol_ys[i, 0]
                vel_t = sol_ys[i, 1]
                ke, pe, te = calculate_energy_variable_mass(pos_t, vel_t, masses, G, length, softening)
                all_times.append(sol_ts[i])
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
        from sim_plots import plot_density_fields_and_positions, plot_position_vs_radius_blobs
        print("Creating density fields and positions plot...")
        fig = plot_density_fields_and_positions(
            G, t_f, dt, length, n_part, initial_field, sol_ys[0, 0], final_field, sol_ys[-1, 0],
            density_scaling=density_scaling, solver=solver,)
        fig.savefig(os.path.join(base_dir, "density.png"))
        fig2 = plot_position_vs_radius_blobs(sol_ts, sol_ys, blobs_params, length, time_idx=0)
        fig2.savefig(os.path.join(base_dir, "position_vs_radius_blobs.png"))
        print("Density fields and positions plots saved successfully")
    
    if plot_settings['timeseries_plot'].get("do"):
        from sim_plots import plot_timesteps
        print("Creating timesteps plot...")
        plot_timesteps_num = config.get("plot_timesteps", 10)
        fig, _ = plot_timesteps(sol_ts, sol_ys, length, G, t_f, dt, n_part, num_timesteps=plot_timesteps_num, softening=softening, masses=masses, solver=solver,
                                enable_energy_tracking=enable_energy_tracking, density_scaling=density_scaling,
                                energy_data=energy_data)
        fig.savefig(os.path.join(base_dir, "timesteps.png"))
        print("Timesteps plot saved successfully")
    
    if plot_settings['trajectories_plot'].get("do"):
        from sim_plots import plot_trajectories
        print("Creating trajectories plot...")
        num_trajectories = plot_settings['trajectories_plot'].get("num_trajectories", 10)
        zoom = plot_settings['trajectories_plot'].get("zoom_for_trajectories", True)
        fig = plot_trajectories(sol_ys, G, t_f, dt, length, n_part, solver, num_trajectories=num_trajectories, 
                                zoom=zoom)
        fig.savefig(os.path.join(base_dir, "trajectories.png"))
        print("Trajectories plot saved successfully")
    
    if plot_settings['velocity_plot'].get("do"):
        from sim_plots import plot_velocity_distributions, plot_velocity_vs_radius_blobs
        print("Creating velocity distributions plot...")
        fig = plot_velocity_distributions(initial_field, final_field, sol_ys, G, t_f, dt, length, n_part, solver)
        fig.savefig(os.path.join(base_dir, "velocity.png"))
        fig2 = plot_velocity_vs_radius_blobs(sol_ts, sol_ys, blobs_params, G, masses, softening)
        fig2.savefig(os.path.join(base_dir, "velocity_wrt_radius.png"))
        print("Velocity distributions plot saved successfully")

    if plot_settings['generate_video'].get("do"):
        print("Creating simulation video...")
        from sim_plots import create_video
        video_type = plot_settings['generate_video'].get("video_type", "particles")
        if video_type not in ["particles", "density", "velocity"]:
            print(f"Warning: Unknown video type '{video_type}'. Defaulting to 'particles'.")
            video_type = "particles"
        
        video_path = os.path.join(base_dir, f"simulation_{video_type}.mp4")
        fps = plot_settings['generate_video'].get("video_fps", 10)
        dpi = plot_settings['generate_video'].get("video_dpi", 100)
        
        create_video(
            sol_ts, sol_ys, length, G, t_f, dt, n_part, 
            save_path=video_path, 
            fps=fps, 
            dpi=dpi, 
            density_scaling=density_scaling, 
            solver=solver,
            video_type=video_type,
            softening=softening, 
            masses=masses,
            enable_energy_tracking=enable_energy_tracking,
            energy_data=energy_data
        )
        print(f"Simulation {video_type} video saved successfully")
        
    shutil.copy(config_path, os.path.join(base_dir, "config.yaml")) # Save a copy of the config file in the result directory
    
    if mode == "sim":
        print("Simulation completed.")
        return

    # Load data
    data = final_field

    # Likelihood
    likelihood_type = config.get("likelihood_type", None)
    likelihood_kwargs = config.get("likelihood_kwargs", {})
    noise = likelihood_kwargs.get("noise", 1.0)

    # Initial parameters for sampling or MLE
    from utils import sampling_params_init
    init_params = sampling_params_init(params_infos, initial_position)

    if likelihood_type is None:
        raise ValueError("No likelihood type specified in config file. Please provide 'likelihood_type' in your configuration.")

    
    if mode == "sampling":
        print('Starting sampling process...')

        # Prior 
        from utils import prior_params_extract
        prior_params_array = prior_params_extract(prior_type, prior_params, params_infos)

        # --- SAMPLING ---   
        from likelihood import log_posterior
        # Posterior
        log_posterior_fn = lambda params: log_posterior(params, likelihood_type, data, model_fn, data_key, prior_params_array, prior_type, noise)

        # Sampling initialization
        rng_key = jax.random.PRNGKey(config.get("sampling_seed", 12345))
        num_samples = config.get("num_samples", 1000)
        
        progress_bar = config.get("progress_bar", False)

        if config["sampler"] == "nuts":
            from sampling import run_nuts
            num_warmup = config.get("num_warmup", 1000)
            print("Running NUTS sampler...")
            print(f"Warmup steps: {num_warmup}")
            print(f"Sampling steps: {num_samples}")
            print(f"Initial position: {initial_position}")
            samples = run_nuts(
                log_posterior_fn,
                init_params,
                rng_key,
                num_samples,
                num_warmup,
                progress_bar=progress_bar
            )

        # Run the sampler
        elif config["sampler"] == "hmc": #TO BE ADAPTED for warmup
            from sampling import run_hmc
            num_integration_steps = config.get("num_integration_steps", 50)
            num_warmup = config.get("num_warmup", 1000)
            print("Running HMC sampler...")
            print(f"Warmup steps: {num_warmup}")
            print(f"Sampling steps: {num_samples}")
            print(f"Initial position: {initial_position}")
            print(f"HMC parameters: num_integration_steps={num_integration_steps}")
            samples = run_hmc(log_posterior_fn, init_params, num_integration_steps, rng_key, num_samples, num_warmup, progress_bar=progress_bar)
            
        
        elif config["sampler"] == "rwm": 
            print("Running Random Walk Metropolis sampler...")
            from sampling import run_rwm
            step_size = config.get("step_size", 0.1)
            samples = run_rwm(
                log_posterior_fn,
                init_params,
                step_size,
                rng_key,
                num_samples
            )
        """
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
                autotuning=autotuning
            )

        else:
            raise ValueError("Unknown sampler: should be 'hmc', 'nuts', 'rwm', or 'mala'")
        """
        samples_dict = {}
        for i in range(len(params_infos)):
            for j, key in enumerate(params_infos[i]['changing_param_order']):
                samples_dict[f'blob{i}_{key}'] = samples[:, i, j]  
        np.savez(os.path.join(base_dir, "samples.npz"), **{k: np.array(v) for k, v in samples_dict.items()})
        print(f"Results saved to {os.path.join(base_dir, 'samples.npz')}")
        
        data_dict = {} 
        for i in range(len(params_infos)):
            for j, key in enumerate(params_infos[i]['changing_param_order']):
                data_dict[f'blob{i}_{key}'] = data_params[i, j]  
        np.savez(os.path.join(base_dir, "truth.npz"), **{k: np.array(v) for k, v in data_dict.items()})
        print(f"True values saved to {os.path.join(base_dir, 'truth.npz')}")
        return
    
    else:  # mode == "mle"
        # --- MLE OPTIMIZATION ---
        print("Starting MLE optimization...")
        
        from likelihood import log_likelihood_1
        # For MLE, we only need the likelihood (no prior)
        log_likelihood_fn = lambda params: log_likelihood_1(params, data, noise, model_fn, data_key)
        
        # MLE configuration
        mle_config = config.get("mle", {})
        optimizer_name = mle_config.get("optimizer", "adam")
        learning_rate = mle_config.get("learning_rate", 0.001)
        num_iterations = mle_config.get("num_iterations", 1000)
        print_every = mle_config.get("print_every", 100)
        multi_start = mle_config.get("multi_start", False)
        num_starts = mle_config.get("num_starts", 5)
        perturbation_scale = mle_config.get("perturbation_scale", 0.1)
        
        # Run optimization
        from mle import run_mle_optimization, run_multi_start_mle, create_random_initial_positions
        
        if multi_start:
            # Create multiple starting points
            initial_positions = create_random_initial_positions(
                init_params, num_starts, perturbation_scale
            )
            
            # Run multi-start optimization
            best_params, best_history, all_results = run_multi_start_mle(
                log_likelihood_fn, initial_positions, optimizer_name, 
                learning_rate, num_iterations, print_every
            )
            
            # Save all results
            mle_results = {
                'best_params': best_params,
                'best_history': best_history,
                'all_results': all_results,
                'config': mle_config
            }
            
        else:
            # Single optimization run
            best_params, history = run_mle_optimization(
                log_likelihood_fn, init_params, optimizer_name, 
                learning_rate, num_iterations, print_every
            )
            
            mle_results = {
                'best_params': best_params,
                'history': history,
                'config': mle_config
            }
        print(f"MLE results: {mle_results['best_params']}")
        print(f"Best log-likelihood: {-history['loss'][-1]}")
        
        # Save MLE results
        np.savez(os.path.join(base_dir, "mle_results.npz"), **mle_results)
        
        # Convert MLE results to same format as MCMC samples for comparison
        mle_dict = {}
        for i in range(len(params_infos)):
            for j, key in enumerate(params_infos[i]['changing_param_order']):
                mle_dict[f'blob{i}_{key}'] = np.array([best_params[i, j]])
        
        np.savez(os.path.join(base_dir, "mle_point_estimate.npz"), **mle_dict)
        
        # Save true values for comparison
        data_dict = {}
        for i in range(len(params_infos)):
            for j, key in enumerate(params_infos[i]['changing_param_order']):
                data_dict[f'blob{i}_{key}'] = data_params[i, j]
        
        np.savez(os.path.join(base_dir, "truth.npz"), **data_dict)
        
        print(f"MLE results saved to {base_dir}")
        print("MLE optimization completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCMC sampling for N-body initial distribution parameters inference")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)