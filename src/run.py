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
    if mode not in ["sim", "sampling", "mle", "grid"]:
        raise ValueError(f"Invalid mode: {mode}. Must be either 'sim', 'sampling', 'mle', or 'grid'.")

    # --- Output directory ---
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_base = os.path.splitext(os.path.basename(config_path))[0]
    if mode == "sampling":
        base_dir = os.path.join("results", "sampling_results", f"{config_base}_{now_str}")
    elif mode == "grid":
        base_dir = os.path.join("results", "grid_results", f"{config_base}_{now_str}")
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
    elif mode == "grid":
        initial_position = None
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

    observable = model_params.get("observable", ["density", "vx", "vy", "vz"])

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
            observable=observable
        )

    result = model_fn(
        data_params,
        data_key
    )

    initial_field, final_field, final_observable_field, sol_ts, sol_ys, masses = result 

    print('Compute FFT and Power Spectrum of the simulation outputs')

    from utils import compute_fft_and_power_spectrum_3d_fields

    fft_initial, k_center_initial, power_spectrum_initial = compute_fft_and_power_spectrum_3d_fields(initial_field, length)
    fft_final, k_center_final, power_spectrum_final = compute_fft_and_power_spectrum_3d_fields(final_field, length)
    
    if mode == "sim":
        print(f"Simulation completed.")
        print(f"Total particles: {n_part}, Total mass: {total_mass:.2f}")
    else:
        print(f"Mock data generated.")
        print(f"Total particles: {n_part}, Total mass: {total_mass:.2f}") 
    
    # --- Plots and video ---
    plot_settings = config.get("plot_settings", {})
    units = plot_settings["units"]
    kpc_per_pixel = units.get("kpc_per_pixel", 1.0)  # kpc per pixel for density field plots
    kpc_per_pixel = float(kpc_per_pixel)
    msun_per_mass_unit = units.get("msun_per_mass_unit", 1.0)  # Solar mass per mass unit
    msun_per_mass_unit = float(msun_per_mass_unit)  # Ensure it's a float
    gyr_per_time_unit = units.get("gyr_per_time_unit", 1.0)  # Gyr per time unit
    gyr_per_time_unit = float(gyr_per_time_unit)  # Ensure it's a float
    jouleE50_per_unit = units.get("jouleE50_per_unit", 1.0)  # Energy unit conversion factor
    jouleE50_per_unit = float(jouleE50_per_unit)  # Ensure it's a float
    
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
            density_scaling=density_scaling, solver=solver, kpc_per_pixel=kpc_per_pixel)
        fig.savefig(os.path.join(base_dir, "density.png"))
        fig2 = plot_position_vs_radius_blobs(sol_ts, sol_ys, blobs_params, length, time_idx=0)
        fig2.savefig(os.path.join(base_dir, "position_vs_radius_blobs.png"))
        print("Density fields and positions plots saved successfully")
    
    if plot_settings['timeseries_plot'].get("do"):
        from sim_plots import plot_timesteps, plot_lagrangian_radii_fractionwise
        print("Creating timesteps plot...")
        plot_timesteps_num = config.get("plot_timesteps", 10)
        fig, _ = plot_timesteps(sol_ts, sol_ys, length, G, t_f, dt, n_part, num_timesteps=plot_timesteps_num, softening=softening, masses=masses, solver=solver,
                                enable_energy_tracking=enable_energy_tracking, density_scaling=density_scaling,
                                energy_data=energy_data, kpc_per_pixel=kpc_per_pixel, msun_per_mass_unit=msun_per_mass_unit,gyr_per_time_unit=gyr_per_time_unit, jouleE50_per_unit=jouleE50_per_unit)
        fig.savefig(os.path.join(base_dir, "timesteps.png"))
        fig2 = plot_lagrangian_radii_fractionwise(
            sol_ts=sol_ts,
            sol_ys=sol_ys,
            blobs_params=blobs_params,
            masses=masses,
            length=length,
            kpc_per_pixel=kpc_per_pixel,
            gyr_per_time_unit=gyr_per_time_unit,
            fractions=(0.1, 0.5, 0.9)
        )
        fig2.savefig(os.path.join(base_dir, "lagrangian_radii_fractionwise.png"))
        print("Timesteps plot saved successfully")
    
    if plot_settings['trajectories_plot'].get("do"):
        from sim_plots import plot_trajectories
        print("Creating trajectories plot...")
        num_trajectories = plot_settings['trajectories_plot'].get("num_trajectories", 10)
        zoom = plot_settings['trajectories_plot'].get("zoom_for_trajectories", True)
        fig = plot_trajectories(sol_ys, G, t_f, dt, length, n_part, solver, num_trajectories=num_trajectories, 
                                zoom=zoom, kpc_per_pixel=kpc_per_pixel,)
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

    if plot_settings.get('fft_plot', {}).get("do", False):
        from sim_plots import plot_fft_fields
        print("Creating FFT fields plot...")
        fig_fft = plot_fft_fields(
            fft_initial, fft_final, length, G, t_f, dt, n_part, solver, kpc_per_pixel)
        fig_fft.savefig(os.path.join(base_dir, "fft_fields.png"), dpi=150, bbox_inches='tight')
        print("FFT fields plot saved successfully")

    if plot_settings.get('power_spectrum_plot', {}).get("do", False):
        from sim_plots import plot_power_spectra
        print("Creating power spectra plot...")
        fig_ps, fig_ps_comp = plot_power_spectra(
            k_center_initial, power_spectrum_initial, k_center_final, power_spectrum_final,
            length, G, t_f, dt, n_part, solver, kpc_per_pixel)
        fig_ps.savefig(os.path.join(base_dir, "power_spectra.png"), dpi=150, bbox_inches='tight')
        fig_ps_comp.savefig(os.path.join(base_dir, "power_spectra_comparison.png"), dpi=150, bbox_inches='tight')
        print("Power spectra plots saved successfully")
        
    shutil.copy(config_path, os.path.join(base_dir, "config.yaml")) # Save a copy of the config file in the result directory

    if config.get("save", True ):
        print("Saving final fields data...")
        fields_data = {
            'initial_field': np.array(initial_field),
            'final_field': np.array(final_field),
            'final_observable_field': np.array(final_observable_field),
            'field_names': ['density', 'vx', 'vy', 'vz'],
            'observable_names': observable,
            'box_size': length,
            'kpc_per_pixel': kpc_per_pixel,
            'density_scaling': density_scaling,
            'velocity_scaling': velocity_scaling
        }
        np.savez_compressed(os.path.join(base_dir, "final_fields.npz"), **fields_data)

        # Save FFT data  
        print("Saving FFT data...")
        fft_data = {
            'fft_initial_real': np.array(jnp.real(fft_initial)),
            'fft_initial_imag': np.array(jnp.imag(fft_initial)),
            'fft_final_real': np.array(jnp.real(fft_final)),
            'fft_final_imag': np.array(jnp.imag(fft_final)),
            'field_names': ['density', 'vx', 'vy', 'vz'],
            'box_size': length,
            'kpc_per_pixel': kpc_per_pixel,
            'grid_shape': list(initial_field.shape[:3])
        }
        np.savez_compressed(os.path.join(base_dir, "fft_data.npz"), **fft_data)

        # Save power spectrum data
        print("Saving power spectrum data...")
        power_spectrum_data = {
            'k_center_initial': np.array(k_center_initial),
            'power_spectrum_initial': np.array(power_spectrum_initial),
            'k_center_final': np.array(k_center_final),
            'power_spectrum_final': np.array(power_spectrum_final),
            'field_names': ['density', 'vx', 'vy', 'vz'],
            'box_size': length,
            'kpc_per_pixel': kpc_per_pixel,
            'units': {
                'k_units': 'kpc^-1',
                'power_units': 'arbitrary',
                'box_size_units': 'pixels'
            }
        }
        np.savez_compressed(os.path.join(base_dir, "power_spectrum_data.npz"), **power_spectrum_data)

        print("Data files saved successfully:")
        print(f"  - Final fields: {os.path.join(base_dir, 'final_fields.npz')}")
        print(f"  - FFT data: {os.path.join(base_dir, 'fft_data.npz')}")
        print(f"  - Power spectrum: {os.path.join(base_dir, 'power_spectrum_data.npz')}")

    
    if mode == "sim":       
        comparison_settings = config.get("comparison_settings", {})
        if comparison_settings.get("do_comparison", False):
            print("Performing field comparison...")
            
            reference_base_path = comparison_settings.get("reference_data_path")
            if reference_base_path:
                # Construct paths to reference data files
                if reference_base_path.endswith('.npz'):
                    # Remove extension if provided
                    reference_base_path = reference_base_path[:-4]
                
                ref_fields_path = f"{reference_base_path}/final_fields.npz"
                ref_fft_path = f"{reference_base_path}/fft_data.npz"
                ref_ps_path = f"{reference_base_path}/power_spectrum_data.npz"
                
                # Check if reference files exist
                if all(os.path.exists(path) for path in [ref_fields_path, ref_fft_path, ref_ps_path]):
                    print(f"Loading reference data from: {reference_base_path}")
                    
                    # Current data paths
                    current_fields_path = os.path.join(base_dir, "final_fields.npz")
                    current_fft_path = os.path.join(base_dir, "fft_data.npz")
                    current_ps_path = os.path.join(base_dir, "power_spectrum_data.npz")
                    
                    # Import comparison functions
                    from comparison_metrics import (
                        compute_all_comparison_metrics_from_files, save_comparison_results_csv, 
                        print_comparison_summary, plot_comparison_summary
                    )
                    
                    # Choose resize method based on settings
                    resize_method = comparison_settings.get("resize_method", "fourier")
                    print(f"Using {resize_method} method for field resizing")
                    
                    # Compute comparison metrics using saved data files
                    comparison_metrics = compute_all_comparison_metrics_from_files(
                        current_fields_path, current_fft_path, current_ps_path,
                        ref_fields_path, ref_fft_path, ref_ps_path,
                        field_names=['density', 'vx', 'vy', 'vz'],
                        resize_method=resize_method
                    )
                    
                    # Save results in both CSV and NPZ formats
                    save_comparison_results_csv(comparison_metrics, os.path.join(base_dir, "comparison_metrics.npz"))
                    print_comparison_summary(comparison_metrics)
                    
                    # Create and save summary plots
                    plot_comparison_summary(comparison_metrics, os.path.join(base_dir, "comparison_summary.png"))
                    print("Comparison summary plots saved successfully")
                    
                else:
                    print("Warning: Some reference data files not found:")
                    print(f"  Fields: {ref_fields_path} {'✓' if os.path.exists(ref_fields_path) else '✗'}")
                    print(f"  FFT: {ref_fft_path} {'✓' if os.path.exists(ref_fft_path) else '✗'}")
                    print(f"  Power Spectrum: {ref_ps_path} {'✓' if os.path.exists(ref_ps_path) else '✗'}")
                    
                    # Fallback to fields-only comparison if available
                    if os.path.exists(ref_fields_path):
                        print("Falling back to fields-only comparison...")
                        from comparison_metrics import compute_all_comparison_metrics
                        
                        ref_data = np.load(ref_fields_path)
                        ref_final_field = ref_data['final_field']
                        ref_box_size = ref_data.get('box_size', length)
                        
                        comparison_metrics = compute_all_comparison_metrics(
                            final_field, ref_final_field, 
                            box_size1=length, box_size2=ref_box_size,
                            field_names=['density', 'vx', 'vy', 'vz'],
                            resize_method=resize_method,
                            k_centers1=k_center_final, power_spec1=power_spectrum_final,
                            k_centers2=None, power_spec2=None,
                            fft1=fft_final, fft2=None
                        )
                        
                        save_comparison_results_csv(comparison_metrics, os.path.join(base_dir, "comparison_metrics.npz"))
                        print_comparison_summary(comparison_metrics)
                        plot_comparison_summary(comparison_metrics, os.path.join(base_dir, "comparison_summary.png"))
                        print("Limited comparison completed (FFT/PS data not available for reference)")
            else:
                print("Warning: Reference data path not specified")

        print("Simulation completed.")
        return

    # Load data
    noise_rate = model_params.get("noise_rate", 0.1)  # Default noise rate
    n_fields = final_observable_field.shape[-1]  
    noise = jnp.zeros_like(final_observable_field) 
    noise_levels = jnp.zeros(n_fields) 
    noise_keys = jax.random.split(data_key, n_fields) 
    for i in range(n_fields):
        field = final_observable_field[..., i]  
        rms_value = jnp.sqrt(jnp.mean(field**2))  
        noise_level = noise_rate * rms_value 
        noise_levels = noise_levels.at[i].set(noise_level) 
        noise = noise.at[..., i].set(jax.random.normal(noise_keys[i], field.shape) * noise_level)  

    # Add noise to the final observable field
    data = final_observable_field + noise
    

    """""
    from sim_plots import plot_density_fields_and_positions
    fig = plot_density_fields_and_positions(
            G, t_f, dt, length, n_part, noise, sol_ys[-1, 0], data, sol_ys[-1, 0],
           density_scaling=density_scaling, solver=solver, kpc_per_pixel=kpc_per_pixel)
    fig.savefig(os.path.join(base_dir, "noise.png")) 
    """""
    
    from likelihood import log_likelihood_1
    log_likelihood_fn = lambda params: log_likelihood_1(params, data, noise_levels, model_fn, data_key)

    if mode in ["sampling", "grid"]:
        # Prior 
        from utils import prior_params_extract
        prior_params_array, params_labels = prior_params_extract(prior_type, prior_params, params_infos)
        from likelihood import log_prior
        log_prior_fn = lambda params: log_prior(params, prior_type, prior_params_array)

        # Posterior
        from likelihood import log_posterior
        log_posterior_fn = lambda params: log_posterior(params, log_likelihood_fn, log_prior_fn)

    """    import time
    start = time.time()
    grad = jax.grad(log_posterior_fn, argnums=0)
    grad_value = grad(data_params)
    end = time.time()
    print(f"Gradient computation took {end - start:.4f} seconds")
    print(f"Gradient at data parameters: {grad_value}")

    import time
    start = time.time()
    grad = jax.grad(log_posterior_fn, argnums=0)
    grad_value = grad(data_params)
    end = time.time()
    print(f"Gradient computation took {end - start:.4f} seconds")
    print(f"Gradient at data parameters: {grad_value}")

    params_modifed = data_params * 1.02
    log_lik_value = log_likelihood_fn(data_params)
    relative_gradient = grad_value / abs(log_lik_value)
    print(f"Log-likelihood value: {log_lik_value}")
    print(f"Relative gradient: {relative_gradient}")
    """
    
    # Initial parameters for sampling or MLE
    if mode in ["sampling", "mle"]:
        from utils import params_init
        init_params = params_init(params_infos, initial_position)
    
    if mode == "sampling":
        print('Starting sampling process...')
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
    
    elif mode == "grid":
        print("Starting grid search...")
        from grid import create_parameter_grid, evaluate_likelihood, evaluate_gradients, plot_chi2_distribution, value_surface, values_slices, quiver_grad_surface
        # Extract hypercube parameters
        hypercube_params = config.get("hypercube_params", {})
        n_points_per_dim = hypercube_params.get("n_points_per_dim", 10)
        param_bounds = hypercube_params.get("param_bounds", {})
        mini_batch_size_value = config['hypercube_params'].get('mini_batch_size_value', 50)
        mini_batch_size_grad = config['hypercube_params'].get('mini_batch_size_grad', 50)
        # Create parameter grid
        print("Creating parameter grid for hypercube search...")
        parameter_sets = create_parameter_grid(
            n_points_per_dim=n_points_per_dim,
            param_bounds=param_bounds,
            params_infos=params_infos
        )
        
        # Evaluate likelihood and gradients on the grid
        print(f"Evaluating likelihood for {len(parameter_sets)} parameter sets using mini-batch computation...")
        log_posterior_values, log_lik_values, chi2_values, evaluation_stats, valid_mask = evaluate_likelihood(parameter_sets, mini_batch_size_value, log_posterior_fn, log_prior_fn)

        # Plotting
        print("Creating likelihood and posterior values marginal surfaces...")
        value_surface(data_params, parameter_sets, log_posterior_values, log_lik_values, params_labels, valid_mask, base_dir)

        print("Creating 1D likelihood and posterior values marginal slices")
        values_slices(data_params, parameter_sets, log_posterior_values, log_lik_values, params_labels, valid_mask, base_dir)

        print("Creating chi2 distribution plot...")
        dof = len(observable) * length^3
        plot_chi2_distribution(chi2_values, dof, base_dir)

       
                # Parameter statistics
        if np.sum(valid_mask) > 0:
            print("=== BEST PARAMETERS ON THE GRID ===\n")
            # Best point
            best_idx = jnp.argmax(log_posterior_values[valid_mask])
            best_point_global = np.where(valid_mask)[0][best_idx]
            print("=== BEST PARAMETER POINT ===\n")
            print(f"Best posterior value: {log_posterior_values[best_point_global]:.6f}\n")
            print(f"Best likelihood value: {log_lik_values[best_point_global]:.6f}\n")
            print("Best parameters: \n")
            best_params = parameter_sets[best_point_global]
            for i in range(len(params_infos)):
                for j, key in enumerate(params_infos[i]['changing_param_order']):
                    print(f"blob{i}_{key}: {best_params[i, j]:.6f}\n")
       
        #if len(parameter_sets) < 50000:
            #print(f"Evaluating gradients for {len(parameter_sets)} parameter sets using mini-batch computation...")
            #posterior_gradient_values, likelihood_gradient_values, evaluation_stats_grad, valid_mask_grad = evaluate_gradients#(parameter_sets, mini_batch_size_grad, log_posterior_fn, log_prior_fn)

            #print("Creating quiver plot of gradients on the parameter surface...")
            #quiver_grad_surface(parameter_sets, log_lik_values, log_posterior_values, likelihood_gradient_values, #posterior_gradient_values, params_labels, valid_mask_grad, base_dir)
        
        #else:
            #print(f"Skipping gradient evaluation and quiver plot for {len(parameter_sets)} parameter sets (too many points).")

        return
    

    else:  # mode == "mle"
        # --- MLE OPTIMIZATION ---
        print("Starting MLE optimization...")
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