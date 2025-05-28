# run_experiment.py

import os
import sys
import yaml

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
from model import gaussian_model
from likelihood import get_log_posterior
from sampling import run_hmc, run_nuts
from plotting import (
    plot_density_fields_and_positions,
    plot_all_timesteps,
    plot_trajectories,
    plot_velocity_distributions,
    plot_trace_subplots,
    plot_corner_after_burnin,
)

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    progress_bar = config.get("progress_bar", True)  # Add this line

    # --- Generate data using model and data_params ---
    print("Generating mock data using gaussian_model...")
    data_params = config["data_params"]
    # Extract parameters for gaussian_model
    sigma = data_params["sigma"]
    mean = data_params["mean"]
    vel_sigma = data_params["vel_sigma"]
    n_part = data_params.get("n_part", 1000)
    G = data_params.get("G", 5.0)
    length = data_params.get("length", 64)
    softening = data_params.get("softening", 0.1)
    t_f = data_params.get("t_f", 1.0)
    dt = data_params.get("dt", 0.5)
    # Optionally: allow for random seed
    data_seed = data_params.get("data_seed", 0)
    data_key = jax.random.PRNGKey(data_seed)
    params = jnp.array([sigma, mean, vel_sigma])
    # Run model to generate mock data (only the output field is needed)
    input_field, init_pos, final_pos, output_field, sol = gaussian_model(
        params,
        n_part=n_part,
        G=G,
        length=length,
        softening=softening,
        t_f=t_f,
        dt=dt,
        key=data_key
    )
    print("Mock data generated.")

    data = output_field  # This is the mock data we will use for inference

    # --- Prepare output directory for plots ---
    output_path = config["output_path"]
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # --- Save plots for generated mock data ---
    # 1. Density fields and positions
    fig = plot_density_fields_and_positions(
        G, t_f, dt, length, n_part, input_field, init_pos, final_pos, output_field
    )
    fig.savefig(os.path.join(output_dir, "mock_density_fields_and_positions.png"))
    # 2. All timesteps
    fig, _ = plot_all_timesteps(sol, length, G, t_f, dt, n_part, skip=1)
    fig.savefig(os.path.join(output_dir, "mock_all_timesteps.png"))
    # 3. Trajectories
    fig = plot_trajectories(sol, G, t_f, dt, length, n_part)
    fig.savefig(os.path.join(output_dir, "mock_trajectories.png"))
    # 4. Velocity distributions
    fig, _ = plot_velocity_distributions(sol, G, t_f, dt, length, n_part)
    fig.savefig(os.path.join(output_dir, "mock_velocity_distributions.png"))

    # 2. Construire la fonction de log-posterior adaptée
    prior_type = config.get("prior_type", "gaussian")
    log_posterior = get_log_posterior(
        config["likelihood_type"], 
        data, 
        prior_params=config.get("prior_params", None),
        prior_type=prior_type,
        # Pass model parameters to likelihood functions
        n_part=n_part,
        G=G,
        length=length,
        softening=softening,
        t_f=t_f,
        dt=dt,
        **config.get("likelihood_kwargs", {})
    )

    # 3. Initialisation des paramètres
    initial_position = config["initial_position"]
    rng_key = jax.random.PRNGKey(config.get("random_seed", 12345))
    num_samples = config.get("num_samples", 5000)
    
    # 4. Lancer le sampler
    print("Starting sampling...")
    if config["sampler"] == "hmc":
        inv_mass_matrix = np.array(config["inv_mass_matrix"])
        step_size = config.get("step_size", 1e-3)
        num_integration_steps = config.get("num_integration_steps", 50)
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
    # Convert samples to dict if needed
    samples_dict = samples if isinstance(samples, dict) else {
        "sigma": samples[:, 0],
        "mean": samples[:, 1],
        "vel_sigma": samples[:, 2],
    }
    # Trace plots
    fig, _ = plot_trace_subplots(
        samples_dict,
        theta={"pos_std": sigma, "pos_mean": mean, "vel_std": vel_sigma},
        G=G, t_f=t_f, dt=dt, softening=softening, length=length, n_part=n_part,
        method=config.get("sampler", "sampler").upper()
    )
    fig.savefig(os.path.join(output_dir, "trace_sampling.png"))
    # Corner plot
    fig = plot_corner_after_burnin(
        samples_dict,
        theta={"pos_std": sigma, "pos_mean": mean, "vel_std": vel_sigma},
        burnin=config.get("num_warmup", 1000)
    )
    fig.savefig(os.path.join(output_dir, "corner_sampling.png"))

    # 5. Sauvegarde des résultats
    np.savez(config["output_path"], **{k: np.array(v) for k, v in samples.items()})
    print(f"Results saved to {config['output_path']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCMC sampling for N-body initial distribution parameters inference")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    set_cuda_device_from_config(args.config)
    main(args.config)
