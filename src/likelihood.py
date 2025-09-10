# likelihood.py

import jax
import jax.numpy as jnp
import numpy as np

# Define likelihood functions
def log_likelihood_1(params, data, noise, model_fn, key):
    out = model_fn(params, key)
    output_field = out[2]   
    return compute_likelihood_1(output_field, data, noise)

@jax.jit
def compute_likelihood_1(output_field, data, noise):
    residuals = output_field - data  # Shape: (..., n_fields)
    chi2_per_field = jnp.sum(residuals ** 2, axis=(0, 1, 2)) / (noise ** 2)  
    n_points_per_field = residuals.shape[0] * residuals.shape[1] * residuals.shape[2]
    norm_per_field = n_points_per_field * jnp.log(noise) + 0.5 * n_points_per_field * jnp.log(2 * jnp.pi)
    log_likelihood = jnp.sum(-0.5 * chi2_per_field - norm_per_field)
    return log_likelihood

# Define priors
@jax.jit
def gaussian_logpdf(x, mu, sigma):
    return -0.5 * jnp.log(2 * jnp.pi * sigma**2) - 0.5 * ((x - mu) / sigma)**2

@jax.jit
def blob_gaussian_prior(params, prior_params_array):
    mu = prior_params_array[:, ::2]  # Every even index
    sigma = prior_params_array[:, 1::2]  # Every odd index
    log_priors = gaussian_logpdf(params, mu, sigma)
    return jnp.sum(log_priors)

@jax.jit
def log_uniform(x, low, high):
    return jnp.where((x >= low) & (x <= high), -jnp.log(high - low), -jnp.inf)

@jax.jit
def blob_uniform_prior(params, prior_params_array):
    low_bounds = prior_params_array[:, ::2]  # Every even index
    high_bounds = prior_params_array[:, 1::2]  # Every odd index
    log_priors = log_uniform(params, low_bounds, high_bounds)
    return jnp.sum(log_priors)

def log_prior(params, prior_type, prior_params_array):
    if prior_type == "blob_gaussian":
        log_p = blob_gaussian_prior(params, prior_params_array)
    elif prior_type == "blob_uniform":
        log_p = blob_uniform_prior(params, prior_params_array)
    else:
        raise ValueError(f"Unknown prior_type: {prior_type}")
    return log_p

# Posterior 
def log_posterior(params, log_likelihood_fn, log_prior_fn):  
    return log_likelihood_fn(params) + log_prior_fn(params)


