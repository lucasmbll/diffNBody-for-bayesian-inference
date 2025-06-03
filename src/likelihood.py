# likelihood.py

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from model import gaussian_model 
import numpy as np

# Define likelihood functions
def log_likelihood_1(parameters, data, noise, model_fn, **kwargs):
    """
    -1/2σ² ||out-data||^2 - n/2 log(2πσ²)
    """
    out = model_fn(parameters, **kwargs)
    output_field = out[3]
    residuals = output_field - data
    n_points = data.size
    chi2 = jnp.sum(residuals ** 2) / noise**2
    norm = n_points * jnp.log(noise) + 0.5 * n_points * jnp.log(2 * np.pi)
    return -0.5 * chi2 - norm

def log_likelihood_2(parameters, data, noise, model_fn, **kwargs):
    """
    Gaussian likelihood with known noise.
    """
    out = model_fn(parameters, **kwargs)
    output_field = out[3]
    residuals = output_field - data
    n_points = data.size
    chi2 = jnp.sum(residuals ** 2) / (noise**2)
    log_norm = 0.5 * n_points * jnp.log(2 * np.pi * noise**2)
    return -0.5 * chi2 - log_norm

def log_likelihood_3(parameters, data, noise, n_realizations, model_fn, **kwargs):
    """
    Averaged likelihood over several model realizations (Monte Carlo, batched over keys).
    noise = σ
    """
    base_key = kwargs.pop("key", None)
    if base_key is None:
        base_key = jax.random.PRNGKey(0)
    keys = jax.random.split(base_key, n_realizations)
    def single_ll(key):
        model_output = model_fn(parameters, key=key, **kwargs)
        model_output = model_output[3]  # output_field
        sq_errors = jnp.sum((model_output - data) ** 2)
        n_points = jnp.size(data)
        log_norm = 0.5 * n_points * jnp.log(2 * np.pi * noise**2)
        return -0.5 * sq_errors / (noise**2) - log_norm
    batched_ll = jax.vmap(single_ll)(keys)
    return jnp.mean(batched_ll)

# Define priors
def gaussian_prior(params_dict, prior_params):
    sigma = params_dict["sigma"]
    mean = params_dict["mean"]
    vel_sigma = params_dict["vel_sigma"]
    p_sigma = stats.norm.logpdf(sigma, prior_params["sigma"]["mu"], prior_params["sigma"]["sigma"])
    p_mean = stats.norm.logpdf(mean, prior_params["mean"]["mu"], prior_params["mean"]["sigma"])
    p_vel = stats.norm.logpdf(vel_sigma, prior_params["vel_sigma"]["mu"], prior_params["vel_sigma"]["sigma"])
    return p_sigma + p_mean + p_vel

def uniform_prior(params_dict, prior_params):
    sigma = params_dict["sigma"]
    mean = params_dict["mean"]
    vel_sigma = params_dict["vel_sigma"]
    # Each param: [low, high]
    def log_uniform(x, low, high):
        return jnp.where((x >= low) & (x <= high), -jnp.log(high - low), -jnp.inf)
    p_sigma = log_uniform(sigma, prior_params["sigma"]["low"], prior_params["sigma"]["high"])
    p_mean = log_uniform(mean, prior_params["mean"]["low"], prior_params["mean"]["high"])
    p_vel = log_uniform(vel_sigma, prior_params["vel_sigma"]["low"], prior_params["vel_sigma"]["high"])
    return p_sigma + p_mean + p_vel

PRIOR_REGISTRY = {
    "gaussian": gaussian_prior,
    "uniform": uniform_prior,
    # Add more priors here as needed
}

# Get log posterior function
def get_log_posterior(likelihood_type, data, prior_params=None, prior_type="gaussian", model_fn=None, **likelihood_kwargs):
    """
    Returns a function log_posterior(params_dict) that you can plug into BlackJax.
    prior_type: string key for prior, e.g. "gaussian", "uniform"
    """
    if prior_params is None:
        raise ValueError("prior_params must be provided.")

    if model_fn is None:
        raise ValueError("model_fn must be provided.")
    
    # Extract noise parameter from likelihood_kwargs - this will be used for all likelihood types
    noise = likelihood_kwargs.get("noise", 0.05)  # Default noise value if not specified
    
    # Filter out noise and other special parameters from kwargs passed to model
    model_kwargs = {k: v for k, v in likelihood_kwargs.items() if k not in ['noise', 'n_realizations']}
    
    # Choose likelihood
    if likelihood_type == "ll1":
        likelihood_fn = lambda params: log_likelihood_1(params, data, noise, model_fn=model_fn, **model_kwargs)
    elif likelihood_type == "ll2":
        likelihood_fn = lambda params: log_likelihood_2(params, data, noise, model_fn=model_fn, **model_kwargs)
    elif likelihood_type == "ll3":
        n_realizations = likelihood_kwargs.get("n_realizations", 10)
        likelihood_fn = lambda params: log_likelihood_3(params, data, noise, n_realizations, model_fn=model_fn, **model_kwargs)
    else:
        raise ValueError(f"Unknown likelihood_type: {likelihood_type}")

    # Choose prior
    prior_fn = PRIOR_REGISTRY.get(prior_type)
    if prior_fn is None:
        raise ValueError(f"Unknown prior_type: {prior_type}")

    def log_posterior(params_dict):
        params = jnp.array([params_dict["sigma"], params_dict["mean"], params_dict["vel_sigma"]])
        log_p = prior_fn(params_dict, prior_params)
        log_lik = likelihood_fn(params)
        return log_p + log_lik

    return log_posterior
