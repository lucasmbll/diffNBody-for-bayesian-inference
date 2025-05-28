# likelihood.py

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from model import gaussian_model 
import numpy as np

# Define likelihood functions
def log_likelihood_1(parameters, data, **kwargs): 
    """-||out-data||^2"""
    out = gaussian_model(parameters)
    output_field = out[3]  
    residuals = output_field - data
    return - jnp.sum(residuals ** 2)


def log_likelihood_2(parameters, data, noise=0.1, **kwargs):
    """Gaussian likelihood with known noise."""
    out = gaussian_model(parameters)
    output_field = out[3]  
    residuals = output_field - data
    chi2 = jnp.sum(residuals ** 2) / noise ** 2
    norm = data.size * jnp.log(noise)
    return -0.5 * chi2 - norm

def log_likelihood_3(parameters, data, variance=1.0, n_realizations=10, keys=None, **kwargs):
    """Averaged likelihood over several model realizations (Monte Carlo)."""
    total_ll = 0
    n_points = jnp.size(data)
    log_norm = -0.5 * n_points * np.log(2 * np.pi * variance)
    for i in range(n_realizations):
        key = jax.random.PRNGKey(i) if keys is None else keys[i]
        model_output = gaussian_model(parameters, key=key)
        model_output = model_output[3]  # Assuming output_field is at index 
        sq_errors = jnp.sum((model_output - data) ** 2)
        realization_ll = -0.5 * sq_errors / variance + log_norm
        total_ll += realization_ll
    return total_ll / n_realizations

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
def get_log_posterior(likelihood_type, data, prior_params=None, prior_type="gaussian", **likelihood_kwargs):
    """
    Returns a function log_posterior(params_dict) that you can plug into BlackJax.
    prior_type: string key for prior, e.g. "gaussian", "uniform"
    """
    if prior_params is None:
        # Defaults for gaussian prior
        prior_params = dict(
            sigma=dict(mu=10.0, sigma=0.5),
            mean=dict(mu=30.0, sigma=0.5),
            vel_sigma=dict(mu=1.0, sigma=0.1),
        )

    # Choose likelihood
    if likelihood_type == "ll1":
        likelihood_fn = lambda params: log_likelihood_1(params, data, **likelihood_kwargs)
    elif likelihood_type == "ll2":
        likelihood_fn = lambda params: log_likelihood_2(params, data, **likelihood_kwargs)
    elif likelihood_type == "ll3":
        likelihood_fn = lambda params: log_likelihood_3(params, data, **likelihood_kwargs)
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
