# likelihood.py

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from model import model
import numpy as np

# Define likelihood functions
def log_likelihood_1(parameters, data, noise, model_fn, init_params=None, **kwargs):
    """
    -1/2σ² ||out-data||^2 - n/2 log(2πσ²)
    
    Parameters:
    -----------
    parameters : dict or array
        Parameters to infer (either a dict for new format or an array for legacy)
    data : array
        Observed data (density field)
    noise : float
        Noise level
    model_fn : function
        Model function to use
    model_type : str
        Type of model ("blobs", "gaussian", or "gaussian_2blobs")
    init_params : list of dict, optional
        Initial parameters for blobs (only used for "blobs" model_type)
    **kwargs : dict
        Additional parameters for the model
    """
   
    if init_params is None:
        raise ValueError("init_params must be provided for 'blobs' model_type")
        
    # Deep copy the init_params and update with parameters to infer
    updated_params = []
    
    for blob_idx, blob in enumerate(init_params):
        updated_blob = dict(blob)  # Shallow copy of the blob dict
        
        # Update position parameters if needed
        if updated_blob['pos_type'] == 'gaussian':
            updated_pos_params = dict(updated_blob['pos_params'])
            if f"blob{blob_idx}_sigma" in parameters:
                updated_pos_params['sigma'] = parameters[f"blob{blob_idx}_sigma"]
            if f"blob{blob_idx}_center" in parameters:
                updated_pos_params['center'] = parameters[f"blob{blob_idx}_center"]
            updated_blob['pos_params'] = updated_pos_params
        elif updated_blob['pos_type'] == 'nfw':
            updated_pos_params = dict(updated_blob['pos_params'])
            if f"blob{blob_idx}_rs" in parameters:
                updated_pos_params['rs'] = parameters[f"blob{blob_idx}_rs"]
            if f"blob{blob_idx}_c" in parameters:
                updated_pos_params['c'] = parameters[f"blob{blob_idx}_c"]
            if f"blob{blob_idx}_center" in parameters:
                updated_pos_params['center'] = parameters[f"blob{blob_idx}_center"]
            updated_blob['pos_params'] = updated_pos_params
        
        # Update velocity parameters if needed
        updated_vel_params = dict(updated_blob['vel_params'])
        if updated_blob['vel_type'] == 'cold' and f"blob{blob_idx}_vel_dispersion" in parameters:
            updated_vel_params['vel_dispersion'] = parameters[f"blob{blob_idx}_vel_dispersion"]
        elif updated_blob['vel_type'] == 'virial' and f"blob{blob_idx}_virial_ratio" in parameters:
            updated_vel_params['virial_ratio'] = parameters[f"blob{blob_idx}_virial_ratio"]
        elif updated_blob['vel_type'] == 'circular' and f"blob{blob_idx}_vel_factor" in parameters:
            updated_vel_params['vel_factor'] = parameters[f"blob{blob_idx}_vel_factor"]
        updated_blob['vel_params'] = updated_vel_params
        
        updated_params.append(updated_blob)
        
    # Run the model with updated parameters
    out = model_fn(updated_params, **kwargs)
    output_field = out[3]  # Extract output_field
        
    
    # Calculate likelihood
    residuals = output_field - data
    n_points = data.size
    chi2 = jnp.sum(residuals ** 2) / noise**2
    norm = n_points * jnp.log(noise) + 0.5 * n_points * jnp.log(2 * np.pi)
    return -0.5 * chi2 - norm

def log_likelihood_2(parameters, data, noise, n_realizations, model_fn, model_type="blobs", init_params=None, **kwargs):
    """
    Averaged likelihood over several model realizations (Monte Carlo, batched over keys).
    noise = σ
    
    Parameters:
    -----------
    parameters : dict or array
        Parameters to infer (either a dict for new format or an array for legacy)
    data : array
        Observed data (density field)
    noise : float
        Noise level
    n_realizations : int
        Number of model realizations to average over
    model_fn : function
        Model function to use
    model_type : str
        Type of model ("blobs", "gaussian", or "gaussian_2blobs")
    init_params : list of dict, optional
        Initial parameters for blobs (only used for "blobs" model_type)
    **kwargs : dict
        Additional parameters for the model
    """
    base_key = kwargs.pop("key", None)
    if base_key is None:
        base_key = jax.random.PRNGKey(0)
    keys = jax.random.split(base_key, n_realizations)
    
    def single_ll(key):
        kwargs_with_key = dict(kwargs, key=key)
        
        if model_type == "blobs":
            # New format: parameters is a dict of parameters to update in init_params
            if init_params is None:
                raise ValueError("init_params must be provided for 'blobs' model_type")
            
            # Deep copy the init_params and update with parameters to infer
            updated_params = []
            
            for blob_idx, blob in enumerate(init_params):
                updated_blob = dict(blob)  # Shallow copy of the blob dict
                
                # Update position parameters if needed
                if updated_blob['pos_type'] == 'gaussian':
                    updated_pos_params = dict(updated_blob['pos_params'])
                    if f"blob{blob_idx}_sigma" in parameters:
                        updated_pos_params['sigma'] = parameters[f"blob{blob_idx}_sigma"]
                    if f"blob{blob_idx}_center" in parameters:
                        updated_pos_params['center'] = parameters[f"blob{blob_idx}_center"]
                    updated_blob['pos_params'] = updated_pos_params
                elif updated_blob['pos_type'] == 'nfw':
                    updated_pos_params = dict(updated_blob['pos_params'])
                    if f"blob{blob_idx}_rs" in parameters:
                        updated_pos_params['rs'] = parameters[f"blob{blob_idx}_rs"]
                    if f"blob{blob_idx}_c" in parameters:
                        updated_pos_params['c'] = parameters[f"blob{blob_idx}_c"]
                    if f"blob{blob_idx}_center" in parameters:
                        updated_pos_params['center'] = parameters[f"blob{blob_idx}_center"]
                    updated_blob['pos_params'] = updated_pos_params
                
                # Update velocity parameters if needed
                updated_vel_params = dict(updated_blob['vel_params'])
                if updated_blob['vel_type'] == 'cold' and f"blob{blob_idx}_vel_dispersion" in parameters:
                    updated_vel_params['vel_dispersion'] = parameters[f"blob{blob_idx}_vel_dispersion"]
                elif updated_blob['vel_type'] == 'virial' and f"blob{blob_idx}_virial_ratio" in parameters:
                    updated_vel_params['virial_ratio'] = parameters[f"blob{blob_idx}_virial_ratio"]
                elif updated_blob['vel_type'] == 'circular' and f"blob{blob_idx}_vel_factor" in parameters:
                    updated_vel_params['vel_factor'] = parameters[f"blob{blob_idx}_vel_factor"]
                updated_blob['vel_params'] = updated_vel_params
                
                updated_params.append(updated_blob)
            
            # Run the model with updated parameters
            model_output = model_fn(updated_params, **kwargs_with_key)
            model_output = model_output[3]  # Extract output_field
            
        else:
            # Legacy format: parameters is an array
            model_output = model_fn(parameters, **kwargs_with_key)
            model_output = model_output[3]  # output_field is the 4th output
        
        # Calculate likelihood
        sq_errors = jnp.sum((model_output - data) ** 2)
        n_points = jnp.size(data)
        log_norm = 0.5 * n_points * jnp.log(2 * np.pi * noise**2)
        return -0.5 * sq_errors / (noise**2) - log_norm
    
    batched_ll = jax.vmap(single_ll)(keys)
    return jnp.mean(batched_ll)


# Define priors
def blob_gaussian_prior(params_dict, prior_params):
    """
    Gaussian prior for blob parameters.
    
    Parameters:
    -----------
    params_dict : dict
        Dictionary of parameters to infer
    prior_params : dict
        Dictionary of prior parameters
        
    Returns:
    --------
    log_prior : float
        Log prior probability
    """
    log_prior = 0.0
    
    for param_name, param_value in params_dict.items():
        if param_name in prior_params:
            prior_mean = prior_params[param_name]["mu"]
            prior_std = prior_params[param_name]["sigma"]
            
            # Handle vector parameters (like center)
            if isinstance(param_value, (list, tuple, jnp.ndarray)) and isinstance(prior_mean, (list, tuple, jnp.ndarray)):
                param_value = jnp.array(param_value)
                prior_mean = jnp.array(prior_mean)
                
                # If a single sigma is provided for a vector parameter, use it for all dimensions
                if isinstance(prior_std, (int, float)):
                    log_prior += jnp.sum(stats.norm.logpdf(param_value, prior_mean, prior_std))
                else:
                    # Otherwise, use element-wise sigma values
                    prior_std = jnp.array(prior_std)
                    log_prior += jnp.sum(stats.norm.logpdf(param_value, prior_mean, prior_std))
            else:
                # Scalar parameter
                log_prior += stats.norm.logpdf(param_value, prior_mean, prior_std)
    
    return log_prior

def blob_uniform_prior(params_dict, prior_params):
    """
    Uniform prior for blob parameters.
    
    Parameters:
    -----------
    params_dict : dict
        Dictionary of parameters to infer
    prior_params : dict
        Dictionary of prior parameters
        
    Returns:
    --------
    log_prior : float
        Log prior probability
    """
    log_prior = 0.0
    
    def log_uniform(x, low, high):
        return jnp.where((x >= low) & (x <= high), -jnp.log(high - low), -jnp.inf)
    
    for param_name, param_value in params_dict.items():
        if param_name in prior_params:
            prior_low = prior_params[param_name]["low"]
            prior_high = prior_params[param_name]["high"]
            
            # Handle vector parameters (like center)
            if isinstance(param_value, (list, tuple, jnp.ndarray)) and isinstance(prior_low, (list, tuple, jnp.ndarray)):
                param_value = jnp.array(param_value)
                prior_low = jnp.array(prior_low)
                prior_high = jnp.array(prior_high)
                
                # Element-wise uniform prior for vectors
                log_prior += jnp.sum(log_uniform(param_value, prior_low, prior_high))
            else:
                # Scalar parameter
                log_prior += log_uniform(param_value, prior_low, prior_high)
    
    return log_prior

PRIOR_REGISTRY = {
    "blob_gaussian": blob_gaussian_prior,
    "blob_uniform": blob_uniform_prior
}

# Get log posterior function
def get_log_posterior(likelihood_type, data, prior_params=None, prior_type="gaussian", model_fn=None, init_params=None, **likelihood_kwargs):
    """
    Returns a function log_posterior(params_dict) that you can plug into BlackJAX.
    
    Parameters:
    -----------
    likelihood_type : str
        Type of likelihood function to use ('ll1' or 'll2')
    data : array
        Observed data (density field)
    prior_params : dict
        Dictionary of prior parameters
    prior_type : str
        Type of prior to use ('gaussian', 'uniform', 'blob_gaussian', 'blob_uniform')
    model_fn : function
        Model function to use
    model_type : str
        Type of model ('blobs', 'gaussian', 'gaussian_2blobs')
    init_params : list of dict, optional
        Initial parameters for blobs (only used for 'blobs' model_type)
    auto_generate_priors : bool
        Whether to automatically generate prior parameters from init_params
    **likelihood_kwargs : dict
        Additional parameters for the likelihood function
    
    Returns:
    --------
    log_posterior : function
        Function that calculates the log posterior probability
    """
    if model_fn is None:
        raise ValueError("model_fn must be provided.")
    
    # Auto-generate prior parameters if requeste
    if prior_params is None:
        raise ValueError("prior_params must be provided if auto_generate_priors is False.")
    
    # Extract noise parameter from likelihood_kwargs - this will be used for all likelihood types
    noise = likelihood_kwargs.get("noise", 1)  # Default noise value if not specified
    
    # Filter out noise and other special parameters from kwargs passed to model
    model_kwargs = {k: v for k, v in likelihood_kwargs.items() if k not in ['noise', 'n_realizations']}
    
    # Choose likelihood
    if likelihood_type == "ll1":
        likelihood_fn = lambda params: log_likelihood_1(
            params, data, noise, model_fn=model_fn, init_params=init_params, **model_kwargs)
    elif likelihood_type == "ll2":
        n_realizations = likelihood_kwargs.get("n_realizations", 10)
        likelihood_fn = lambda params: log_likelihood_2(
            params, data, noise, n_realizations, model_fn=model_fn, init_params=init_params, **model_kwargs)
    else:
        raise ValueError(f"Unknown likelihood_type: {likelihood_type}. Available options: 'll1', 'll2'")

    # Choose prior
    prior_fn = PRIOR_REGISTRY.get(prior_type)
    if prior_fn is None:
        raise ValueError(f"Unknown prior_type: {prior_type}")

    def log_posterior(params_dict):
        log_p = prior_fn(params_dict, prior_params)        
        log_lik = likelihood_fn(params_dict)
        return log_p + log_lik

    return log_posterior
