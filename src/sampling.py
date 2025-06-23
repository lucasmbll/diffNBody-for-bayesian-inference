# sampling.py

import jax
import jax.numpy as jnp
import blackjax
import numpy as np
import time

def inference_loop(rng_key, kernel, initial_state, num_samples, progress_bar=False):
    if not progress_bar:
        @jax.jit
        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state
        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)
        return states
    else:
        from tqdm import tqdm
        state = initial_state
        states = []
        keys = jax.random.split(rng_key, num_samples)
        for i in tqdm(range(num_samples), desc="Sampling"):
            state, _ = kernel(keys[i], state)
            states.append(state)
        # Stack states to match lax.scan output
        return jax.tree_util.tree_map(lambda *xs: np.stack(xs), *states)

def extract_params_to_infer(init_params):
    """
    Extract parameters to infer from initialization parameters.
    
    Parameters:
    -----------
    init_params : list of dict
        List of dictionaries with blob initialization parameters
        
    Returns:
    --------
    params_to_infer : dict
        Dictionary mapping parameter names to their default values
    """
    params_to_infer = {}
    
    for blob_idx, blob in enumerate(init_params):
        # Extract position parameters
        if blob['pos_type'] == 'gaussian':
            params_to_infer[f"blob{blob_idx}_sigma"] = blob['pos_params']['sigma']
            params_to_infer[f"blob{blob_idx}_center"] = blob['pos_params']['center']
        elif blob['pos_type'] == 'nfw':
            params_to_infer[f"blob{blob_idx}_rs"] = blob['pos_params']['rs']
            params_to_infer[f"blob{blob_idx}_c"] = blob['pos_params']['c']
            params_to_infer[f"blob{blob_idx}_center"] = blob['pos_params']['center']
        
        # Extract velocity parameters
        if blob['vel_type'] == 'cold':
            params_to_infer[f"blob{blob_idx}_vel_dispersion"] = blob['vel_params'].get('vel_dispersion', 1e-6)
        elif blob['vel_type'] == 'virial':
            params_to_infer[f"blob{blob_idx}_virial_ratio"] = blob['vel_params'].get('virial_ratio', 1.0)
        elif blob['vel_type'] == 'circular':
            params_to_infer[f"blob{blob_idx}_vel_factor"] = blob['vel_params'].get('vel_factor', 1.0)
    
    return params_to_infer

def run_hmc(log_posterior, initial_position, inv_mass_matrix, step_size, num_integration_steps, rng_key, num_samples, num_warmup=1000, progress_bar=False):
    print(f"HMC parameters: step_size={step_size}, num_integration_steps={num_integration_steps}")
    print(f"Initial position: {initial_position}")
    print("\n" + "STARTING SAMPLING PHASE")
    sampling_start_time = time.time()

    hmc = blackjax.hmc(log_posterior, step_size, inv_mass_matrix, num_integration_steps)
    hmc_kernel = jax.jit(hmc.step)
    initial_state = hmc.init(initial_position)
    
    if num_warmup > 0:
        states = inference_loop(rng_key, hmc_kernel, initial_state, num_samples+num_warmup, progress_bar=progress_bar)
    
    else :
        states = inference_loop(rng_key, hmc_kernel, initial_state, num_samples, progress_bar=progress_bar)

    sampling_end_time = time.time()
    sampling_duration = sampling_end_time - sampling_start_time
    print("SAMPLING PHASE COMPLETED")
    print(f"Sampling duration: {sampling_duration:.2f} seconds")
    print(f"Average time per sample: {sampling_duration/num_samples:.4f} seconds")
    print(f"Samples per second: {num_samples/sampling_duration:.2f}")
    
    return states.position

def run_nuts(log_posterior, initial_position, rng_key, num_samples, num_warmup=1000, progress_bar=False):
    print(f"Warmup steps: {num_warmup}")
    print(f"Sampling steps: {num_samples}")
    print(f"Initial position: {initial_position}")
    
    warmup = blackjax.window_adaptation(blackjax.nuts, log_posterior)
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    
    print("\n" + "STARTING WARMUP PHASE")
    warmup_start_time = time.time()
    
    (state, parameters), warmup_info = warmup.run(warmup_key, initial_position, num_steps=num_warmup)
    
    warmup_end_time = time.time()
    warmup_duration = warmup_end_time - warmup_start_time
    
    print("WARMUP PHASE COMPLETED")
    print(f"Warmup duration: {warmup_duration:.2f} seconds")
    print(f"Final step size: {parameters['step_size']:.6f}")
    print(f"Final inverse mass matrix diagonal: {jnp.diag(parameters['inverse_mass_matrix'])}")
    #print(f"Final number of leapfrog steps: {parameters['num_leapfrog_steps']}")
    
    if hasattr(warmup_info, 'acceptance_rate'):
        print(f"Warmup acceptance rate: {warmup_info.acceptance_rate:.3f}")
    
    print("\n" + "STARTING SAMPLING PHASE")
    sampling_start_time = time.time()
    
    kernel = blackjax.nuts(log_posterior, **parameters).step
    states = inference_loop(sample_key, kernel, state, num_samples, progress_bar=progress_bar)
    
    sampling_end_time = time.time()
    sampling_duration = sampling_end_time - sampling_start_time
    total_duration = warmup_duration + sampling_duration
    
    print("SAMPLING PHASE COMPLETED")
    print(f"Sampling duration: {sampling_duration:.2f} seconds")
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Average time per sample: {sampling_duration/num_samples:.4f} seconds")
    print(f"Samples per second: {num_samples/sampling_duration:.2f}")
    
    return states.position


def run_rwm(log_posterior, initial_position, step_size, rng_key, num_samples, progress_bar=False):
    """
    Run Random Walk Metropolis sampler with optional warmup phase.
    
    Parameters:
    -----------
    log_posterior : function
        Log posterior function
    initial_position : dict
        Initial parameter values
    step_size : dict or float
        Step sizes for each parameter (dict) or single step size for all
    rng_key : jax.random.PRNGKey
        Random key for sampling
    num_samples : int
        Number of samples to draw
    progress_bar : bool
        Whether to show progress bar
        
    Returns:
    --------
    samples : dict
        Dictionary of samples for each parameter
    """
    print(f"RWM parameters: step_size={step_size}")
    print(f"Initial position: {initial_position}")
    print(f"Sampling steps: {num_samples}")
        
    print("\n" + "STARTING SAMPLING PHASE")
    sampling_start_time = time.time()
    
    sample_key = rng_key
    
    # Initialize RWM with provided step size
    rwm = blackjax.additive_step_random_walk(log_posterior, blackjax.mcmc.random_walk.normal(step_size))
    rwm_kernel = jax.jit(rwm.step)
    initial_state = rwm.init(initial_position)
    
    # Run sampling
    states = inference_loop(sample_key, rwm_kernel, initial_state, num_samples, progress_bar=progress_bar)
    
    sampling_end_time = time.time()
    sampling_duration = sampling_end_time - sampling_start_time
    
    print("SAMPLING PHASE COMPLETED")
    print(f"Sampling duration: {sampling_duration:.2f} seconds")
    print(f"Average time per sample: {sampling_duration/num_samples:.4f} seconds")
    print(f"Samples per second: {num_samples/sampling_duration:.2f}")
    
    return states.position

def run_mala(log_posterior, initial_position, step_size, rng_key, num_samples, 
             num_warmup=1000, progress_bar=False, autotuning=False):
    """
    Run MALA sampler with optional warmup phase.
    
    Parameters:
    -----------
    log_posterior : function
        Log posterior function
    initial_position : dict
        Initial parameter values
    step_size : float
        Step size for MALA
    rng_key : jax.random.PRNGKey
        Random key for sampling
    num_samples : int
        Number of samples to draw
    num_warmup : int, optional
        Number of warmup steps for step size adaptation (default: 0, no warmup)
    progress_bar : bool
        Whether to show progress bar
    autotuning : bool, optional
        Whether to adapt step size during warmup (default: False)
        
    Returns:
    --------
    samples : dict
        Dictionary of samples for each parameter
    """
    print(f"MALA parameters: step_size={step_size}")
    print(f"Initial position: {initial_position}")
    print(f"Warmup steps: {num_warmup}")
    print(f"Sampling steps: {num_samples}")
    
    if autotuning:
        from utils import tune_step_size
        print("\n" + "STARTING AUTOTUNING PHASE")
        autotuning_start_time = time.time()
        
        step_size_adapted = tune_step_size(blackjax.mala, log_posterior, initial_position, rng_key, num_trials=100)

        autotuning_end_time = time.time()

        autotuning_duration = autotuning_end_time - autotuning_start_time
        
        print("AUTOTUNING PHASE COMPLETED")
        print(f"Autotuning duration: {autotuning_duration:.2f} seconds")
        print(f"Original step size: {step_size}")
        print(f"Adapted step size: {step_size_adapted}")
        
        # Use adapted parameters for sampling
        step_size = step_size_adapted
    
    sample_key = rng_key    
    # Initialize MALA with provided step size
    mala = blackjax.mala(log_posterior, step_size)
    mala_kernel = jax.jit(mala.step)
    initial_state = mala.init(initial_position)
    
    print("\n" + "STARTING SAMPLING PHASE")
    start_time = time.time()


    if num_warmup > 0:

        states = inference_loop(sample_key, mala_kernel, initial_state, num_warmup + num_samples, progress_bar=progress_bar)      
    
    else :
        states = inference_loop(sample_key, mala_kernel, initial_state, num_samples, progress_bar=progress_bar)

    end_time = time.time()
    duration = end_time - start_time
    print("SAMPLING PHASE COMPLETED")
    print(f"Sampling duration: {duration:.2f} seconds")
    return states.position