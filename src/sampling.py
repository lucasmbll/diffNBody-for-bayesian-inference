# sampling.py

import jax
import blackjax
import time

def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state
    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)
    return states

def warmup(algorithm, log_posterior, initial_position, warmup_key, num_warmup, progress_bar, **kwargs):
    warmup = blackjax.window_adaptation(algorithm, log_posterior, progress_bar=progress_bar, **kwargs)
    (state, parameters), warmup_info = warmup.run(warmup_key, initial_position, num_steps=num_warmup)
    return state, parameters, warmup_info

def run_hmc(log_posterior, initial_position, num_integration_steps, rng_key, num_samples, num_warmup, progress_bar):
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    print("\n" + "STARTING WARMUP PHASE")
    warmup_start_time = time.time()
    state, parameters, warmup_info = warmup(blackjax.hmc, log_posterior, initial_position, warmup_key, num_warmup, progress_bar, num_integration_steps=num_integration_steps)
    warmup_end_time = time.time()
    warmup_duration = warmup_end_time - warmup_start_time
    print("WARMUP PHASE COMPLETED")
    print(f"Warmup duration: {warmup_duration:.2f} seconds")
    print(f"Final HMC params: {parameters}")
    print(f"Warmup info: {warmup_info}")
    print(f"Final state: {state}")
    print("\n" + "STARTING SAMPLING PHASE")
    sampling_start_time = time.time()
    hmc = blackjax.hmc(log_posterior, **parameters)
    hmc_kernel = jax.jit(hmc.step)
    states = inference_loop(sample_key, hmc_kernel, state, num_samples)
    sampling_end_time = time.time()
    sampling_duration = sampling_end_time - sampling_start_time
    total_duration = warmup_duration + sampling_duration
    print("SAMPLING PHASE COMPLETED")
    print(f"Sampling duration: {sampling_duration:.2f} seconds")
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Average time per sample: {sampling_duration/num_samples:.4f} seconds")
    print(f"Samples per second: {num_samples/sampling_duration:.2f}")  
    return states.position

def run_nuts(log_posterior, initial_position, rng_key, num_samples, num_warmup, progress_bar):
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    print("\n" + "STARTING WARMUP PHASE")
    warmup_start_time = time.time()
    state, parameters, warmup_info = warmup(blackjax.nuts, log_posterior, initial_position, warmup_key, num_warmup, progress_bar)
    warmup_end_time = time.time()
    warmup_duration = warmup_end_time - warmup_start_time
    print("WARMUP PHASE COMPLETED")
    print(f"Warmup duration: {warmup_duration:.2f} seconds")
    print(f"Final NUTS params: {parameters}")
    print(f"Warmup info: {warmup_info}")
    print(f"Final state: {state}")
    print("\n" + "STARTING SAMPLING PHASE")
    sampling_start_time = time.time()
    kernel = jax.jit(blackjax.nuts(log_posterior, **parameters).step)
    states = inference_loop(sample_key, kernel, state, num_samples)
    sampling_end_time = time.time()
    sampling_duration = sampling_end_time - sampling_start_time
    total_duration = warmup_duration + sampling_duration
    print("SAMPLING PHASE COMPLETED")
    print(f"Sampling duration: {sampling_duration:.2f} seconds")
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Average time per sample: {sampling_duration/num_samples:.4f} seconds")
    print(f"Samples per second: {num_samples/sampling_duration:.2f}")  
    return states.position


def run_rwm(log_posterior, initial_position, step_size, rng_key, num_samples):
    print(f"RWM parameters: step_size={step_size}")
    print(f"Initial position: {initial_position}")
    print(f"Sampling steps: {num_samples}")  
    print("\n" + "STARTING SAMPLING PHASE")
    sampling_start_time = time.time()
    sample_key = rng_key
    rwm = blackjax.additive_step_random_walk(log_posterior, blackjax.mcmc.random_walk.normal(step_size))
    rwm_kernel = jax.jit(rwm.step)
    initial_state = rwm.init(initial_position)
    states = inference_loop(sample_key, rwm_kernel, initial_state, num_samples)
    sampling_end_time = time.time()
    sampling_duration = sampling_end_time - sampling_start_time
    print("SAMPLING PHASE COMPLETED")
    print(f"Sampling duration: {sampling_duration:.2f} seconds")
    print(f"Average time per sample: {sampling_duration/num_samples:.4f} seconds")
    print(f"Samples per second: {num_samples/sampling_duration:.2f}")
    return states.position

def run_mala(log_posterior, initial_position, step_size, rng_key, num_samples, 
             num_warmup=1000, autotuning=False):
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

        states = inference_loop(sample_key, mala_kernel, initial_state, num_warmup + num_samples)      
    
    else:
        states = inference_loop(sample_key, mala_kernel, initial_state, num_samples)

    end_time = time.time()
    duration = end_time - start_time
    print("SAMPLING PHASE COMPLETED")
    print(f"Sampling duration: {duration:.2f} seconds")
    return states.position


