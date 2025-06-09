# sampling.py

import jax
import jax.numpy as jnp
import blackjax
import numpy as np

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

def run_hmc(log_posterior, initial_position, inv_mass_matrix, step_size, num_integration_steps, rng_key, num_samples, progress_bar=False):
    hmc = blackjax.hmc(log_posterior, step_size, inv_mass_matrix, num_integration_steps)
    hmc_kernel = jax.jit(hmc.step)
    initial_state = hmc.init(initial_position)
    states = inference_loop(rng_key, hmc_kernel, initial_state, num_samples, progress_bar=progress_bar)
    return states.position

def run_nuts(log_posterior, initial_position, rng_key, num_samples, num_warmup=1000, progress_bar=False):
    import time
    
    print(f"Starting NUTS sampling...")
    print(f"Warmup steps: {num_warmup}")
    print(f"Sampling steps: {num_samples}")
    print(f"Initial position: {initial_position}")
    
    warmup = blackjax.window_adaptation(blackjax.nuts, log_posterior)
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    
    print("\n" + "="*50)
    print("STARTING WARMUP PHASE")
    print("="*50)
    warmup_start_time = time.time()
    
    (state, parameters), warmup_info = warmup.run(warmup_key, initial_position, num_steps=num_warmup)
    
    warmup_end_time = time.time()
    warmup_duration = warmup_end_time - warmup_start_time
    
    print("="*50)
    print("WARMUP PHASE COMPLETED")
    print("="*50)
    print(f"Warmup duration: {warmup_duration:.2f} seconds")
    print(f"Final step size: {parameters['step_size']:.6f}")
    print(f"Final inverse mass matrix diagonal: {jnp.diag(parameters['inverse_mass_matrix'])}")
    
    if hasattr(warmup_info, 'acceptance_rate'):
        print(f"Warmup acceptance rate: {warmup_info.acceptance_rate:.3f}")
    
    print("\n" + "="*50)
    print("STARTING SAMPLING PHASE")
    print("="*50)
    sampling_start_time = time.time()
    
    kernel = blackjax.nuts(log_posterior, **parameters).step
    states = inference_loop(sample_key, kernel, state, num_samples, progress_bar=progress_bar)
    
    sampling_end_time = time.time()
    sampling_duration = sampling_end_time - sampling_start_time
    total_duration = warmup_duration + sampling_duration
    
    print("="*50)
    print("SAMPLING PHASE COMPLETED")
    print("="*50)
    print(f"Sampling duration: {sampling_duration:.2f} seconds")
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Average time per sample: {sampling_duration/num_samples:.4f} seconds")
    print(f"Samples per second: {num_samples/sampling_duration:.2f}")
    
    return states.position
