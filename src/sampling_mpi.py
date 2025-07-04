import numpy as np
import os
import time
from typing import Dict, List, Union, Optional, Tuple
import jax
import jax.numpy as jnp


def create_initial_positions(initial_position: Union[Dict, List[Dict]], 
                           num_chains: int, 
                           rank: int, 
                           size: int,
                           rng_key: jax.random.PRNGKey) -> Tuple[List[Dict], int, int]:
    """Create initial position for this MPI process"""
    # Calculate which chains this process will handle
    chains_per_process = num_chains // size
    remainder = num_chains % size
    
    if rank < remainder:
        start_chain = rank * (chains_per_process + 1)
        num_local_chains = chains_per_process + 1
    else:
        start_chain = rank * chains_per_process + remainder
        num_local_chains = chains_per_process
    
    if isinstance(initial_position, list):
        # List of positions provided - use specific ones for this process
        if len(initial_position) == num_chains:
            # Use assigned positions for this process
            local_positions = initial_position[start_chain:start_chain + num_local_chains]
        else:
            # Repeat/cycle through available positions
            local_positions = []
            for i in range(num_local_chains):
                pos_idx = (start_chain + i) % len(initial_position)
                local_positions.append(initial_position[pos_idx])
    else:
        # Single position dict - create variations with noise
        local_positions = []
        for i in range(num_local_chains):
            chain_key = jax.random.fold_in(rng_key, start_chain + i)
            pos_with_noise = {}
            
            for param_name, param_value in initial_position.items():
                param_key = jax.random.fold_in(chain_key, hash(param_name) % 2**31)
                
                if isinstance(param_value, (list, tuple, jnp.ndarray)):
                    noise = jax.random.normal(param_key, shape=jnp.array(param_value).shape) * 0.1
                    pos_with_noise[param_name] = jnp.array(param_value) + noise
                else:
                    noise = jax.random.normal(param_key) * abs(param_value) * 0.1
                    pos_with_noise[param_name] = param_value + noise
            
            local_positions.append(pos_with_noise)
    
    return local_positions, start_chain, num_local_chains

def calculate_rhat_parallel(samples_dict: Dict, comm) -> Dict:
    """Calculate R-hat across all MPI processes"""
    rank = comm.Get_rank()
    
    # Gather all samples at root
    all_samples = comm.gather(samples_dict, root=0)
    
    if rank == 0:
        # Calculate R-hat for each parameter
        rhat_values = {}
        param_names = list(all_samples[0].keys())
        
        for param_name in param_names:
            # Collect samples from all processes
            param_samples = []
            for process_samples in all_samples:
                if isinstance(process_samples[param_name], list):
                    # Each element in the list is samples from one chain
                    for chain_samples in process_samples[param_name]:
                        param_samples.append(chain_samples)
                else:
                    param_samples.append(process_samples[param_name])
            
            # Stack into array: (n_total_chains, n_samples, ...)
            if len(param_samples) > 1:
                # Convert to arrays first if needed
                param_samples = [jnp.array(ps) for ps in param_samples]
                chains_array = jnp.stack(param_samples, axis=0)
                rhat_values[param_name] = calculate_rhat_scalar(chains_array)
            else:
                rhat_values[param_name] = None
        
        return rhat_values
    else:
        return None

def calculate_rhat_scalar(chains):
    """Calculate R-hat for chains array of shape (n_chains, n_samples, ...)"""
    if len(chains.shape) < 2:
        return None
    
    n_chains, n_samples = chains.shape[:2]
    
    if n_chains < 2:
        return None
    
    # Handle vector parameters
    if len(chains.shape) > 2:
        # Calculate R-hat for each component
        original_shape = chains.shape[2:]
        chains_flat = chains.reshape(n_chains, n_samples, -1)
        
        rhat_components = []
        for i in range(chains_flat.shape[2]):
            component_chains = chains_flat[:, :, i]
            rhat_components.append(_calculate_rhat_scalar(component_chains))
        
        return jnp.array(rhat_components).reshape(original_shape)
    else:
        # Scalar parameter
        return _calculate_rhat_scalar(chains)

def _calculate_rhat_scalar(chains):
    """Calculate R-hat for a single scalar parameter across chains"""
    n_chains, n_samples = chains.shape
    
    # Calculate between-chain and within-chain variances
    chain_means = jnp.mean(chains, axis=1)
    overall_mean = jnp.mean(chain_means)
    
    B = n_samples * jnp.var(chain_means, ddof=1)
    W = jnp.mean(jnp.var(chains, axis=1, ddof=1))
    
    # Calculate R-hat
    var_hat = ((n_samples - 1) / n_samples) * W + B / n_samples
    rhat = jnp.sqrt(var_hat / W)
    
    return rhat

def run_mpi_sampling(sampler_type: str, 
                    log_posterior, 
                    rank,
                    size,
                    comm,
                    config: Dict,
                    base_dir: str,
                    rng_key: jax.random.PRNGKey) -> Tuple[Dict, Optional[Dict]]:
    """
    Run MCMC sampling with MPI parallelization and R-hat monitoring
    
    Parameters:
    -----------
    sampler_type : str
        Type of sampler to use
    log_posterior : function
        Log posterior function
    config : dict
        Configuration dictionary
    base_dir : str
        Base directory for saving results
    rng_key : jax.random.PRNGKey
        Random key for sampling
        
    Returns:
    --------
    samples : dict
        Dictionary of samples
    rhat_values : dict or None
        R-hat values (only for root process)
    """
    
    # Extract configuration
    num_chains = config.get("num_chains", size)
    num_samples = config.get("num_samples", 1000)
    num_warmup = config.get("num_warmup", 1000)
    initial_position = config.get("initial_position", {})
    rhat_check_interval = config.get("rhat_check_interval", 100)
    
    # Create initial positions for this process
    process_key = jax.random.fold_in(rng_key, rank)
    local_positions, start_chain, num_local_chains = create_initial_positions(
        initial_position, num_chains, rank, size, process_key
    )
    
    if rank == 0:
        print(f"Running {sampler_type} with MPI:")
        print(f"  Total chains: {num_chains}")
        print(f"  MPI processes: {size}")
        print(f"  Samples per chain: {num_samples}")
        print(f"  Warmup per chain: {num_warmup}")
        print(f"  R-hat check interval: {rhat_check_interval}")
    
    print(f"Process {rank}: Handling {num_local_chains} chains (indices {start_chain}-{start_chain+num_local_chains-1})")
    
    # Import sampler functions
    from sampling import run_nuts, run_hmc, run_rwm, run_mala
    
    # Run chains for this process
    local_samples = {}
    
    for i, init_pos in enumerate(local_positions):
        chain_key = jax.random.fold_in(process_key, i)
        chain_id = start_chain + i
        
        print(f"Process {rank}: Starting chain {chain_id} with initial position: {init_pos}")
        
        # Run the appropriate sampler
        if sampler_type == "nuts":
            samples = run_nuts(log_posterior, init_pos, chain_key, num_samples, num_warmup)
        elif sampler_type == "hmc":
            inv_mass_matrix = np.array(config["inv_mass_matrix"])
            step_size = config.get("step_size", 1e-3)
            num_integration_steps = config.get("num_integration_steps", 50)
            samples = run_hmc(log_posterior, init_pos, inv_mass_matrix, 
                            step_size, num_integration_steps, chain_key, num_samples, num_warmup)
        elif sampler_type == "rwm":
            step_size = config.get("step_size", 0.1)
            samples = run_rwm(log_posterior, init_pos, step_size, chain_key, num_samples)
        elif sampler_type == "mala":
            step_size = config.get("step_size", 0.01)
            autotuning = config.get("autotuning", False)
            samples = run_mala(log_posterior, init_pos, step_size, chain_key, 
                             num_samples, num_warmup, autotuning)
        else:
            raise ValueError(f"Unknown sampler: {sampler_type}")
        
        # Store samples for this chain
        for param_name, param_samples in samples.items():
            if param_name not in local_samples:
                local_samples[param_name] = []
            local_samples[param_name].append(param_samples)
        
        print(f"Process {rank}: Completed chain {chain_id}")
    
    # Calculate R-hat
    if rank == 0:
        print("\n" + "="*50)
        print("CALCULATING R-HAT DIAGNOSTICS")
        print("="*50)
    
    rhat_values = calculate_rhat_parallel(local_samples, comm)
    
    if rank == 0 and rhat_values:
        print("R-hat convergence diagnostics:")
        for param_name, rhat in rhat_values.items():
            if rhat is None:
                print(f"  {param_name}: R-hat = N/A (insufficient chains)")
                continue
                
            if isinstance(rhat, jnp.ndarray):
                max_rhat = jnp.max(rhat)
                print(f"  {param_name}: max R-hat = {max_rhat:.3f}")
                if max_rhat > 1.1:
                    print(f"    ⚠️  Warning: {param_name} may not have converged (R-hat > 1.1)")
                elif max_rhat > 1.01:
                    print(f"    ✓ {param_name} shows good convergence")
                else:
                    print(f"    ✓ {param_name} shows excellent convergence")
            else:
                print(f"  {param_name}: R-hat = {rhat:.3f}")
                if rhat > 1.1:
                    print(f"    ⚠️  Warning: {param_name} may not have converged (R-hat > 1.1)")
                elif rhat > 1.01:
                    print(f"    ✓ {param_name} shows good convergence")
                else:
                    print(f"    ✓ {param_name} shows excellent convergence")
    
    # Gather all samples at root for final output
    all_samples = comm.gather(local_samples, root=0)
    
    if rank == 0:
        # Combine all samples
        combined_samples = {}
        for param_name in local_samples.keys():
            combined_samples[param_name] = []
            for process_samples in all_samples:
                combined_samples[param_name].extend(process_samples[param_name])
        
        # Save chain-separated samples
        chain_samples = {}
        for param_name, chains in combined_samples.items():
            chain_samples[param_name] = np.stack(chains, axis=0)  # (n_chains, n_samples, ...)
        
        np.savez(os.path.join(base_dir, "samples_by_chain.npz"), **chain_samples)
        print(f"Chain-separated samples saved to {os.path.join(base_dir, 'samples_by_chain.npz')}")
        
        # Save R-hat diagnostics
        if rhat_values:
            rhat_dict = {k: np.array(v) if v is not None else None for k, v in rhat_values.items()}
            np.savez(os.path.join(base_dir, "rhat_diagnostics.npz"), **rhat_dict)
            print(f"R-hat diagnostics saved to {os.path.join(base_dir, 'rhat_diagnostics.npz')}")
        
    if rank == 0:
        # Prepare final samples for return (combine all chains for corner plotting)
        final_samples = {}
        for param_name in param_names:
            all_chain_samples = []
            
            # Collect samples from all processes and chains
            for process_samples in all_samples:
                if isinstance(process_samples[param_name], list):
                    # Multiple chains from this process
                    for chain_samples in process_samples[param_name]:
                        # Convert to array and add to collection
                        all_chain_samples.append(jnp.array(chain_samples))
                else:
                    # Single chain from this process
                    all_chain_samples.append(jnp.array(process_samples[param_name]))
            
            # Stack all chains and flatten for corner plotting
            if len(all_chain_samples) > 0:
                # Stack: (n_chains, n_samples, ...)
                stacked = jnp.stack(all_chain_samples, axis=0)
                # Flatten chains: (n_chains * n_samples, ...)
                final_samples[param_name] = stacked.reshape(-1, *stacked.shape[2:])
            else:
                final_samples[param_name] = jnp.array([])
        
        return final_samples, rhat_values
    else:
        return {}, None