# mle.py

import jax
import jax.numpy as jnp
import optax
import numpy as np
import time
from functools import partial

def run_mle_optimization(log_posterior, initial_position, optimizer_name="adam", learning_rate=0.001, 
                        num_iterations=1000, print_every=100, convergence_threshold=1e-6):
    print(f"Starting MLE optimization with {optimizer_name}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max iterations: {num_iterations}")
    print(f"Initial position: {initial_position}")
    
    # Choose optimizer
    if optimizer_name == "adam":
        optimizer = optax.adam(learning_rate)
    elif optimizer_name == "sgd":
        optimizer = optax.sgd(learning_rate)
    elif optimizer_name == "rmsprop":
        optimizer = optax.rmsprop(learning_rate)
    elif optimizer_name == "adamw":
        optimizer = optax.adamw(learning_rate)
    elif optimizer_name == "adagrad":
        optimizer = optax.adagrad(learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Define objective function (we want to maximize log_posterior, so minimize negative)
    def objective(params):
        return -log_posterior(params)
    
    # Get gradient function
    grad_fn = jax.jit(jax.grad(objective))
    
    # Initialize optimizer state
    opt_state = optimizer.init(initial_position)
    params = initial_position
    
    # History tracking
    history = {
        'loss': [],
        'params': [],
        'gradients': [],
        'grad_norm': []
    }
    
    print("\nStarting optimization...")
    start_time = time.time()
    
    for iteration in range(num_iterations):
        # Compute loss and gradients
        print(params)
        loss = objective(params)
        print(loss)
        grads = grad_fn(params)
        print(grads)
        grad_norm = jnp.linalg.norm(grads)
        print(grad_norm)
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        print(params)
        
        # Store history
        history['loss'].append(float(loss))
        history['params'].append(params.copy())
        history['gradients'].append(grads.copy())
        history['grad_norm'].append(float(grad_norm))
        
        # Print progress
        if iteration % print_every == 0 or iteration == num_iterations - 1:
            log_prob = -loss
            print(f"Iteration {iteration:4d}: Loss = {loss:.6f}, Log-prob = {log_prob:.6f}, "
                  f"Grad norm = {grad_norm:.6f}")
        
        # Check convergence
        if grad_norm < convergence_threshold:
            print(f"\nConverged at iteration {iteration} (grad norm: {grad_norm:.2e})")
            break
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nOptimization completed in {duration:.2f} seconds")
    print(f"Final parameters: {params}")
    print(f"Final log-probability: {-history['loss'][-1]:.6f}")
    
    return params, history

def run_multi_start_mle(log_posterior, initial_positions, optimizer_name="adam", 
                       learning_rate=0.001, num_iterations=1000, print_every=100):
    """
    Run MLE optimization with multiple starting points to find global optimum.
    
    Parameters:
    -----------
    log_posterior : function
        Log posterior function to maximize
    initial_positions : list of arrays
        List of initial parameter values for different starts
    optimizer_name : str
        Name of optimizer
    learning_rate : float
        Learning rate for optimization
    num_iterations : int
        Maximum number of optimization iterations
    print_every : int
        Print progress every N iterations
        
    Returns:
    --------
    best_params : array
        Best optimized parameter values across all starts
    best_history : dict
        History of best optimization run
    all_results : list
        Results from all optimization runs
    """
    
    print(f"Running multi-start MLE with {len(initial_positions)} starting points")
    
    all_results = []
    best_log_prob = -jnp.inf
    best_params = None
    best_history = None
    
    for i, init_pos in enumerate(initial_positions):
        print(f"\n--- Starting point {i+1}/{len(initial_positions)} ---")
        
        try:
            params, history = run_mle_optimization(
                log_posterior, init_pos, optimizer_name, learning_rate, 
                num_iterations, print_every
            )
            
            final_log_prob = -history['loss'][-1]
            all_results.append({
                'params': params,
                'history': history,
                'final_log_prob': final_log_prob,
                'initial_position': init_pos
            })
            
            if final_log_prob > best_log_prob:
                best_log_prob = final_log_prob
                best_params = params
                best_history = history
                print(f"New best result found! Log-prob: {best_log_prob:.6f}")
            
        except Exception as e:
            print(f"Optimization failed for starting point {i+1}: {e}")
            continue
    
    if best_params is None:
        raise RuntimeError("All optimization runs failed!")
    
    print(f"\n=== BEST RESULT ===")
    print(f"Best log-probability: {best_log_prob:.6f}")
    print(f"Best parameters: {best_params}")
    
    return best_params, best_history, all_results

def create_random_initial_positions(reference_position, num_starts=5, perturbation_scale=0.1, 
                                  seed=42):
    """
    Create multiple random initial positions around a reference position.
    
    Parameters:
    -----------
    reference_position : array
        Reference parameter values
    num_starts : int
        Number of random starting positions to generate
    perturbation_scale : float
        Scale of random perturbations
    seed : int
        Random seed
        
    Returns:
    --------
    initial_positions : list
        List of initial positions
    """
    
    key = jax.random.PRNGKey(seed)
    positions = []
    
    for i in range(num_starts):
        key, subkey = jax.random.split(key)
        perturbation = jax.random.normal(subkey, shape=reference_position.shape) * perturbation_scale
        perturbed_position = reference_position + perturbation
        positions.append(perturbed_position)
    
    return positions