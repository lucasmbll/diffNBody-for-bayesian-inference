import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0" # Set memory fraction to 100% for JAX

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import yaml
import argparse
from pathlib import Path

from model import model
from likelihood import get_log_posterior

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def generate_mock_data(config):
    """Generate mock data using the configuration."""
    model_params = config['model_params']
    
    # Extract parameters for model
    blobs_params = model_params['blobs_params']
    G = model_params['G']
    length = model_params['length']
    softening = model_params['softening']
    t_f = model_params['t_f']
    dt = model_params['dt']
    
    # Generate mock data
    key = jax.random.PRNGKey(model_params.get('data_seed', 42))
    
    # Extract additional model parameters
    solver = model_params.get('solver', 'LeapfrogMidpoint')
    density_scaling = model_params.get('density_scaling', 'none')
    scaling_kwargs = model_params.get('scaling_kwargs', {})
    
    mock_data = model(
        blobs_params=blobs_params,
        G=G,
        length=length,
        softening=softening,
        t_f=t_f,
        dt=dt,
        key=key,
        solver=solver,
        density_scaling=density_scaling,
        **scaling_kwargs
    )
    
    return mock_data

def create_parameter_grid(config):
    """Create a discretized hypercube in parameter space."""
    hypercube_params = config['hypercube_params']
    n_points = hypercube_params['n_points_per_dim']
    param_bounds = hypercube_params['param_bounds']
    prior_params = config['prior_params']
    
    # Filter param_bounds to only include parameters that have priors
    filtered_param_bounds = {}
    for param_name, bounds in param_bounds.items():
        if param_name in prior_params:
            filtered_param_bounds[param_name] = bounds
        else:
            print(f"Warning: Parameter '{param_name}' has bounds but no prior - excluding from grid")
    
    if not filtered_param_bounds:
        raise ValueError("No parameters found with both bounds and priors defined!")
    
    print(f"Creating parameter grid for: {list(filtered_param_bounds.keys())}")
    
    # Create grid points for each parameter
    param_grids = {}
    for param_name, bounds in filtered_param_bounds.items():
        if 'center' in param_name.lower():
            # For center parameters, use scalar bounds (all coordinates are the same)
            min_val = bounds['min'] if isinstance(bounds['min'], (int, float)) else bounds['min'][0]
            max_val = bounds['max'] if isinstance(bounds['max'], (int, float)) else bounds['max'][0]
            param_grids[param_name] = np.linspace(min_val, max_val, n_points)
        else:  # Scalar parameter
            param_grids[param_name] = np.linspace(bounds['min'], bounds['max'], n_points)
    
    # Generate all combinations
    param_names = list(param_grids.keys())
    param_values = list(param_grids.values())
    param_combinations = list(product(*param_values))
    
    # Convert to parameter dictionaries
    parameter_sets = []
    for combo in param_combinations:
        param_dict = {}
        for i, param_name in enumerate(param_names):
            if 'center' in param_name.lower():
                # For center parameters, replicate scalar to 3D vector
                scalar_val = combo[i]
                param_dict[param_name] = jnp.array([scalar_val, scalar_val, scalar_val])
            else:
                # Regular scalar parameter
                param_dict[param_name] = combo[i]
        
        parameter_sets.append(param_dict)
    
    return parameter_sets, param_names

def evaluate_likelihood_and_gradients(parameter_sets, log_posterior_fn, config):
    """Evaluate likelihood and gradients for all parameter sets using mini-batch computation."""
    print(f"Evaluating likelihood and gradients for {len(parameter_sets)} parameter sets using mini-batch computation...")
    
    # Convert parameter sets to a format suitable for batching
    # We need to stack arrays for each parameter
    param_names = list(parameter_sets[0].keys())
    batched_params = {}
    
    for param_name in param_names:
        param_values = []
        for param_set in parameter_sets:
            param_values.append(param_set[param_name])
        
        # Stack the parameter values
        if isinstance(param_values[0], jnp.ndarray):
            batched_params[param_name] = jnp.stack(param_values)
        else:
            batched_params[param_name] = jnp.array(param_values)
    
    # Create a function that takes batched parameters and returns individual parameter dicts
    def unbatch_params(batched_params, idx):
        """Extract parameter dict for a specific batch index."""
        param_dict = {}
        for param_name in param_names:
            param_dict[param_name] = batched_params[param_name][idx]
        return param_dict
    
    # Create value_and_grad function
    value_and_grad_fn = jax.value_and_grad(log_posterior_fn)
    
    # Vectorize the evaluation function over the batch dimension
    def evaluate_single(idx):
        """Evaluate likelihood and gradient for a single parameter set."""
        params = unbatch_params(batched_params, idx)
        log_post_val, grad_val = value_and_grad_fn(params)
        return log_post_val, grad_val
    
    # Get mini batch size from config
    mini_batch_size = config['hypercube_params'].get('mini_batch_size', 50)
    batch_size = min(mini_batch_size, len(parameter_sets))
    print(f"Using mini-batch size: {batch_size}")
    
    log_posterior_values = []
    gradient_values = []
    
    # Track NaN statistics
    nan_count = 0
    inf_count = 0
    total_evaluations = 0
    failed_batches = 0
    
    # Process in mini-batches
    for start_idx in range(0, len(parameter_sets), batch_size):
        end_idx = min(start_idx + batch_size, len(parameter_sets))
        batch_indices_subset = jnp.arange(start_idx, end_idx)
        
        print(f"Processing batch {start_idx//batch_size + 1}/{(len(parameter_sets) + batch_size - 1)//batch_size}")
        
        try:
            # Try mini-batch evaluation
            vmapped_evaluate = jax.vmap(evaluate_single)
            batch_log_vals, batch_grad_vals = vmapped_evaluate(batch_indices_subset)
            
            # Process results and check for NaNs
            for i in range(len(batch_indices_subset)):
                total_evaluations += 1
                log_val = float(batch_log_vals[i])
                grad_dict = {k: np.array(v[i]) for k, v in batch_grad_vals.items()}
                
                # Check for NaNs and Infs
                log_is_nan = np.isnan(log_val)
                log_is_inf = np.isinf(log_val)
                grad_has_nan = any(np.any(np.isnan(grad)) for grad in grad_dict.values())
                grad_has_inf = any(np.any(np.isinf(grad)) for grad in grad_dict.values())
                
                if log_is_nan or grad_has_nan:
                    nan_count += 1
                    param_idx = start_idx + i
                    print(f"  NaN detected at parameter set {param_idx}:")
                    print(f"    Parameters: {parameter_sets[param_idx]}")
                    print(f"    Log posterior NaN: {log_is_nan}")
                    print(f"    Gradient NaN: {grad_has_nan}")
                    if grad_has_nan:
                        for k, v in grad_dict.items():
                            if np.any(np.isnan(v)):
                                print(f"      {k} gradient has NaN: {v}")
                
                if log_is_inf or grad_has_inf:
                    inf_count += 1
                    param_idx = start_idx + i
                    print(f"  Inf detected at parameter set {param_idx}:")
                    print(f"    Parameters: {parameter_sets[param_idx]}")
                    print(f"    Log posterior Inf: {log_is_inf} (value: {log_val})")
                    print(f"    Gradient Inf: {grad_has_inf}")
                    if grad_has_inf:
                        for k, v in grad_dict.items():
                            if np.any(np.isinf(v)):
                                print(f"      {k} gradient has Inf: {v}")
                
                log_posterior_values.append(log_val)
                gradient_values.append(grad_dict)
                
        except Exception as batch_e:
            failed_batches += 1
            print(f"  Batch failed: {batch_e}, falling back to individual evaluation...")
            # Fall back to individual evaluation for this batch
            for global_idx in range(start_idx, end_idx):
                total_evaluations += 1
                try:
                    params = parameter_sets[global_idx]
                    log_post_val, grad_val = value_and_grad_fn(params)
                    
                    # Check individual evaluation for NaNs
                    log_val = float(log_post_val)
                    log_is_nan = np.isnan(log_val)
                    log_is_inf = np.isinf(log_val)
                    grad_has_nan = any(np.any(np.isnan(grad)) for grad in grad_val.values())
                    grad_has_inf = any(np.any(np.isinf(grad)) for grad in grad_val.values())
                    
                    if log_is_nan or grad_has_nan:
                        nan_count += 1
                        print(f"  NaN detected at parameter set {global_idx} (individual eval):")
                        print(f"    Parameters: {params}")
                        print(f"    Log posterior NaN: {log_is_nan}")
                        print(f"    Gradient NaN: {grad_has_nan}")
                    
                    if log_is_inf or grad_has_inf:
                        inf_count += 1
                        print(f"  Inf detected at parameter set {global_idx} (individual eval):")
                        print(f"    Parameters: {params}")
                        print(f"    Log posterior Inf: {log_is_inf} (value: {log_val})")
                        print(f"    Gradient Inf: {grad_has_inf}")
                    
                    log_posterior_values.append(log_val)
                    gradient_values.append(grad_val)
                    
                except Exception as e:
                    nan_count += 1  # Count failed evaluations as NaN
                    print(f"    Warning: Failed to evaluate parameters at index {global_idx}: {e}")
                    print(f"    Parameters: {parameter_sets[global_idx]}")
                    log_posterior_values.append(np.nan)
                    gradient_values.append({k: np.nan for k in param_names})
    
    log_posterior_values = np.array(log_posterior_values)
    
    # Print NaN/Inf summary
    print(f"\n=== NaN/Inf DETECTION SUMMARY ===")
    print(f"Total evaluations: {total_evaluations}")
    print(f"NaN occurrences: {nan_count} ({100*nan_count/total_evaluations:.1f}%)")
    print(f"Inf occurrences: {inf_count} ({100*inf_count/total_evaluations:.1f}%)")
    print(f"Failed batches: {failed_batches}")
    print(f"Successful evaluations: {total_evaluations - nan_count - inf_count}")
    
    if nan_count > 0:
        print("WARNING: NaN values detected! This indicates numerical instability.")
        print("Consider:")
        print("  - Reducing parameter bounds")
        print("  - Increasing softening parameter")
        print("  - Checking for extreme parameter combinations")
    
    if inf_count > 0:
        print("WARNING: Infinite values detected! This indicates numerical overflow.")
        print("Consider:")
        print("  - Reducing parameter bounds")
        print("  - Checking for parameter combinations that lead to singularities")
    
    evaluation_stats = {
        'total_evaluations': total_evaluations,
        'nan_count': nan_count,
        'inf_count': inf_count,
        'failed_batches': failed_batches
    }
    
    return log_posterior_values, gradient_values, evaluation_stats

def create_1d_sigma_slices(parameter_sets, log_posterior_values, likelihood_values, grad_norms, individual_grads, param_arrays, param_names, valid_mask, output_dir):
    """Create 1D slice plots for sigma parameters, averaging over all other parameters."""
    print("Creating 1D sigma slice plots...")
    
    # Find sigma parameters
    sigma_params = [name for name in param_names if 'sigma' in name.lower()]
    
    if len(sigma_params) == 0:
        print("No sigma parameters found, skipping 1D slice plots")
        return
    
    # Create figure for sigma slices
    fig, axes = plt.subplots(len(sigma_params), 3, figsize=(18, 6 * len(sigma_params)))
    if len(sigma_params) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('1D Sigma Parameter Slices (Averaged over Other Parameters)', fontsize=16)
    
    for sigma_idx, sigma_param in enumerate(sigma_params):
        print(f"Processing {sigma_param}...")
        
        # Get unique sigma values
        sigma_values = np.unique(param_arrays[sigma_param][valid_mask])
        
        # Initialize arrays for averaged quantities
        avg_likelihood = []
        avg_posterior = []
        avg_grad_norm = []
        std_likelihood = []
        std_posterior = []
        std_grad_norm = []
        
        # For each unique sigma value, compute averages across all other parameter combinations
        for sigma_val in sigma_values:
            # Find all parameter sets with this sigma value
            sigma_mask = valid_mask & (param_arrays[sigma_param] == sigma_val)
            
            if np.sum(sigma_mask) > 0:
                # Get values for this sigma
                like_vals = likelihood_values[sigma_mask]
                post_vals = log_posterior_values[sigma_mask]
                grad_vals = grad_norms[sigma_mask]
                
                # Remove any remaining NaN values
                like_vals_clean = like_vals[~np.isnan(like_vals)]
                post_vals_clean = post_vals[~np.isnan(post_vals)]
                grad_vals_clean = grad_vals[~np.isnan(grad_vals)]
                
                # Compute averages and standard deviations
                avg_likelihood.append(np.mean(like_vals_clean) if len(like_vals_clean) > 0 else np.nan)
                avg_posterior.append(np.mean(post_vals_clean) if len(post_vals_clean) > 0 else np.nan)
                avg_grad_norm.append(np.mean(grad_vals_clean) if len(grad_vals_clean) > 0 else np.nan)
                
                std_likelihood.append(np.std(like_vals_clean) if len(like_vals_clean) > 1 else 0)
                std_posterior.append(np.std(post_vals_clean) if len(post_vals_clean) > 1 else 0)
                std_grad_norm.append(np.std(grad_vals_clean) if len(grad_vals_clean) > 1 else 0)
            else:
                # No data for this sigma value
                avg_likelihood.append(np.nan)
                avg_posterior.append(np.nan)
                avg_grad_norm.append(np.nan)
                std_likelihood.append(0)
                std_posterior.append(0)
                std_grad_norm.append(0)
        
        # Convert to numpy arrays
        avg_likelihood = np.array(avg_likelihood)
        avg_posterior = np.array(avg_posterior)
        avg_grad_norm = np.array(avg_grad_norm)
        std_likelihood = np.array(std_likelihood)
        std_posterior = np.array(std_posterior)
        std_grad_norm = np.array(std_grad_norm)
        
        # Remove NaN values for plotting
        valid_plot_mask = ~(np.isnan(avg_likelihood) | np.isnan(avg_posterior) | np.isnan(avg_grad_norm))
        
        if np.sum(valid_plot_mask) > 0:
            sigma_vals_plot = sigma_values[valid_plot_mask]
            avg_like_plot = avg_likelihood[valid_plot_mask]
            avg_post_plot = avg_posterior[valid_plot_mask]
            avg_grad_plot = avg_grad_norm[valid_plot_mask]
            std_like_plot = std_likelihood[valid_plot_mask]
            std_post_plot = std_posterior[valid_plot_mask]
            std_grad_plot = std_grad_norm[valid_plot_mask]
            
            # Plot 1: Averaged Likelihood vs Sigma
            ax = axes[sigma_idx, 0]
            ax.errorbar(sigma_vals_plot, avg_like_plot, yerr=std_like_plot, 
                       marker='o', linestyle='-', capsize=5, alpha=0.8, linewidth=2, markersize=6)
            ax.set_xlabel(f'{sigma_param}')
            ax.set_ylabel('Average Log Likelihood')
            ax.set_title(f'Average Likelihood vs {sigma_param}')
            ax.grid(True, alpha=0.3)
            
            # Add text showing number of realizations averaged
            n_realizations = [np.sum(valid_mask & (param_arrays[sigma_param] == sv)) for sv in sigma_vals_plot]
            ax.text(0.02, 0.98, f'Avg over {min(n_realizations)}-{max(n_realizations)} realizations', 
                   transform=ax.transAxes, verticalalignment='top', fontsize=10, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Plot 2: Averaged Posterior vs Sigma
            ax = axes[sigma_idx, 1]
            ax.errorbar(sigma_vals_plot, avg_post_plot, yerr=std_post_plot, 
                       marker='s', linestyle='-', capsize=5, alpha=0.8, linewidth=2, markersize=6, color='orange')
            ax.set_xlabel(f'{sigma_param}')
            ax.set_ylabel('Average Log Posterior')
            ax.set_title(f'Average Posterior vs {sigma_param}')
            ax.grid(True, alpha=0.3)
            
            # Add text showing number of realizations averaged
            ax.text(0.02, 0.98, f'Avg over {min(n_realizations)}-{max(n_realizations)} realizations', 
                   transform=ax.transAxes, verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Plot 3: Averaged Gradient Norm vs Sigma
            ax = axes[sigma_idx, 2]
            ax.errorbar(sigma_vals_plot, avg_grad_plot, yerr=std_grad_plot, 
                       marker='^', linestyle='-', capsize=5, alpha=0.8, linewidth=2, markersize=6, color='red')
            ax.set_xlabel(f'{sigma_param}')
            ax.set_ylabel('Average Gradient Norm')
            ax.set_title(f'Average Gradient Norm vs {sigma_param}')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            # Add text showing number of realizations averaged
            ax.text(0.02, 0.98, f'Avg over {min(n_realizations)}-{max(n_realizations)} realizations', 
                   transform=ax.transAxes, verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
        else:
            # No valid data for this sigma parameter
            for col in range(3):
                ax = axes[sigma_idx, col]
                ax.text(0.5, 0.5, f'No valid data for {sigma_param}', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_xlabel(f'{sigma_param}')
    
    plt.tight_layout()
    plt.savefig(output_dir / '1d_sigma_slices.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_comprehensive_analysis(parameter_sets, log_posterior_values, gradient_values, param_names, config, evaluation_stats):
    """Create comprehensive parameter analysis plots."""
    import datetime
    
    # Create timestamped output directory
    script_dir = Path(__file__).parent
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = script_dir / 'test_outputs' / 'gradient_check' / f'analysis_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating comprehensive analysis in: {output_dir}")
    
    # Extract parameter values into arrays
    param_arrays = {}
    for param_name in param_names:
        values = []
        for params in parameter_sets:
            val = params[param_name]
            if 'center' in param_name.lower():
                values.append(float(val[0]))  # Use scalar value for center
            else:
                values.append(float(val))
        param_arrays[param_name] = np.array(values)
    
    # Calculate likelihood values (posterior - prior)
    likelihood_values = []
    for i, params in enumerate(parameter_sets):
        from likelihood import PRIOR_REGISTRY
        prior_fn = PRIOR_REGISTRY.get(config['prior_type'])
        try:
            log_prior = prior_fn(params, config['prior_params'])
            log_likelihood = log_posterior_values[i] - log_prior
            likelihood_values.append(log_likelihood)
        except:
            likelihood_values.append(np.nan)
    
    likelihood_values = np.array(likelihood_values)
    
    # Calculate gradient norms and individual gradients
    grad_norms = []
    individual_grads = {param: [] for param in param_names}
    
    for i, grad_dict in enumerate(gradient_values):
        if isinstance(grad_dict, dict):
            total_grad_norm = 0.0
            has_nan = False
            for key, grad_val in grad_dict.items():
                if isinstance(grad_val, (list, jnp.ndarray, np.ndarray)):
                    if np.any(np.isnan(grad_val)):
                        has_nan = True
                        individual_grads[key].append(np.nan)
                        break
                    if 'center' in key.lower():
                        grad_norm = float(jnp.linalg.norm(grad_val))
                        individual_grads[key].append(grad_norm)
                        total_grad_norm += float(jnp.sum(jnp.array(grad_val)**2))
                    else:
                        individual_grads[key].append(float(grad_val))
                        total_grad_norm += float(grad_val**2)
                else:
                    if np.isnan(grad_val):
                        has_nan = True
                        individual_grads[key].append(np.nan)
                        break
                    individual_grads[key].append(float(grad_val))
                    total_grad_norm += float(grad_val**2)
            
            if has_nan:
                grad_norms.append(np.nan)
            else:
                grad_norms.append(np.sqrt(total_grad_norm))
        else:
            grad_norms.append(np.nan)
            for key in param_names:
                individual_grads[key].append(np.nan)
    
    grad_norms = np.array(grad_norms)
    for key in individual_grads:
        individual_grads[key] = np.array(individual_grads[key])
    
    # Create valid mask
    valid_mask = ~(np.isnan(log_posterior_values) | np.isnan(likelihood_values) | np.isnan(grad_norms))
    for param_name in param_arrays:
        valid_mask &= ~(np.isnan(param_arrays[param_name]) | np.isinf(param_arrays[param_name]))
    
    if np.sum(valid_mask) == 0:
        print("ERROR: No valid parameter evaluations found!")
        return
    
    print(f"Creating analysis with {np.sum(valid_mask)} valid evaluations...")
    
    # NEW: Create 1D sigma slice plots
    create_1d_sigma_slices(parameter_sets, log_posterior_values, likelihood_values, grad_norms, 
                          individual_grads, param_arrays, param_names, valid_mask, output_dir)
    
    # Get parameter medians for fixing values
    param_medians = {}
    for param_name in param_names:
        param_medians[param_name] = np.median(param_arrays[param_name][valid_mask])
    
    # 1. LIKELIHOOD VS PARAMETERS (Individual)
    print("Creating likelihood vs parameters plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Likelihood vs Individual Parameters', fontsize=16)
    
    for i, param_name in enumerate(param_names):
        if i < 4:
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            # Plot likelihood vs this parameter, others fixed at median
            param_vals = param_arrays[param_name][valid_mask]
            like_vals = likelihood_values[valid_mask]
            
            # Sort by parameter values for smooth curves
            sort_idx = np.argsort(param_vals)
            ax.plot(param_vals[sort_idx], like_vals[sort_idx], 'o-', alpha=0.7)
            ax.set_xlabel(param_name)
            ax.set_ylabel('Log Likelihood')
            ax.set_title(f'Likelihood vs {param_name}')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'likelihood_vs_parameters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. POSTERIOR VS PARAMETERS (Individual)
    print("Creating posterior vs parameters plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Posterior vs Individual Parameters', fontsize=16)
    
    for i, param_name in enumerate(param_names):
        if i < 4:
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            param_vals = param_arrays[param_name][valid_mask]
            post_vals = log_posterior_values[valid_mask]
            
            sort_idx = np.argsort(param_vals)
            ax.plot(param_vals[sort_idx], post_vals[sort_idx], 'o-', alpha=0.7)
            ax.set_xlabel(param_name)
            ax.set_ylabel('Log Posterior')
            ax.set_title(f'Posterior vs {param_name}')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'posterior_vs_parameters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. LIKELIHOOD AND POSTERIOR SURFACES (Parameter Pairs)
    print("Creating likelihood and posterior surfaces...")
    param_pairs = [(param_names[i], param_names[j]) for i in range(len(param_names)) 
                   for j in range(i+1, len(param_names))]
    
    for pair_idx, (param1, param2) in enumerate(param_pairs):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Surfaces: {param1} vs {param2}', fontsize=14)
        
        # Get parameter grids
        p1_vals = param_arrays[param1][valid_mask]
        p2_vals = param_arrays[param2][valid_mask]
        like_vals = likelihood_values[valid_mask]
        post_vals = log_posterior_values[valid_mask]
        
        # Create meshgrid for contour plots
        try:
            # Likelihood surface
            ax = axes[0]
            scatter1 = ax.tricontourf(p1_vals, p2_vals, like_vals, levels=20, cmap='viridis', alpha=0.8)
            ax.scatter(p1_vals, p2_vals, c=like_vals, cmap='viridis', s=10, alpha=0.6)
            ax.set_xlabel(param1)
            ax.set_ylabel(param2)
            ax.set_title('Likelihood Surface')
            plt.colorbar(scatter1, ax=ax, label='Log Likelihood')
            
            # Posterior surface
            ax = axes[1]
            scatter2 = ax.tricontourf(p1_vals, p2_vals, post_vals, levels=20, cmap='plasma', alpha=0.8)
            ax.scatter(p1_vals, p2_vals, c=post_vals, cmap='plasma', s=10, alpha=0.6)
            ax.set_xlabel(param1)
            ax.set_ylabel(param2)
            ax.set_title('Posterior Surface')
            plt.colorbar(scatter2, ax=ax, label='Log Posterior')
            
        except Exception as e:
            print(f"Warning: Could not create surface plot for {param1} vs {param2}: {e}")
            for ax in axes:
                ax.text(0.5, 0.5, f'Surface plot failed:\n{str(e)}', 
                       transform=ax.transAxes, ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'surfaces_{param1}_vs_{param2}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. GRADIENT NORM VS PARAMETERS
    print("Creating gradient norm plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Gradient Norm vs Parameters', fontsize=16)
    
    for i, param_name in enumerate(param_names):
        if i < 4:
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            param_vals = param_arrays[param_name][valid_mask]
            grad_vals = grad_norms[valid_mask]
            valid_grad_mask = ~np.isnan(grad_vals)
            
            if np.sum(valid_grad_mask) > 0:
                sort_idx = np.argsort(param_vals[valid_grad_mask])
                ax.semilogy(param_vals[valid_grad_mask][sort_idx], 
                           grad_vals[valid_grad_mask][sort_idx], 'o-', alpha=0.7)
                ax.set_xlabel(param_name)
                ax.set_ylabel('Gradient Norm (log scale)')
                ax.set_title(f'Gradient Norm vs {param_name}')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No valid gradients', transform=ax.transAxes, ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_norm_vs_parameters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. INDIVIDUAL GRADIENTS VS PARAMETERS
    print("Creating individual gradient plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Individual Gradients vs Parameters', fontsize=16)
    
    for i, param_name in enumerate(param_names):
        if i < 4:
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            param_vals = param_arrays[param_name][valid_mask]
            grad_vals = individual_grads[param_name][valid_mask]
            valid_grad_mask = ~np.isnan(grad_vals)
            
            if np.sum(valid_grad_mask) > 0:
                sort_idx = np.argsort(param_vals[valid_grad_mask])
                ax.plot(param_vals[valid_grad_mask][sort_idx], 
                       grad_vals[valid_grad_mask][sort_idx], 'o-', alpha=0.7)
                ax.set_xlabel(param_name)
                ax.set_ylabel(f'∂/∂{param_name}')
                ax.set_title(f'Gradient w.r.t. {param_name}')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No valid gradients', transform=ax.transAxes, ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'individual_gradients_vs_parameters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. GRADIENT SURFACES (Parameter Pairs)
    print("Creating gradient surfaces...")
    for pair_idx, (param1, param2) in enumerate(param_pairs):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Gradient Surfaces: {param1} vs {param2}', fontsize=14)
        
        p1_vals = param_arrays[param1][valid_mask]
        p2_vals = param_arrays[param2][valid_mask]
        grad1_vals = individual_grads[param1][valid_mask]
        grad2_vals = individual_grads[param2][valid_mask]
        
        try:
            # Gradient w.r.t. param1
            ax = axes[0]
            valid_grad1 = ~np.isnan(grad1_vals)
            if np.sum(valid_grad1) > 10:  # Need sufficient points
                scatter1 = ax.tricontourf(p1_vals[valid_grad1], p2_vals[valid_grad1], 
                                        grad1_vals[valid_grad1], levels=20, cmap='RdBu_r', alpha=0.8)
                ax.scatter(p1_vals[valid_grad1], p2_vals[valid_grad1], 
                          c=grad1_vals[valid_grad1], cmap='RdBu_r', s=10, alpha=0.6)
                plt.colorbar(scatter1, ax=ax, label=f'∂/∂{param1}')
            ax.set_xlabel(param1)
            ax.set_ylabel(param2)
            ax.set_title(f'Gradient w.r.t. {param1}')
            
            # Gradient w.r.t. param2
            ax = axes[1]
            valid_grad2 = ~np.isnan(grad2_vals)
            if np.sum(valid_grad2) > 10:
                scatter2 = ax.tricontourf(p1_vals[valid_grad2], p2_vals[valid_grad2], 
                                        grad2_vals[valid_grad2], levels=20, cmap='RdBu_r', alpha=0.8)
                ax.scatter(p1_vals[valid_grad2], p2_vals[valid_grad2], 
                          c=grad2_vals[valid_grad2], cmap='RdBu_r', s=10, alpha=0.6)
                plt.colorbar(scatter2, ax=ax, label=f'∂/∂{param2}')
            ax.set_xlabel(param1)
            ax.set_ylabel(param2)
            ax.set_title(f'Gradient w.r.t. {param2}')
            
        except Exception as e:
            print(f"Warning: Could not create gradient surface for {param1} vs {param2}: {e}")
            for ax in axes:
                ax.text(0.5, 0.5, f'Gradient surface failed:\n{str(e)}', 
                       transform=ax.transAxes, ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'gradient_surfaces_{param1}_vs_{param2}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 7. NaN/INF DETECTION VISUALIZATION
    print("Creating NaN/Inf detection plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('NaN/Inf Detection in Parameter Space', fontsize=16)
    
    # Create masks for different types of failures
    nan_in_likelihood = np.isnan(likelihood_values)
    nan_in_posterior = np.isnan(log_posterior_values)
    nan_in_gradients = np.isnan(grad_norms)
    inf_in_likelihood = np.isinf(likelihood_values)
    
    failure_types = [
        ('NaN in Likelihood', nan_in_likelihood),
        ('NaN in Posterior', nan_in_posterior),
        ('NaN in Gradients', nan_in_gradients),
        ('Inf in Likelihood', inf_in_likelihood)
    ]
    
    for i, (title, mask) in enumerate(failure_types):
        if i < 4:
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            if len(param_names) >= 2:
                param1, param2 = param_names[0], param_names[1]
                p1_vals = param_arrays[param1]
                p2_vals = param_arrays[param2]
                
                # Plot all points in gray
                ax.scatter(p1_vals, p2_vals, c='lightgray', alpha=0.3, s=20, label='Valid')
                
                # Highlight failed points in red
                if np.sum(mask) > 0:
                    ax.scatter(p1_vals[mask], p2_vals[mask], c='red', alpha=0.8, s=30, label='Failed')
                
                ax.set_xlabel(param1)
                ax.set_ylabel(param2)
                ax.set_title(f'{title}\n({np.sum(mask)}/{len(mask)} failures)')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'{title}\n{np.sum(mask)}/{len(mask)} failures', 
                       transform=ax.transAxes, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'nan_inf_detection.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. PARAMETER SENSITIVITY ANALYSIS
    print("Creating parameter sensitivity analysis...")
    if np.sum(valid_mask) > 0:
        # Find best point (highest posterior)
        best_idx = np.argmax(log_posterior_values[valid_mask])
        best_point_global = np.where(valid_mask)[0][best_idx]
        best_params = {name: param_arrays[name][best_point_global] for name in param_names}
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Parameter Sensitivity Around Best Point', fontsize=16)
        
        for i, param_name in enumerate(param_names):
            if i < 4:
                row, col = i // 2, i % 2
                ax = axes[row, col]
                
                # Get parameter range around best point
                param_range = param_arrays[param_name][valid_mask]
                param_std = np.std(param_range)
                param_center = best_params[param_name]
                
                # Create sensitivity curve
                param_values_sens = np.linspace(param_center - 2*param_std, 
                                              param_center + 2*param_std, 20)
                
                # Find points close to this sensitivity line
                tolerance = param_std / 5
                close_mask = valid_mask & (np.abs(param_arrays[param_name] - param_center) < 2*param_std)
                
                if np.sum(close_mask) > 3:
                    p_vals = param_arrays[param_name][close_mask]
                    like_vals = likelihood_values[close_mask]
                    post_vals = log_posterior_values[close_mask]
                    
                    # Sort for smooth plotting
                    sort_idx = np.argsort(p_vals)
                    
                    ax.plot(p_vals[sort_idx], like_vals[sort_idx], 'o-', 
                           label='Likelihood', alpha=0.7)
                    ax.plot(p_vals[sort_idx], post_vals[sort_idx], 's-', 
                           label='Posterior', alpha=0.7)
                    
                    # Mark best point
                    ax.axvline(param_center, color='red', linestyle='--', alpha=0.7, label='Best')
                    
                    ax.set_xlabel(param_name)
                    ax.set_ylabel('Log Value')
                    ax.set_title(f'Sensitivity: {param_name}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'Insufficient data\nfor sensitivity', 
                           transform=ax.transAxes, ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'parameter_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 9. SAVE COMPREHENSIVE SUMMARY
    print("Saving comprehensive summary...")
    summary_file = output_dir / 'comprehensive_analysis_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("=== COMPREHENSIVE PARAMETER ANALYSIS SUMMARY ===\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output directory: {output_dir}\n\n")
        
        # Data quality
        f.write("=== DATA QUALITY ===\n")
        f.write(f"Total parameter sets: {len(parameter_sets)}\n")
        f.write(f"Valid evaluations: {np.sum(valid_mask)}\n")
        f.write(f"NaN in likelihood: {np.sum(np.isnan(likelihood_values))}\n")
        f.write(f"NaN in posterior: {np.sum(np.isnan(log_posterior_values))}\n")
        f.write(f"NaN in gradients: {np.sum(np.isnan(grad_norms))}\n")
        f.write(f"Inf in likelihood: {np.sum(np.isinf(likelihood_values))}\n")
        f.write(f"Inf in posterior: {np.sum(np.isinf(log_posterior_values))}\n\n")
        
        # Sigma parameter analysis
        sigma_params = [name for name in param_names if 'sigma' in name.lower()]
        if sigma_params:
            f.write("=== SIGMA PARAMETER ANALYSIS ===\n")
            for sigma_param in sigma_params:
                sigma_values = np.unique(param_arrays[sigma_param][valid_mask])
                f.write(f"{sigma_param}:\n")
                f.write(f"  Values tested: {sigma_values}\n")
                f.write(f"  Range: [{np.min(sigma_values):.3f}, {np.max(sigma_values):.3f}]\n")
                
                # Average metrics for each sigma value
                for sigma_val in sigma_values:
                    sigma_mask = valid_mask & (param_arrays[sigma_param] == sigma_val)
                    n_realizations = np.sum(sigma_mask)
                    
                    if n_realizations > 0:
                        avg_like = np.mean(likelihood_values[sigma_mask][~np.isnan(likelihood_values[sigma_mask])])
                        avg_post = np.mean(log_posterior_values[sigma_mask][~np.isnan(log_posterior_values[sigma_mask])])
                        avg_grad = np.mean(grad_norms[sigma_mask][~np.isnan(grad_norms[sigma_mask])])
                        
                        f.write(f"  σ={sigma_val:.2f}: {n_realizations} realizations, "
                               f"avg_like={avg_like:.3f}, avg_post={avg_post:.3f}, avg_grad={avg_grad:.3e}\n")
            f.write("\n")
        
        # Parameter statistics
        if np.sum(valid_mask) > 0:
            f.write("=== PARAMETER STATISTICS ===\n")
            
            # Best point
            best_idx = np.argmax(log_posterior_values[valid_mask])
            best_point_global = np.where(valid_mask)[0][best_idx]
            f.write("=== BEST PARAMETER POINT ===\n")
            f.write(f"Best posterior value: {log_posterior_values[best_point_global]:.6f}\n")
            f.write(f"Best likelihood value: {likelihood_values[best_point_global]:.6f}\n")
            f.write("Best parameters:\n")
            for param_name in param_names:
                f.write(f"  {param_name}: {param_arrays[param_name][best_point_global]:.6f}\n")
            f.write("\n")
            
            # Gradient statistics
            f.write("=== GRADIENT STATISTICS ===\n")
            f.write(f"Gradient norm range: [{np.min(grad_norms[valid_mask]):.6e}, {np.max(grad_norms[valid_mask]):.6e}]\n")
            f.write(f"Mean gradient norm: {np.mean(grad_norms[valid_mask]):.6e}\n")
            f.write(f"Median gradient norm: {np.median(grad_norms[valid_mask]):.6e}\n\n")
            
            for param_name in param_names:
                grad_vals = individual_grads[param_name][valid_mask]
                grad_vals_clean = grad_vals[~np.isnan(grad_vals)]
                if len(grad_vals_clean) > 0:
                    f.write(f"Gradient w.r.t. {param_name}:\n")
                    f.write(f"  Range: [{np.min(grad_vals_clean):.6e}, {np.max(grad_vals_clean):.6e}]\n")
                    f.write(f"  Mean: {np.mean(grad_vals_clean):.6e}\n")
                    f.write(f"  Std: {np.std(grad_vals_clean):.6e}\n")
        
        # Files generated
        f.write("\n=== FILES GENERATED ===\n")
        f.write("- 1d_sigma_slices.png (NEW: Averaged 1D slices for sigma parameters)\n")
        f.write("- likelihood_vs_parameters.png\n")
        f.write("- posterior_vs_parameters.png\n")
        f.write("- gradient_norm_vs_parameters.png\n")
        f.write("- individual_gradients_vs_parameters.png\n")
        f.write("- nan_inf_detection.png\n")
        f.write("- parameter_sensitivity.png\n")
        for param1, param2 in [(param_names[i], param_names[j]) for i in range(len(param_names)) for j in range(i+1, len(param_names))]:
            f.write(f"- surfaces_{param1}_vs_{param2}.png\n")
            f.write(f"- gradient_surfaces_{param1}_vs_{param2}.png\n")
        f.write("- comprehensive_analysis_summary.txt\n")
    
    print(f"Comprehensive analysis completed!")
    print(f"All outputs saved to: {output_dir}")
    print(f"Summary saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Gradient sanity check for N-body parameter inference')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    cuda_num = config.get("cuda_visible_devices", None)
    if cuda_num is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_num)
        print(f"CUDA device set to: {cuda_num}")

    # Generate mock data
    print("Generating mock data...")
    mock_data = generate_mock_data(config)
    observed_data = mock_data[3]  # output_field
    
    print(f"Mock data shape: {observed_data.shape}")
    print(f"Mock data range: [{np.min(observed_data):.3f}, {np.max(observed_data):.3f}]")
    
    # Extract parameters to infer
    init_params = config['model_params']['blobs_params']
    
    # Create log posterior function
    print("Setting up log posterior function...")
    log_posterior_fn = get_log_posterior(
        likelihood_type=config['likelihood_type'],
        data=observed_data,
        prior_params=config['prior_params'],
        prior_type=config['prior_type'],
        model_fn=model,
        init_params=init_params,
        **config['likelihood_kwargs'],
        **{k: v for k, v in config['model_params'].items() if k != 'blobs_params'}
    )
    
    # Create parameter grid
    print("Creating parameter grid...")
    parameter_sets, param_names = create_parameter_grid(config)
    print(f"Created {len(parameter_sets)} parameter combinations")
    
    # Evaluate likelihood and gradients
    print("Evaluating likelihood and gradients...")
    log_posterior_values, gradient_values, evaluation_stats = evaluate_likelihood_and_gradients(parameter_sets, log_posterior_fn, config)
    
    # Create comprehensive analysis
    print("Creating comprehensive analysis...")
    create_comprehensive_analysis(parameter_sets, log_posterior_values, gradient_values, param_names, config, evaluation_stats)
    
    print("Gradient sanity check completed!")

if __name__ == "__main__":
    main()
