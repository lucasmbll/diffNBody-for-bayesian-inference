# grid.py

import jax
import jax.numpy as jnp
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import os

def create_parameter_grid(n_points_per_dim, param_bounds, params_infos):
    changing_param_order = params_infos[0]['changing_param_order']
    
    # Create grid points for each parameter
    param_grids = {}
    for param_name in changing_param_order:
        found_bounds = False
        for blob_idx in range(len(params_infos)):
            param_key = f"blob{blob_idx}_{param_name}"
            if param_key in param_bounds:
                bounds = param_bounds[param_key]
                param_grids[param_key] = np.linspace(bounds['min'], bounds['max'], n_points_per_dim)
                found_bounds = True
                break
        
        if not found_bounds:
            raise ValueError(f"No bounds found for parameter '{param_name}' in param_bounds. "
                           f"Expected at least one of: {[f'blob{i}_{param_name}' for i in range(len(params_infos))]}")

    # Generate all combinations of parameter values for a single blob
    param_names = list(param_grids.keys())
    param_values = list(param_grids.values())
    single_blob_combinations = list(product(*param_values))

    # Convert single blob combinations into full parameter sets for all blobs
    num_blobs = len(params_infos)
    parameter_sets = []
    
    for combo in single_blob_combinations:
        # Create a parameter array for all blobs
        blob_params = []
        for blob_idx in range(num_blobs):
            blob_params.append(jnp.array(combo, dtype=jnp.float32))
        parameter_sets.append(jnp.stack(blob_params))

    print(f"Created parameter grid with {len(parameter_sets)} parameter combinations")
    print(f"Parameters: {param_names}")
    print(f"Grid dimensions: {[len(values) for values in param_values]}")

    parameter_sets = jnp.array(parameter_sets, dtype=jnp.float32)
    print(f"Parameter sets shape: {parameter_sets.shape}")
    return parameter_sets

def evaluate_likelihood(parameter_sets, mini_batch_size, log_posterior_fn, log_prior_fn):
    # Vectorize the evaluation function over the batch dimension
    def evaluate_single(params):
        """Evaluate likelihood and gradient for a single parameter set."""
        log_post_val = log_posterior_fn(params)
        log_prior_val = log_prior_fn(params)
        log_likelihood_val = log_post_val - log_prior_val
        return log_post_val, log_likelihood_val
    
    # Get mini batch size from config
    n_grid_points = parameter_sets.shape[0]
    batch_size = min(mini_batch_size, n_grid_points)
    print(f"Using mini-batch size: {batch_size}")

    log_posterior_values = []
    log_likelihood_values = []
    
    # Track NaN statistics
    nan_count = 0
    inf_count = 0
    total_evaluations = 0
    failed_batches = 0
    
    # Process in mini-batches
    for start_idx in range(0, n_grid_points, batch_size):
        end_idx = min(start_idx + batch_size, n_grid_points)
        batch_params = parameter_sets[start_idx:end_idx]
        
        print(f"Processing batch {start_idx//batch_size + 1}/{(n_grid_points + batch_size - 1)//batch_size}")
        
        try:
            # Try mini-batch evaluation
            vmapped_evaluate = jax.vmap(evaluate_single)
            batch_log_post_val, batch_log_likelihood_val = vmapped_evaluate(batch_params)
            
            # Process results and check for NaNs
            for i in range(batch_params.shape[0]):
                total_evaluations += 1
                log_post_val = float(batch_log_post_val[i])
                log_lik_val = float(batch_log_likelihood_val[i])
    
                
                # Check for NaNs and Infs
                log_post_is_nan = np.isnan(log_post_val)
                log_post_is_inf = np.isinf(log_post_val)
                log_lik_is_nan = np.isnan(log_lik_val)
                log_lik_is_inf = np.isinf(log_lik_val)
            
                if log_post_is_nan or log_lik_is_nan:
                    nan_count += 1
                    param_idx = start_idx + i
                    print(f"  NaN detected at parameter set {param_idx}:")
                    print(f"    Parameters: {parameter_sets[param_idx]}")
                    print(f"    Log posterior NaN: {log_post_is_nan}")
                    print(f"    Log likelihood NaN: {log_lik_is_nan}")
                    
                if log_post_is_inf or log_lik_is_inf:
                    inf_count += 1
                    param_idx = start_idx + i
                    print(f"  Inf detected at parameter set {param_idx}:")
                    print(f"    Parameters: {parameter_sets[param_idx]}")
                    print(f"    Log posterior Inf: {log_post_is_inf} (value: {log_post_val})")
                    print(f"    Log likelihood Inf: {log_lik_is_inf} (value: {log_lik_val})")
            
                log_posterior_values.append(log_post_val)
                log_likelihood_values.append(log_lik_val)
                
        except Exception as batch_e:
            failed_batches += 1
            raise ValueError(f"Batch evaluation failed: {batch_e}.")
            
    log_posterior_values = np.array(log_posterior_values)
    log_lik_values = np.array(log_likelihood_values)
    
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
    
    # Create valid mask
    valid_mask = ~(jnp.isnan(log_posterior_values) | jnp.isnan(log_lik_values))
    
    n_valid_evaluations = jnp.sum(valid_mask)
    print(f"There are {n_valid_evaluations} valid evaluations...")
    
    if n_valid_evaluations == 0:
        raise ValueError("No valid evaluations found after filtering NaNs/Infs.")
    
    evaluation_stats = {
        'total_evaluations': total_evaluations,
        'valid_evaluations': n_valid_evaluations,
        'nan_count': nan_count,
        'inf_count': inf_count,
        'failed_batches': failed_batches
    }    
    return log_posterior_values, log_lik_values, evaluation_stats, valid_mask

def value_surface(parameter_sets, log_posterior_values, log_lik_values, param_names, valid_mask, output_dir):
    n_params_per_blob = parameter_sets.shape[2]
    # Extract parameter values into arrays
    param_arrays = {}
    for idx, param_name in enumerate(param_names):
        blob_idx = idx // n_params_per_blob
        param_idx = idx % n_params_per_blob
        values = parameter_sets[:, blob_idx, param_idx]
        param_arrays[param_name] = values
    

    param_pairs = [(param_names[i], param_names[j]) for i in range(len(param_names)) 
                   for j in range(i+1, len(param_names))]
    
    for pair_idx, (param1, param2) in enumerate(param_pairs):
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'Surfaces: {param1} vs {param2}', fontsize=16)
        
        # Get parameter grids
        p1_vals = param_arrays[param1][valid_mask]
        p2_vals = param_arrays[param2][valid_mask]
        like_vals = log_lik_values[valid_mask]
        post_vals = log_posterior_values[valid_mask]
        
        try:
            # Simple averaging with numpy
            coords = np.column_stack([p1_vals, p2_vals])
            unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
            
            like_avg = np.array([like_vals[inverse == i].mean() for i in range(len(unique_coords))])
            post_avg = np.array([post_vals[inverse == i].mean() for i in range(len(unique_coords))])
            
            p1_avg = unique_coords[:, 0]
            p2_avg = unique_coords[:, 1]
            
            # First row: 2D surfaces
            # Likelihood surface 2D
            ax1 = plt.subplot(2, 2, 1)
            scatter1 = ax1.tricontourf(p1_avg, p2_avg, like_avg, levels=20, cmap='plasma', alpha=0.8)
            ax1.scatter(p1_avg, p2_avg, c=like_avg, cmap='viridis', s=10, alpha=0.6)
            ax1.set_xlabel(param1)
            ax1.set_ylabel(param2)
            ax1.set_title('2D Likelihood Surface (Averaged)')
            plt.colorbar(scatter1, ax=ax1, label='Log Likelihood')
            
            # Posterior surface 2D
            ax2 = plt.subplot(2, 2, 2)
            scatter2 = ax2.tricontourf(p1_avg, p2_avg, post_avg, levels=20, cmap='plasma', alpha=0.8)
            ax2.scatter(p1_avg, p2_avg, c=post_avg, cmap='plasma', s=10, alpha=0.6)
            ax2.set_xlabel(param1)
            ax2.set_ylabel(param2)
            ax2.set_title('2D Posterior Surface (Averaged)')
            plt.colorbar(scatter2, ax=ax2, label='Log Posterior')
            
            # Second row: 3D surfaces
            # Likelihood surface 3D
            ax3 = plt.subplot(2, 2, 3, projection='3d')
            ax3.plot_trisurf(p1_avg, p2_avg, like_avg, cmap='viridis', alpha=0.8)
            ax3.scatter(p1_avg, p2_avg, like_avg, c=like_avg, cmap='viridis', s=20, alpha=0.6)
            ax3.set_xlabel(param1)
            ax3.set_ylabel(param2)
            ax3.set_zlabel('Log Likelihood')
            ax3.set_title('3D Likelihood Surface (Averaged)')
            
            # Posterior surface 3D
            ax4 = plt.subplot(2, 2, 4, projection='3d')
            ax4.plot_trisurf(p1_avg, p2_avg, post_avg, cmap='plasma', alpha=0.8)
            ax4.scatter(p1_avg, p2_avg, post_avg, c=post_avg, cmap='plasma', s=20, alpha=0.6)
            ax4.set_xlabel(param1)
            ax4.set_ylabel(param2)
            ax4.set_zlabel('Log Posterior')
            ax4.set_title('3D Posterior Surface (Averaged)')
            
        except Exception as e:
            print(f"Warning: Could not create surface plot for {param1} vs {param2}: {e}")
            # Create error messages for all subplots
            ax1 = plt.subplot(2, 2, 1)
            ax2 = plt.subplot(2, 2, 2)
            ax3 = plt.subplot(2, 2, 3, projection='3d')
            ax4 = plt.subplot(2, 2, 4, projection='3d')
            
            for ax in [ax1, ax2]:
                ax.text(0.5, 0.5, f'Surface plot failed:\n{str(e)}', 
                       transform=ax.transAxes, ha='center', va='center')
            for ax in [ax3, ax4]:
                ax.text(0.5, 0.5, 0.5, f'3D surface failed:\n{str(e)}', 
                       transform=ax.transAxes, ha='center', va='center')
        
        fig_name = f'surfaces_{param1}_vs_{param2}.png'
        dir = os.path.join(output_dir, fig_name)
        plt.tight_layout()
        plt.savefig(dir, dpi=300, bbox_inches='tight')
        plt.close()


def values_slices(parameter_sets, log_posterior_values, log_lik_values, param_names, valid_mask, output_dir):
    n_params_per_blob = parameter_sets.shape[2]
    # Extract parameter values into arrays
    param_arrays = {}
    for idx, param_name in enumerate(param_names):
        blob_idx = idx // n_params_per_blob
        param_idx = idx % n_params_per_blob
        values = parameter_sets[:, blob_idx, param_idx]
        param_arrays[param_name] = values

    # Create slices for each parameter
    for param_idx, param_name in enumerate(param_names):
        print(f"Creating slice for parameter: {param_name}")
        
        # Get unique values for this parameter
        param_values = np.unique(param_arrays[param_name][valid_mask])
        
        # Initialize arrays for averaged quantities
        avg_likelihood = []
        avg_posterior = []
        std_likelihood = []
        std_posterior = []
        
        # For each unique parameter value, compute averages across all other parameter combinations
        for param_val in param_values:
            # Find all parameter sets with this parameter value
            param_mask = valid_mask & (param_arrays[param_name] == param_val)
            
            if np.sum(param_mask) > 0:
                # Get values for this parameter value
                like_vals = log_lik_values[param_mask]
                post_vals = log_posterior_values[param_mask]
                
                # Remove any remaining NaN values
                like_vals_clean = like_vals[~np.isnan(like_vals)]
                post_vals_clean = post_vals[~np.isnan(post_vals)]
                
                # Compute averages and standard deviations
                avg_likelihood.append(np.mean(like_vals_clean) if len(like_vals_clean) > 0 else np.nan)
                avg_posterior.append(np.mean(post_vals_clean) if len(post_vals_clean) > 0 else np.nan)
                
                std_likelihood.append(np.std(like_vals_clean) if len(like_vals_clean) > 1 else 0)
                std_posterior.append(np.std(post_vals_clean) if len(post_vals_clean) > 1 else 0)
            else:
                # No data for this parameter value
                avg_likelihood.append(np.nan)
                avg_posterior.append(np.nan)
                std_likelihood.append(0)
                std_posterior.append(0)
        
        # Convert to numpy arrays
        avg_likelihood = np.array(avg_likelihood)
        avg_posterior = np.array(avg_posterior)
        std_likelihood = np.array(std_likelihood)
        std_posterior = np.array(std_posterior)
        
        # Remove NaN values for plotting
        valid_plot_mask = ~(np.isnan(avg_likelihood) | np.isnan(avg_posterior))
        
        if np.sum(valid_plot_mask) > 0:
            param_vals_plot = param_values[valid_plot_mask]
            avg_like_plot = avg_likelihood[valid_plot_mask]
            avg_post_plot = avg_posterior[valid_plot_mask]
            std_like_plot = std_likelihood[valid_plot_mask]
            std_post_plot = std_posterior[valid_plot_mask]
            
            # Create subplot for this parameter
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'Parameter Slice: {param_name} (Averaged over Other Parameters)', fontsize=14)
            
            # Plot 1: Averaged Likelihood vs Parameter
            ax = axes[0]
            ax.errorbar(param_vals_plot, avg_like_plot, yerr=std_like_plot, 
                       marker='o', linestyle='-', capsize=5, alpha=0.8, linewidth=2, markersize=6)
            ax.set_xlabel(param_name)
            ax.set_ylabel('Average Log Likelihood')
            ax.set_title(f'Average Likelihood vs {param_name}')
            ax.grid(True, alpha=0.3)
            
            # Add text showing number of realizations averaged
            n_realizations = [np.sum(valid_mask & (param_arrays[param_name] == pv)) for pv in param_vals_plot]
            ax.text(0.02, 0.98, f'Avg over {min(n_realizations)}-{max(n_realizations)} realizations', 
                   transform=ax.transAxes, verticalalignment='top', fontsize=10, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Plot 2: Averaged Posterior vs Parameter
            ax = axes[1]
            ax.errorbar(param_vals_plot, avg_post_plot, yerr=std_post_plot, 
                       marker='s', linestyle='-', capsize=5, alpha=0.8, linewidth=2, markersize=6, color='orange')
            ax.set_xlabel(param_name)
            ax.set_ylabel('Average Log Posterior')
            ax.set_title(f'Average Posterior vs {param_name}')
            ax.grid(True, alpha=0.3)
            
            # Add text showing number of realizations averaged
            ax.text(0.02, 0.98, f'Avg over {min(n_realizations)}-{max(n_realizations)} realizations', 
                   transform=ax.transAxes, verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            fig_name = f'slice_{param_name}.png'
            dir = os.path.join(output_dir, fig_name)
            plt.tight_layout()
            plt.savefig(dir, dpi=300, bbox_inches='tight')
            plt.close()
            
        else:
            print(f"No valid data for parameter {param_name}")

def evaluate_gradients(parameter_sets, mini_batch_size, log_posterior_fn, log_prior_fn):
    # Create gradient functions
    posterior_grad_fn = jax.grad(log_posterior_fn)
    prior_grad_fn = jax.grad(log_prior_fn)
    
    # Vectorize the evaluation function over the batch dimension
    def evaluate_single_grad(params):
        """Evaluate gradients for a single parameter set."""
        log_post_grad_val = posterior_grad_fn(params)
        log_prior_grad_val = prior_grad_fn(params)
        log_likelihood_grad_val = log_post_grad_val - log_prior_grad_val
        return log_post_grad_val, log_likelihood_grad_val
    
    # Get mini batch size from config
    n_grid_points = parameter_sets.shape[0]
    batch_size = min(mini_batch_size, n_grid_points)
    print(f"Using mini-batch size: {batch_size}")

    posterior_gradient_values = []
    likelihood_gradient_values = []
    
    # Track NaN statistics
    nan_count = 0
    inf_count = 0
    total_evaluations = 0
    failed_batches = 0
    
    # Process in mini-batches
    for start_idx in range(0, n_grid_points, batch_size):
        end_idx = min(start_idx + batch_size, n_grid_points)
        batch_params = parameter_sets[start_idx:end_idx]
        
        print(f"Processing batch {start_idx//batch_size + 1}/{(n_grid_points + batch_size - 1)//batch_size}")
        
        try:
            # Try mini-batch evaluation
            vmapped_evaluate = jax.vmap(evaluate_single_grad)
            batch_log_post_grad_val, batch_log_likelihood_grad_val = vmapped_evaluate(batch_params)
            
            # Process results and check for NaNs
            for i in range(batch_params.shape[0]):
                total_evaluations += 1
                post_grad_val = batch_log_post_grad_val[i]
                lik_grad_val = batch_log_likelihood_grad_val[i]
                
                # Check for NaNs and Infs
                post_grad_has_nan = np.any(np.isnan(post_grad_val))
                post_grad_has_inf = np.any(np.isinf(post_grad_val))
                lik_grad_has_nan = np.any(np.isnan(lik_grad_val))
                lik_grad_has_inf = np.any(np.isinf(lik_grad_val))
                
                if post_grad_has_nan or lik_grad_has_nan:
                    nan_count += 1
                    param_idx = start_idx + i
                    print(f"  NaN detected at parameter set {param_idx}:")
                    print(f"    Parameters: {parameter_sets[param_idx]}")
                    print(f"    Posterior gradient NaN: {post_grad_has_nan}")
                    print(f"    Likelihood gradient NaN: {lik_grad_has_nan}")
                    
                if post_grad_has_inf or lik_grad_has_inf:
                    inf_count += 1
                    param_idx = start_idx + i
                    print(f"  Inf detected at parameter set {param_idx}:")
                    print(f"    Parameters: {parameter_sets[param_idx]}")
                    print(f"    Posterior gradient Inf: {post_grad_has_inf}")
                    print(f"    Likelihood gradient Inf: {lik_grad_has_inf}")
                
                posterior_gradient_values.append(post_grad_val)
                likelihood_gradient_values.append(lik_grad_val)
                
        except Exception as batch_e:
            failed_batches += 1
            raise ValueError(f"Batch evaluation failed: {batch_e}.")
            
    posterior_gradient_values = np.array(posterior_gradient_values)
    likelihood_gradient_values = np.array(likelihood_gradient_values)
    
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
    
    # Create valid mask
    valid_mask = ~(np.any(np.isnan(posterior_gradient_values), axis=(1,2)) | 
                   np.any(np.isnan(likelihood_gradient_values), axis=(1,2)))
    
    n_valid_evaluations = np.sum(valid_mask)
    print(f"There are {n_valid_evaluations} valid evaluations...")
    
    if n_valid_evaluations == 0:
        raise ValueError("No valid evaluations found after filtering NaNs/Infs.")
    
    evaluation_stats = {
        'total_evaluations': total_evaluations,
        'valid_evaluations': n_valid_evaluations,
        'nan_count': nan_count,
        'inf_count': inf_count,
        'failed_batches': failed_batches
    }    
    
    return posterior_gradient_values, likelihood_gradient_values, evaluation_stats, valid_mask

def quiver_grad_surface(parameter_sets, log_lik_values, posterior_gradient_values, likelihood_gradient_values, param_names, valid_mask, output_dir):
    n_params_per_blob = parameter_sets.shape[2]
    # Extract parameter values into arrays
    param_arrays = {}
    for idx, param_name in enumerate(param_names):
        blob_idx = idx // n_params_per_blob
        param_idx = idx % n_params_per_blob
        values = parameter_sets[:, blob_idx, param_idx]
        param_arrays[param_name] = values
    

    param_pairs = [(param_names[i], param_names[j]) for i in range(len(param_names)) 
                   for j in range(i+1, len(param_names))]
    
    # Extract gradient components for each parameter
    grad_arrays = {}
    for idx, param_name in enumerate(param_names):
        blob_idx = idx // n_params_per_blob
        param_idx = idx % n_params_per_blob
        # Extract gradient for this parameter from all parameter sets
        grad_values = []
        for i in range(len(parameter_sets)):
            if valid_mask[i]:
                grad_val = likelihood_gradient_values[i, blob_idx, param_idx]
                grad_values.append(grad_val)
            else:
                grad_values.append(np.nan)
        grad_arrays[param_name] = np.array(grad_values)

    for pair_idx, (param1, param2) in enumerate(param_pairs):
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        fig.suptitle(f'Gradient Quiver Plot: {param1} vs {param2}', fontsize=16)
        
        # Get parameter grids
        p1_vals = param_arrays[param1][valid_mask]
        p2_vals = param_arrays[param2][valid_mask]
        like_vals = log_lik_values[valid_mask]
        
        # Get gradients for this parameter pair
        grad1_vals = grad_arrays[param1][valid_mask]
        grad2_vals = grad_arrays[param2][valid_mask]
        
        try:
            # Create background contour plot of likelihood
            scatter = ax.tricontourf(p1_vals, p2_vals, like_vals, levels=20, cmap='plasma', alpha=0.6)
            plt.colorbar(scatter, ax=ax, label='Log Likelihood')
            
            # Create quiver plot of gradients
            # Subsample points for cleaner visualization
            n_points = len(p1_vals)
            skip = max(1, n_points // 50)  # Show at most 50 arrows
            
            # Filter out NaN gradients
            valid_grad_mask = ~(np.isnan(grad1_vals) | np.isnan(grad2_vals))
            
            if np.sum(valid_grad_mask) > 0:
                p1_quiver = p1_vals[valid_grad_mask][::skip]
                p2_quiver = p2_vals[valid_grad_mask][::skip]
                grad1_quiver = grad1_vals[valid_grad_mask][::skip]
                grad2_quiver = grad2_vals[valid_grad_mask][::skip]
                
                # Normalize gradients for better visualization
                grad_magnitude = np.sqrt(grad1_quiver**2 + grad2_quiver**2)
                max_grad = np.max(grad_magnitude)
                
                if max_grad > 0:
                    scale_factor = 0.1 * (np.max(p1_vals) - np.min(p1_vals)) / max_grad
                    
                    quiver = ax.quiver(p1_quiver, p2_quiver, 
                                        grad1_quiver, grad2_quiver,
                                        grad_magnitude,
                                        cmap='RdBu_r', alpha=0.8, 
                                        scale=1/scale_factor, scale_units='xy',
                                        angles='xy', width=0.003)
                    
                    # Add colorbar for gradient magnitude
                    cbar = plt.colorbar(quiver, ax=ax, label='Gradient Magnitude', 
                                        orientation='horizontal', pad=0.1, shrink=0.8)
                else:
                    ax.text(0.5, 0.5, 'All gradients are zero', 
                            transform=ax.transAxes, ha='center', va='center')
            else:
                ax.text(0.5, 0.5, 'No valid gradients available', 
                        transform=ax.transAxes, ha='center', va='center')
            
            ax.set_xlabel(param1)
            ax.set_ylabel(param2)
            ax.set_title(f'Likelihood Surface with Gradient Field')
            
        except Exception as e:
            print(f"Warning: Could not create quiver plot for {param1} vs {param2}: {e}")
            ax.text(0.5, 0.5, f'Quiver plot failed:\n{str(e)}', 
                    transform=ax.transAxes, ha='center', va='center')
        
        fig_name = f'gradient_quiver_{param1}_vs_{param2}.png'
        dir = os.path.join(output_dir, fig_name)
        plt.tight_layout()
        plt.savefig(dir, dpi=300, bbox_inches='tight')
        plt.close()




"""  
def create_comprehensive_analysis(parameter_sets, log_posterior_values, gradient_values, param_names, config, evaluation_stats):
    
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
    
    print("Gradient sanity check completed!")"""