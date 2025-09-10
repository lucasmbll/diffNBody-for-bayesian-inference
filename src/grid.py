# grid.py

import jax
import jax.numpy as jnp
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import os

def create_parameter_grid(n_points_per_dim, param_bounds, params_infos):
    changing_param_order = params_infos[0]['changing_param_order']
    num_blobs = len(params_infos)
    
    # Create separate parameter grids for each blob
    blob_param_grids = {}
    for blob_idx in range(num_blobs):
        blob_param_grids[blob_idx] = {}
        for param_name in changing_param_order:
            param_key = f"blob{blob_idx}_{param_name}"
            if param_key in param_bounds:
                bounds = param_bounds[param_key]
                blob_param_grids[blob_idx][param_name] = jnp.linspace(
                    bounds['min'], bounds['max'], n_points_per_dim
                )
            else:
                raise ValueError(f"No bounds found for parameter '{param_key}' in param_bounds")
    
    # Generate all combinations across all blobs
    all_param_names = []
    all_param_values = []
    
    for blob_idx in range(num_blobs):
        for param_name in changing_param_order:
            all_param_names.append(f"blob{blob_idx}_{param_name}")
            all_param_values.append(blob_param_grids[blob_idx][param_name])
    
    # Create all combinations
    all_combinations = list(product(*all_param_values))
    
    # Convert to parameter sets
    parameter_sets = []
    for combo in all_combinations:
        # Reshape combination into blob structure
        blob_params = []
        for blob_idx in range(num_blobs):
            start_idx = blob_idx * len(changing_param_order)
            end_idx = start_idx + len(changing_param_order)
            blob_combo = combo[start_idx:end_idx]
            blob_params.append(jnp.array(blob_combo, dtype=jnp.float32))
        parameter_sets.append(jnp.stack(blob_params))
    
    parameter_sets = jnp.array(parameter_sets, dtype=jnp.float32)
    
    print(f"Created parameter grid with {len(parameter_sets)} parameter combinations")
    print(f"Parameters per blob: {changing_param_order}")
    print(f"Number of blobs: {num_blobs}")
    print(f"Grid dimensions per blob: {[n_points_per_dim] * len(changing_param_order)}")
    print(f"Total grid dimensions: {[n_points_per_dim] * len(changing_param_order) * num_blobs}")
    print(f"Parameter sets shape: {parameter_sets.shape}")
    
    return parameter_sets

def evaluate_likelihood(parameter_sets, mini_batch_size, log_posterior_fn, log_prior_fn):
    # Vectorize the evaluation function over the batch dimension
    def evaluate_single(params):
        """Evaluate likelihood and gradient for a single parameter set."""
        log_post_val = log_posterior_fn(params)
        log_prior_val = log_prior_fn(params)
        log_likelihood_val = log_post_val - log_prior_val
        chi2_val = -2.0 * log_likelihood_val
        return log_post_val, log_likelihood_val, chi2_val
    
    # Get mini batch size from config
    n_grid_points = parameter_sets.shape[0]
    batch_size = min(mini_batch_size, n_grid_points)
    print(f"Using mini-batch size: {batch_size}")

    log_posterior_values = []
    log_likelihood_values = []
    chi2_values = []
    
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
            batch_log_post_val, batch_log_likelihood_val, bacth_chi2_val = vmapped_evaluate(batch_params)
            
            # Process results and check for NaNs
            for i in range(batch_params.shape[0]):
                total_evaluations += 1
                log_post_val = float(batch_log_post_val[i])
                log_lik_val = float(batch_log_likelihood_val[i])
                chi2_val = float(bacth_chi2_val[i])

                # Check for NaNs and Infs
                log_post_is_nan = np.isnan(log_post_val)
                log_post_is_inf = np.isinf(log_post_val)
                log_lik_is_nan = np.isnan(log_lik_val)
                log_lik_is_inf = np.isinf(log_lik_val)
                chi2_is_nan = np.isnan(chi2_val)
                chi2_is_inf = np.isinf(chi2_val)
            
                if log_post_is_nan or log_lik_is_nan or chi2_is_nan:
                    nan_count += 1
                    param_idx = start_idx + i
                    print(f"  NaN detected at parameter set {param_idx}:")
                    print(f"    Parameters: {parameter_sets[param_idx]}")
                    print(f"    Log posterior NaN: {log_post_is_nan}")
                    print(f"    Log likelihood NaN: {log_lik_is_nan}")
                    
                if log_post_is_inf or log_lik_is_inf or chi2_is_inf:
                    inf_count += 1
                    param_idx = start_idx + i
                    print(f"  Inf detected at parameter set {param_idx}:")
                    print(f"    Parameters: {parameter_sets[param_idx]}")
                    print(f"    Log posterior Inf: {log_post_is_inf} (value: {log_post_val})")
                    print(f"    Log likelihood Inf: {log_lik_is_inf} (value: {log_lik_val})")
                
                    
                log_posterior_values.append(log_post_val)
                log_likelihood_values.append(log_lik_val)
                chi2_values.append(chi2_val)
                
        except Exception as batch_e:
            failed_batches += 1
            raise ValueError(f"Batch evaluation failed: {batch_e}.")
            
    log_posterior_values = np.array(log_posterior_values)
    log_lik_values = np.array(log_likelihood_values)
    chi2_values = np.array(chi2_values)
    
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
    return log_posterior_values, log_lik_values, chi2_values, evaluation_stats, valid_mask

# In the `value_surface` function, add a circle mark to indicate the point of maximum posterior.

def value_surface(data_params, parameter_sets, log_posterior_values, log_lik_values, param_names, valid_mask, output_dir):
    n_params_per_blob = parameter_sets.shape[2]
    # Extract parameter values into arrays
    param_arrays = {}
    for idx, param_name in enumerate(param_names):
        blob_idx = idx // n_params_per_blob
        param_idx = idx % n_params_per_blob
        values = parameter_sets[:, blob_idx, param_idx]
        param_arrays[param_name] = values
    
    # Extract true parameter values
    true_param_values = {}
    for idx, param_name in enumerate(param_names):
        blob_idx = idx // n_params_per_blob
        param_idx = idx % n_params_per_blob
        true_param_values[param_name] = data_params[blob_idx, param_idx]

    param_pairs = [(param_names[i], param_names[j]) for i in range(len(param_names)) 
                   for j in range(i+1, len(param_names))]
    
    # Find the point of maximum posterior
    max_post_idx = jnp.argmax(log_posterior_values[valid_mask])
    max_post_params = {param: param_arrays[param][valid_mask][max_post_idx] for param in param_names}

    for pair_idx, (param1, param2) in enumerate(param_pairs):
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'Surfaces: {param1} vs {param2}', fontsize=16)
        param1_label = ''
        param2_label = ''
        if "center_x" in param1:
            param1_label =f"$\\mu_x$"
        elif "center_y" in param1:
            param1_label =f"$\\mu_y$"
        elif "center_z" in param1:
            param1_label =f"$\\mu_z$"
        elif "sigma" in param1:
            param1_label =f"$\\sigma$"
        else :
            param1_label = param1
        if "center_x" in param2:
            param2_label =f"$\\mu_x$"
        elif "center_y" in param2:
            param2_label =f"$\\mu_y$"
        elif "center_z" in param2:
            param2_label =f"$\\mu_z$"
        elif "sigma" in param2:
            param2_label =f"$\\sigma$"
        else :
            param2_label = param2            

        # Get parameter grids
        p1_vals = param_arrays[param1][valid_mask]
        p2_vals = param_arrays[param2][valid_mask]
        like_vals = log_lik_values[valid_mask]
        post_vals = log_posterior_values[valid_mask]
        
        # Get true values for this pair
        true_p1 = float(true_param_values[param1])
        true_p2 = float(true_param_values[param2])
        
        # Get maximum posterior values for this pair
        max_p1 = float(max_post_params[param1])
        max_p2 = float(max_post_params[param2])
        
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
            ax1.scatter(true_p1, true_p2, c='red', s=100, marker='*', label=f'True ({true_p1:.3f}, {true_p2:.3f})')
            ax1.scatter(max_p1, max_p2, c='blue', s=100, marker='o', label=f'Max Posterior ({max_p1:.3f}, {max_p2:.3f})')
            ax1.set_xlabel(param1)
            ax1.set_ylabel(param2)
            ax1.set_title('2D Likelihood Surface (Averaged)')
            ax1.legend()
            plt.colorbar(scatter1, ax=ax1, label='Log Likelihood')
            
            # Posterior surface 2D
            ax2 = plt.subplot(2, 2, 2)
            scatter2 = ax2.tricontourf(p1_avg, p2_avg, post_avg, levels=20, cmap='plasma', alpha=0.8)
            ax2.scatter(p1_avg, p2_avg, c=post_avg, cmap='plasma', s=10, alpha=0.6)

                        # Posterior surface 2D
            ax2 = plt.subplot(2, 2, 2)
            scatter2 = ax2.tricontourf(p1_avg, p2_avg, post_avg, levels=20, cmap='plasma', alpha=0.8)
            ax2.scatter(p1_avg, p2_avg, c=post_avg, cmap='plasma', s=10, alpha=0.6)

            # Compute confidence contours (returns increasing levels inside [min,max])
            levels = compute_confidence_contours(p1_avg, p2_avg, post_avg, [0.68, 0.95])

            if len(levels) > 0:
                # draw contours
                contour_lines = ax2.tricontour(
                    p1_avg, p2_avg, post_avg,
                    levels=levels,
                    colors='white', linewidths=2, linestyles='--'
                )
                # label with matching CLs
                cl_labels = [f"{int(cl*100)}%" for cl in [0.68, 0.95][:len(levels)]]
                fmt_map = {lvl: lab for lvl, lab in zip(levels, cl_labels)}
                ax2.clabel(contour_lines, inline=True, fontsize=10, fmt=fmt_map)


            # Update the scatter point labels
            ax2.scatter(true_p1, true_p2, c='red', s=100, marker='*', label=f'True ({true_p1:.3f}, {true_p2:.3f})')
            #map_uncertainties = compute_map_uncertainties(parameter_sets, log_posterior_values, valid_mask, param_names)

            # Update the scatter point labels to show uncertainties
            #map_p1_unc = map_uncertainties[param1]
            #map_p2_unc = map_uncertainties[param2]

            ax2.scatter(max_p1, max_p2, c='blue', s=100, marker='o', 
                    label=f'MAP ({max_p1:.3f}, {max_p2:.3f})')

            ax2.set_xlabel(param1_label)
            ax2.set_ylabel(param2_label)
            ax2.set_title('2D Posterior Surface with Confidence Levels')
            ax2.legend()
            plt.colorbar(scatter2, ax=ax2, label='Log Posterior')
            
            # Second row: 3D surfaces
            # Likelihood surface 3D
            ax3 = plt.subplot(2, 2, 3, projection='3d')
            ax3.plot_trisurf(p1_avg, p2_avg, like_avg, cmap='viridis', alpha=0.8)
            ax3.scatter(p1_avg, p2_avg, like_avg, c=like_avg, cmap='viridis', s=20, alpha=0.6)
            ax3.scatter(true_p1, true_p2, np.interp([true_p1, true_p2], [p1_avg.min(), p2_avg.min()], [like_avg.min(), like_avg.max()])[0], c='red', s=100, marker='*')
            ax3.scatter(max_p1, max_p2, np.interp([max_p1, max_p2], [p1_avg.min(), p2_avg.min()], [like_avg.min(), like_avg.max()])[0], c='blue', s=100, marker='o')
            ax3.set_xlabel(param1_label)
            ax3.set_ylabel(param2_label)
            ax3.set_zlabel('Log Likelihood')
            ax3.set_title('3D Likelihood Surface (Averaged)')
            
            # Posterior surface 3D
            ax4 = plt.subplot(2, 2, 4, projection='3d')
            ax4.plot_trisurf(p1_avg, p2_avg, post_avg, cmap='plasma', alpha=0.8)
            ax4.scatter(p1_avg, p2_avg, post_avg, c=post_avg, cmap='plasma', s=20, alpha=0.6)
            ax4.scatter(true_p1, true_p2, np.interp([true_p1, true_p2], [p1_avg.min(), p2_avg.min()], [post_avg.min(), post_avg.max()])[0], c='red', s=100, marker='*')
            ax4.scatter(max_p1, max_p2, np.interp([max_p1, max_p2], [p1_avg.min(), p2_avg.min()], [post_avg.min(), post_avg.max()])[0], c='blue', s=100, marker='o')
            ax4.set_xlabel(param1_label)
            ax4.set_ylabel(param2_label)
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

def compute_confidence_contours(p1_vals, p2_vals, post_vals, confidence_levels=[0.68, 0.95]):
    """
    Return contour levels in *increasing* order for log-posterior 'post_vals'.
    """
    # Convert to probabilities relative to max log posterior
    max_post = np.nanmax(post_vals)
    log_post_normalized = post_vals - max_post
    probabilities = np.exp(log_post_normalized)

    # Build cumulative mass over descending probs
    sorted_probs = np.sort(probabilities)[::-1]
    cdf = np.cumsum(sorted_probs)
    cdf /= cdf[-1]

    # Map each CL to a probability threshold, then back to log-post level
    levels = []
    for cl in confidence_levels:
        idx = np.searchsorted(cdf, cl, side="left")
        if idx >= len(sorted_probs):
            continue
        prob_threshold = sorted_probs[idx]
        level = np.log(prob_threshold) + max_post  # a log-posterior *value*
        levels.append(level)

    # Sanitize: finite, unique, strictly increasing, and inside data range
    levels = np.asarray(levels, dtype=float)
    levels = levels[np.isfinite(levels)]
    levels = np.unique(levels)

    # Enforce strictly increasing and clip to Z-range
    zmin = np.nanmin(post_vals)
    zmax = np.nanmax(post_vals)
    levels = levels[(levels > zmin) & (levels < zmax)]
    levels.sort()

    # In rare cases two levels collapse to same value; add tiny eps to separate
    if levels.size > 1:
        for i in range(1, levels.size):
            if levels[i] <= levels[i-1]:
                levels[i] = np.nextafter(levels[i-1], np.inf)

    return levels.tolist()


def values_slices(data_params, parameter_sets, log_posterior_values, log_lik_values, param_names, valid_mask, output_dir):
    n_params_per_blob = parameter_sets.shape[2]
    # Extract parameter values into arrays
    param_arrays = {}
    for idx, param_name in enumerate(param_names):
        blob_idx = idx // n_params_per_blob
        param_idx = idx % n_params_per_blob
        values = parameter_sets[:, blob_idx, param_idx]
        param_arrays[param_name] = values

    # Extract true parameter values
    true_param_values = {}
    for idx, param_name in enumerate(param_names):
        blob_idx = idx // n_params_per_blob
        param_idx = idx % n_params_per_blob
        true_param_values[param_name] = data_params[blob_idx, param_idx]

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
            
            # Get true value for this parameter
            true_param_val = float(true_param_values[param_name])
            
            # Create subplot for this parameter
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'Parameter Slice: {param_name} (Averaged over Other Parameters)', fontsize=14)
            
            # Plot 1: Averaged Likelihood vs Parameter
            ax = axes[0]
            ax.errorbar(param_vals_plot, avg_like_plot, yerr=std_like_plot, 
                       marker='o', linestyle='-', capsize=5, alpha=0.8, linewidth=2, markersize=6)
            ax.axvline(true_param_val, color='red', linestyle='--', linewidth=2, 
                      label=f'True value ({true_param_val:.3f})')
            ax.set_xlabel(param_name)
            ax.set_ylabel('Average Log Likelihood')
            ax.set_title(f'Average Likelihood vs {param_name}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add text showing number of realizations averaged
            n_realizations = [np.sum(valid_mask & (param_arrays[param_name] == pv)) for pv in param_vals_plot]
            ax.text(0.02, 0.98, f'Avg over {max(n_realizations)} realizations', 
                   transform=ax.transAxes, verticalalignment='top', fontsize=10, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Plot 2: Averaged Posterior vs Parameter
            ax = axes[1]
            ax.errorbar(param_vals_plot, avg_post_plot, yerr=std_post_plot, 
                       marker='s', linestyle='-', capsize=5, alpha=0.8, linewidth=2, markersize=6, color='orange')
            ax.axvline(true_param_val, color='red', linestyle='--', linewidth=2, 
                      label=f'True value ({true_param_val:.3f})')
            ax.set_xlabel(param_name)
            ax.set_ylabel('Average Log Posterior')
            ax.set_title(f'Average Posterior vs {param_name}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add text showing number of realizations averaged
            ax.text(0.02, 0.98, f'Avg over {max(n_realizations)} realizations', 
                   transform=ax.transAxes, verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            fig_name = f'slice_{param_name}.png'
            dir = os.path.join(output_dir, fig_name)
            plt.tight_layout()
            plt.savefig(dir, dpi=300, bbox_inches='tight')
            plt.close()
            
        else:
            print(f"No valid data for parameter {param_name}")

def plot_chi2_distribution(chi2_values, dof, output_dir):
    import scipy.stats as stats
    print(chi2_values)
    plt.figure(figsize=(8,6))
    plt.hist(chi2_values, bins=30, density=True, alpha=0.6, label="Observed χ² values")
    
    # Theoretical chi² distribution
    x = np.linspace(0, max(chi2_values)*1.1, 200)
    plt.plot(x, stats.chi2.pdf(x, dof), 'r-', lw=2, label=f'χ² PDF (dof={dof})')
    
    plt.xlabel("χ² value")
    plt.ylabel("Density")
    plt.title("Goodness of Fit: χ² Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    fig_name = os.path.join(output_dir, "chi2_distribution.png")
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    plt.close()

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

def quiver_grad_surface(parameter_sets, log_lik_values, log_posterior_values, likelihood_gradient_values, posterior_gradient_values, param_names, valid_mask, output_dir):
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
    likelihood_grad_arrays = {}
    posterior_grad_arrays = {}
    for idx, param_name in enumerate(param_names):
        blob_idx = idx // n_params_per_blob
        param_idx = idx % n_params_per_blob
        # Extract gradient for this parameter from all parameter sets
        lik_grad_values = []
        post_grad_values = []
        for i in range(len(parameter_sets)):
            if valid_mask[i]:
                lik_grad_val = likelihood_gradient_values[i, blob_idx, param_idx]
                post_grad_val = posterior_gradient_values[i, blob_idx, param_idx]
                lik_grad_values.append(lik_grad_val)
                post_grad_values.append(post_grad_val)
            else:
                lik_grad_values.append(np.nan)
                post_grad_values.append(np.nan)
        likelihood_grad_arrays[param_name] = np.array(lik_grad_values)
        posterior_grad_arrays[param_name] = np.array(post_grad_values)

    for pair_idx, (param1, param2) in enumerate(param_pairs):
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))
        fig.suptitle(f'Gradient Quiver Plots: {param1} vs {param2}', fontsize=16)
        
        # Get parameter grids
        p1_vals = param_arrays[param1][valid_mask]
        p2_vals = param_arrays[param2][valid_mask]
        like_vals = log_lik_values[valid_mask]
        post_vals = log_posterior_values[valid_mask]
        
        # Get gradients for this parameter pair
        lik_grad1_vals = likelihood_grad_arrays[param1][valid_mask]
        lik_grad2_vals = likelihood_grad_arrays[param2][valid_mask]
        post_grad1_vals = posterior_grad_arrays[param1][valid_mask]
        post_grad2_vals = posterior_grad_arrays[param2][valid_mask]
        
        try:
            # Marginalize over other parameters by averaging
            coords = np.column_stack([p1_vals, p2_vals])
            unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
            
            # Average values and gradients for each unique coordinate
            like_avg = np.array([like_vals[inverse == i].mean() for i in range(len(unique_coords))])
            post_avg = np.array([post_vals[inverse == i].mean() for i in range(len(unique_coords))])
            
            lik_grad1_avg = np.array([lik_grad1_vals[inverse == i].mean() for i in range(len(unique_coords))])
            lik_grad2_avg = np.array([lik_grad2_vals[inverse == i].mean() for i in range(len(unique_coords))])
            post_grad1_avg = np.array([post_grad1_vals[inverse == i].mean() for i in range(len(unique_coords))])
            post_grad2_avg = np.array([post_grad2_vals[inverse == i].mean() for i in range(len(unique_coords))])
            
            p1_avg = unique_coords[:, 0]
            p2_avg = unique_coords[:, 1]
            
            # Plot 1: Likelihood gradient quiver
            ax = axes[0]
            scatter1 = ax.tricontourf(p1_avg, p2_avg, like_avg, levels=20, cmap='plasma', alpha=0.6)
            plt.colorbar(scatter1, ax=ax, label='Log Likelihood')
            
            # Filter out NaN gradients
            valid_grad_mask = ~(np.isnan(lik_grad1_avg) | np.isnan(lik_grad2_avg))
            
            if np.sum(valid_grad_mask) > 0:
                p1_quiver = p1_avg[valid_grad_mask]
                p2_quiver = p2_avg[valid_grad_mask]
                grad1_quiver = lik_grad1_avg[valid_grad_mask]
                grad2_quiver = lik_grad2_avg[valid_grad_mask]
                
                # Normalize gradients for better visualization
                grad_magnitude = np.sqrt(grad1_quiver**2 + grad2_quiver**2)
                max_grad = np.max(grad_magnitude)
                
                if max_grad > 0:
                    scale_factor = 0.1 * (np.max(p1_avg) - np.min(p1_avg)) / max_grad
                    
                    quiver1 = ax.quiver(p1_quiver, p2_quiver, 
                                        grad1_quiver, grad2_quiver,
                                        grad_magnitude,
                                        cmap='RdBu_r', alpha=0.8, 
                                        scale=1/scale_factor, scale_units='xy',
                                        angles='xy', width=0.003)
                    
                    # Add colorbar for gradient magnitude
                    plt.colorbar(quiver1, ax=ax, label='Gradient Magnitude', 
                                        orientation='horizontal', pad=0.1, shrink=0.8) 
                    
                else:
                    ax.text(0.5, 0.5, 'All gradients are zero', 
                            transform=ax.transAxes, ha='center', va='center')
            else:
                ax.text(0.5, 0.5, 'No valid gradients available', 
                        transform=ax.transAxes, ha='center', va='center')
            
            ax.set_xlabel(param1)
            ax.set_ylabel(param2)
            ax.set_title(f'Likelihood Surface with Gradient Field (Marginalized)')
            
            # Plot 2: Posterior gradient quiver
            ax = axes[1]
            scatter2 = ax.tricontourf(p1_avg, p2_avg, post_avg, levels=20, cmap='plasma', alpha=0.6)
            plt.colorbar(scatter2, ax=ax, label='Log Posterior')
            
            # Filter out NaN gradients
            valid_grad_mask = ~(np.isnan(post_grad1_avg) | np.isnan(post_grad2_avg))
            
            if np.sum(valid_grad_mask) > 0:
                p1_quiver = p1_avg[valid_grad_mask]
                p2_quiver = p2_avg[valid_grad_mask]
                grad1_quiver = post_grad1_avg[valid_grad_mask]
                grad2_quiver = post_grad2_avg[valid_grad_mask]
                
                # Normalize gradients for better visualization
                grad_magnitude = np.sqrt(grad1_quiver**2 + grad2_quiver**2)
                max_grad = np.max(grad_magnitude)
                
                if max_grad > 0:
                    scale_factor = 0.1 * (np.max(p1_avg) - np.min(p1_avg)) / max_grad
                    
                    quiver2 = ax.quiver(p1_quiver, p2_quiver, 
                                        grad1_quiver, grad2_quiver,
                                        grad_magnitude,
                                        cmap='RdBu_r', alpha=0.8, 
                                        scale=1/scale_factor, scale_units='xy',
                                        angles='xy', width=0.003)
                    
                    # Add colorbar for gradient magnitude
                    plt.colorbar(quiver2, ax=ax, label='Gradient Magnitude', 
                                        orientation='horizontal', pad=0.1, shrink=0.8) 
                    
                else:
                    ax.text(0.5, 0.5, 'All gradients are zero', 
                            transform=ax.transAxes, ha='center', va='center')
            else:
                ax.text(0.5, 0.5, 'No valid gradients available', 
                        transform=ax.transAxes, ha='center', va='center')
            
            ax.set_xlabel(param1)
            ax.set_ylabel(param2)
            ax.set_title(f'Posterior Surface with Gradient Field (Marginalized)')
            
        except Exception as e:
            print(f"Warning: Could not create quiver plots for {param1} vs {param2}: {e}")
            for ax in axes:
                ax.text(0.5, 0.5, f'Quiver plot failed:\n{str(e)}', 
                        transform=ax.transAxes, ha='center', va='center')
        
        fig_name = f'gradient_quiver_{param1}_vs_{param2}.png'
        dir = os.path.join(output_dir, fig_name)
        plt.tight_layout()
        plt.savefig(dir, dpi=300, bbox_inches='tight')
        plt.close()
    
def compute_map_uncertainties(parameter_sets, log_posterior_values, valid_mask, param_names):
    """
    Compute uncertainties around MAP estimate using the posterior distribution.
    """
    # Find MAP
    valid_posterior = log_posterior_values[valid_mask]
    max_idx = np.argmax(valid_posterior)
    valid_indices = np.where(valid_mask)[0]
    map_idx = valid_indices[max_idx]
    
    uncertainties = {}
    n_params_per_blob = parameter_sets.shape[2]
    
    for idx, param_name in enumerate(param_names):
        blob_idx = idx // n_params_per_blob
        param_idx = idx % n_params_per_blob
        param_values = parameter_sets[valid_mask, blob_idx, param_idx]
        
        # Convert to probabilities
        valid_post = log_posterior_values[valid_mask]
        max_post = np.max(valid_post)
        #print(f"Max posterior for {param_name}: {max_post}")    
        log_post_norm = valid_post - max_post
        #print(f"Normalized log posterior for {param_name}: {log_post_norm}")
        probs = np.exp(log_post_norm)
        print(f"Probabilities for {param_name}: {probs}")
        print(f"Sum of probabilities for {param_name}: {np.sum(probs)}")
        probs = probs / np.sum(probs)

        print(probs)
        
        # Compute weighted std deviation around MAP
        map_value = float(parameter_sets[map_idx, blob_idx, param_idx])
        #print(f"MAP value: {map_value}")
        weighted_variance = np.sum(probs * (param_values - map_value)**2)
        #print(f"Weighted variance for :{weighted_variance}")
        uncertainty = np.sqrt(weighted_variance)
        uncertainties[param_name] = uncertainty
    
    return uncertainties

