import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

from test_utils import load_config, calculate_radius_evolution
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from model import model
from likelihood import get_log_posterior

# Import functions from other test scripts
from gradient_sanity_check import create_parameter_grid as create_gradient_grid


def create_analysis_parameter_grid(config):
    """Create parameter grid for analysis with all combinations."""
    analysis_params = config['analysis_params']
    param_bounds = analysis_params['param_bounds']
    n_points = analysis_params['n_points_per_param']
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
        n_points = bounds.get('n_points', 5)
        if 'center' in param_name.lower():
            # For center parameters, use scalar bounds (all coordinates are the same)
            min_val = bounds['min']
            max_val = bounds['max']
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

def run_simulations_for_grid(parameter_sets, config):
    """Run simulations for all parameter combinations in the grid."""
    print(f"Running simulations for {len(parameter_sets)} parameter combinations...")
    
    model_params = config['model_params'].copy()
    analysis_params = config['analysis_params']
    t_f_range = analysis_params['t_f_range']
    max_t_f = max(t_f_range)
    
    # Extract model parameters
    G = model_params['G']
    length = model_params['length']
    softening = model_params['softening']
    dt = model_params['dt']
    solver = model_params.get('solver', 'LeapfrogMidpoint')
    density_scaling = model_params.get('density_scaling', 'none')
    scaling_kwargs = model_params.get('scaling_kwargs', {})
    
    batch_size = analysis_params.get('batch_size', 10)
    results = {}
    
    # Process in batches
    for start_idx in range(0, len(parameter_sets), batch_size):
        end_idx = min(start_idx + batch_size, len(parameter_sets))
        print(f"Processing batch {start_idx//batch_size + 1}/{(len(parameter_sets) + batch_size - 1)//batch_size}")
        
        for idx in range(start_idx, end_idx):
            param_set = parameter_sets[idx]
            
            # Update blob parameters with current parameter set
            updated_blobs_params = []
            base_blobs_params = model_params['blobs_params']
            
            for blob_idx, blob in enumerate(base_blobs_params):
                updated_blob = dict(blob)
                
                # Update position parameters
                if updated_blob['pos_type'] == 'gaussian':
                    updated_pos_params = dict(updated_blob['pos_params'])
                    if f"blob{blob_idx}_sigma" in param_set:
                        updated_pos_params['sigma'] = param_set[f"blob{blob_idx}_sigma"]
                    if f"blob{blob_idx}_center" in param_set:
                        updated_pos_params['center'] = param_set[f"blob{blob_idx}_center"]
                    updated_blob['pos_params'] = updated_pos_params
                
                # Update velocity parameters
                updated_vel_params = dict(updated_blob['vel_params'])
                if f"blob{blob_idx}_vel_factor" in param_set:
                    updated_vel_params['vel_factor'] = param_set[f"blob{blob_idx}_vel_factor"]
                updated_blob['vel_params'] = updated_vel_params
                
                updated_blobs_params.append(updated_blob)
            
            # Generate simulation with all time points
            key = jax.random.PRNGKey(model_params.get('data_seed', 42) + idx)
            
            mock_data = model(
                blobs_params=updated_blobs_params,
                G=G,
                length=length,
                softening=softening,
                t_f=max_t_f,
                dt=dt,
                ts=jnp.array(t_f_range),
                key=key,
                solver=solver,
                density_scaling=density_scaling,
                **scaling_kwargs
            )
            
            solution = mock_data[4]
            
            # Calculate radius evolution for all blobs
            radius_data = calculate_radius_evolution(solution, updated_blobs_params)
            
            # Store results for each time point
            for time_idx, t_f in enumerate(t_f_range):
                if t_f not in results:
                    results[t_f] = {
                        'parameter_sets': [],
                        'radius_data': [],
                        'density_fields': [],
                        'solutions': []
                    }
                
                results[t_f]['parameter_sets'].append(param_set)
                results[t_f]['radius_data'].append(radius_data)
                results[t_f]['density_fields'].append(mock_data[3])  # output_field
                results[t_f]['solutions'].append(solution)
                
    return results

def evaluate_likelihood_and_gradients_for_grid(results, config):
    """Evaluate likelihood and gradients for all parameter combinations at each time point."""
    print("Evaluating likelihood and gradients for all time points...")
    
    model_params = config['model_params'].copy()
    ll = config['likelihood_type']
    
    # Create log posterior function template
    init_params = model_params['blobs_params']
    
    for t_f, data in results.items():
        print(f"Evaluating at t_f = {t_f:.3f}...")
        
        log_posterior_values = []
        gradient_values = []
        
        # Use first valid density field as observed data for this time point
        observed_data = data['density_fields']
        
        # Create log posterior function for this time point
        model_params['t_f'] = t_f
        log_posterior_fn = get_log_posterior(
            likelihood_type=ll,
            data=observed_data,
            prior_params=config['prior_params'], #make a prior centered in value
            prior_type=config['prior_type'],
            model_fn=model,
            init_params=init_params,
            **config['likelihood_kwargs'],
            **{k: v for k, v in model_params.items() if k != 'blobs_params'}
        )
        
        value_and_grad_fn = jax.value_and_grad(log_posterior_fn)
        
        # Evaluate for each parameter set
        for param_set in data['parameter_sets']:
            try:
                log_post_val, grad_val = value_and_grad_fn(param_set)
                log_posterior_values.append(float(log_post_val))
                gradient_values.append(grad_val)
            except Exception as e:
                print(f"Warning: Failed to evaluate gradients for {param_set}: {e}")
                log_posterior_values.append(np.nan)
                gradient_values.append({k: np.nan for k in param_set.keys()})
        
        # Store results
        data['log_posterior_values'] = np.array(log_posterior_values)
        data['gradient_values'] = gradient_values
    
    return results

def create_blob_radius_evolution_analysis(results, param_names, output_dir):
    """Create blob radius evolution plots for each sigma parameter."""
    print("Creating blob radius evolution analysis...")
    
    # Find sigma parameters
    sigma_params = [name for name in param_names if 'sigma' in name.lower()]
    
    if len(sigma_params) == 0:
        print("No sigma parameters found, skipping radius evolution analysis")
        return
    
    t_f_values = sorted(results.keys())
    
    # Create figure for radius evolution analysis
    fig, axes = plt.subplots(len(sigma_params), 2, figsize=(15, 6 * len(sigma_params)))
    if len(sigma_params) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Blob Radius Evolution Analysis vs Time', fontsize=16)
    
    for sigma_idx, sigma_param in enumerate(sigma_params):
        print(f"Processing {sigma_param}...")
        
        # Extract blob index from parameter name
        blob_idx = int(sigma_param.split('blob')[1].split('_')[0])
        
        # Get unique sigma values across all time points
        all_sigma_values = set()
        for t_f in t_f_values:
            for param_set in results[t_f]['parameter_sets']:
                if sigma_param in param_set:
                    all_sigma_values.add(float(param_set[sigma_param]))
        
        sigma_values = sorted(list(all_sigma_values))
        colors = plt.cm.viridis(np.linspace(0, 1, len(sigma_values)))
        
        # Plot for each sigma value
        for color_idx, sigma_val in enumerate(sigma_values):
            mean_percentile_radii = []
            std_percentile_radii = []
            mean_rms_radii = []
            std_rms_radii = []
            
            for t_f in t_f_values:
                # Find all parameter sets with this sigma value
                percentile_radii_at_tf = []
                rms_radii_at_tf = []
                
                for param_idx, param_set in enumerate(results[t_f]['parameter_sets']):
                    if (sigma_param in param_set and 
                        abs(float(param_set[sigma_param]) - sigma_val) < 1e-6 and
                        results[t_f]['radius_data'][param_idx] is not None):
                        
                        radius_data = results[t_f]['radius_data'][param_idx]
                        blob_key = f'blob{blob_idx}'
                        
                        if blob_key in radius_data:
                            # Get radius at the final time (last index)
                            perc_radius = radius_data[blob_key]['percentile_radii'][-1]
                            rms_radius = radius_data[blob_key]['rms_radii'][-1]
                            
                            percentile_radii_at_tf.append(float(perc_radius))
                            rms_radii_at_tf.append(float(rms_radius))
                
                # Calculate statistics
                if len(percentile_radii_at_tf) > 0:
                    mean_percentile_radii.append(np.mean(percentile_radii_at_tf))
                    std_percentile_radii.append(np.std(percentile_radii_at_tf))
                    mean_rms_radii.append(np.mean(rms_radii_at_tf))
                    std_rms_radii.append(np.std(rms_radii_at_tf))
                else:
                    mean_percentile_radii.append(np.nan)
                    std_percentile_radii.append(0)
                    mean_rms_radii.append(np.nan)
                    std_rms_radii.append(0)
            
            # Plot percentile radius evolution
            ax = axes[sigma_idx, 0]
            valid_mask = ~np.isnan(mean_percentile_radii)
            if np.sum(valid_mask) > 0:
                ax.errorbar(np.array(t_f_values)[valid_mask], 
                           np.array(mean_percentile_radii)[valid_mask], 
                           yerr=np.array(std_percentile_radii)[valid_mask], 
                           marker='o', linestyle='-', capsize=5, alpha=0.8, 
                           linewidth=2, markersize=6, color=colors[color_idx],
                           label=f'σ={sigma_val:.1f}')
            
            # Plot RMS radius evolution
            ax = axes[sigma_idx, 1]
            valid_mask = ~np.isnan(mean_rms_radii)
            if np.sum(valid_mask) > 0:
                ax.errorbar(np.array(t_f_values)[valid_mask], 
                           np.array(mean_rms_radii)[valid_mask], 
                           yerr=np.array(std_rms_radii)[valid_mask], 
                           marker='s', linestyle='-', capsize=5, alpha=0.8, 
                           linewidth=2, markersize=6, color=colors[color_idx],
                           label=f'σ={sigma_val:.1f}')
        
        # Configure axes
        axes[sigma_idx, 0].set_xlabel('Final Time (t_f)')
        axes[sigma_idx, 0].set_ylabel('90th Percentile Radius')
        axes[sigma_idx, 0].set_title(f'Blob {blob_idx} - 90th Percentile Radius vs Time')
        axes[sigma_idx, 0].legend()
        axes[sigma_idx, 0].grid(True, alpha=0.3)
        
        axes[sigma_idx, 1].set_xlabel('Final Time (t_f)')
        axes[sigma_idx, 1].set_ylabel('RMS Radius')
        axes[sigma_idx, 1].set_title(f'Blob {blob_idx} - RMS Radius vs Time')
        axes[sigma_idx, 1].legend()
        axes[sigma_idx, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'blob_radius_evolution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_parameter_surface_plots(results, param_names, output_dir):
    """Create likelihood, posterior, and gradient surfaces for parameter pairs."""
    print("Creating parameter surface plots...")
    
    t_f_values = sorted(results.keys())
    param_pairs = [(param_names[i], param_names[j]) for i in range(len(param_names)) 
                   for j in range(i+1, len(param_names))]
    
    for param1, param2 in param_pairs:
        print(f"Creating surfaces for {param1} vs {param2}...")
        
        # Select representative time points for visualization
        selected_tf = t_f_values[::max(1, len(t_f_values)//4)]  # ~4 time points
        
        fig, axes = plt.subplots(len(selected_tf), 3, figsize=(18, 6 * len(selected_tf)))
        if len(selected_tf) == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Parameter Analysis: {param1} vs {param2}', fontsize=16)
        
        for time_idx, t_f in enumerate(selected_tf):
            if t_f not in results or 'log_posterior_values' not in results[t_f]:
                continue
            
            # Extract parameter values and metrics
            param1_vals = []
            param2_vals = []
            likelihood_vals = []
            posterior_vals = []
            grad1_vals = []
            grad2_vals = []
            
            for param_idx, param_set in enumerate(results[t_f]['parameter_sets']):
                if param1 in param_set and param2 in param_set:
                    # Get parameter values
                    p1_val = param_set[param1]
                    p2_val = param_set[param2]
                    
                    # Handle vector parameters (extract scalar)
                    if isinstance(p1_val, (list, tuple, jnp.ndarray)):
                        p1_val = float(p1_val[0])
                    else:
                        p1_val = float(p1_val)
                    
                    if isinstance(p2_val, (list, tuple, jnp.ndarray)):
                        p2_val = float(p2_val[0])
                    else:
                        p2_val = float(p2_val)
                    
                    param1_vals.append(p1_val)
                    param2_vals.append(p2_val)
                    
                    # Get posterior value
                    log_post = results[t_f]['log_posterior_values'][param_idx]
                    posterior_vals.append(log_post)
                    
                    # Calculate likelihood (posterior - prior)
                    try:
                        from likelihood import PRIOR_REGISTRY
                        prior_fn = PRIOR_REGISTRY.get(config['prior_type'])
                        log_prior = prior_fn(param_set, config['prior_params'])
                        likelihood_vals.append(log_post - log_prior)
                    except:
                        likelihood_vals.append(np.nan)
                    
                    # Get gradient values
                    grad_dict = results[t_f]['gradient_values'][param_idx]
                    if isinstance(grad_dict, dict):
                        grad1 = grad_dict.get(param1, np.nan)
                        grad2 = grad_dict.get(param2, np.nan)
                        
                        # Handle vector gradients
                        if isinstance(grad1, (list, tuple, jnp.ndarray)):
                            grad1 = float(jnp.linalg.norm(grad1))
                        else:
                            grad1 = float(grad1) if not np.isnan(grad1) else np.nan
                        
                        if isinstance(grad2, (list, tuple, jnp.ndarray)):
                            grad2 = float(jnp.linalg.norm(grad2))
                        else:
                            grad2 = float(grad2) if not np.isnan(grad2) else np.nan
                        
                        grad1_vals.append(grad1)
                        grad2_vals.append(grad2)
                    else:
                        grad1_vals.append(np.nan)
                        grad2_vals.append(np.nan)
            
            # Convert to numpy arrays
            param1_vals = np.array(param1_vals)
            param2_vals = np.array(param2_vals)
            likelihood_vals = np.array(likelihood_vals)
            posterior_vals = np.array(posterior_vals)
            grad1_vals = np.array(grad1_vals)
            grad2_vals = np.array(grad2_vals)
            
            # Create valid mask
            valid_mask = (~np.isnan(likelihood_vals) & ~np.isnan(posterior_vals) & 
                         ~np.isinf(likelihood_vals) & ~np.isinf(posterior_vals))
            
            if np.sum(valid_mask) < 3:
                # Not enough valid points for plotting
                for col in range(3):
                    ax = axes[time_idx, col]
                    ax.text(0.5, 0.5, f'Insufficient data\nat t_f = {t_f:.2f}', 
                           transform=ax.transAxes, ha='center', va='center')
                continue
            
            # Plot likelihood surface
            try:
                ax = axes[time_idx, 0]
                scatter = ax.scatter(param1_vals[valid_mask], param2_vals[valid_mask], 
                                   c=likelihood_vals[valid_mask], cmap='viridis', 
                                   alpha=0.7, s=30)
                ax.set_xlabel(param1)
                ax.set_ylabel(param2)
                ax.set_title(f'Likelihood Surface (t_f = {t_f:.2f})')
                plt.colorbar(scatter, ax=ax, label='Log Likelihood')
                ax.grid(True, alpha=0.3)
            except Exception as e:
                print(f"Warning: Could not create likelihood surface: {e}")
            
            # Plot posterior surface
            try:
                ax = axes[time_idx, 1]
                scatter = ax.scatter(param1_vals[valid_mask], param2_vals[valid_mask], 
                                   c=posterior_vals[valid_mask], cmap='plasma', 
                                   alpha=0.7, s=30)
                ax.set_xlabel(param1)
                ax.set_ylabel(param2)
                ax.set_title(f'Posterior Surface (t_f = {t_f:.2f})')
                plt.colorbar(scatter, ax=ax, label='Log Posterior')
                ax.grid(True, alpha=0.3)
            except Exception as e:
                print(f"Warning: Could not create posterior surface: {e}")
            
            # Plot gradient magnitude surface
            try:
                ax = axes[time_idx, 2]
                grad_mag = np.sqrt(grad1_vals**2 + grad2_vals**2)
                grad_valid_mask = ~np.isnan(grad_mag) & ~np.isinf(grad_mag)
                
                if np.sum(grad_valid_mask) > 0:
                    scatter = ax.scatter(param1_vals[grad_valid_mask], param2_vals[grad_valid_mask], 
                                       c=grad_mag[grad_valid_mask], cmap='RdYlBu_r', 
                                       alpha=0.7, s=30)
                    ax.set_xlabel(param1)
                    ax.set_ylabel(param2)
                    ax.set_title(f'Gradient Magnitude (t_f = {t_f:.2f})')
                    plt.colorbar(scatter, ax=ax, label='|Gradient|')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No valid gradients', transform=ax.transAxes, 
                           ha='center', va='center')
            except Exception as e:
                print(f"Warning: Could not create gradient surface: {e}")
        
        plt.tight_layout()
        plt.savefig(output_dir / f'surfaces_{param1}_vs_{param2}.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_1d_slice_analysis(results, param_names, output_dir):
    """Create 1D slice analysis for each parameter."""
    print("Creating 1D slice analysis...")
    
    t_f_values = sorted(results.keys())
    
    for param_name in param_names:
        print(f"Creating 1D analysis for {param_name}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'1D Analysis: {param_name} vs Time', fontsize=16)
        
        # Collect data across all time points
        times = []
        mean_likelihoods = []
        std_likelihoods = []
        mean_posteriors = []
        std_posteriors = []
        mean_gradients = []
        std_gradients = []
        
        for t_f in t_f_values:
            if t_f not in results or 'log_posterior_values' not in results[t_f]:
                continue
            
            # Get unique parameter values
            param_values = []
            for param_set in results[t_f]['parameter_sets']:
                if param_name in param_set:
                    val = param_set[param_name]
                    if isinstance(val, (list, tuple, jnp.ndarray)):
                        val = float(val[0])
                    else:
                        val = float(val)
                    param_values.append(val)
            
            unique_param_vals = sorted(list(set(param_values)))
            
            # For each unique parameter value, average over other parameters
            likelihood_means = []
            posterior_means = []
            gradient_means = []
            
            for param_val in unique_param_vals:
                likelihoods_for_val = []
                posteriors_for_val = []
                gradients_for_val = []
                
                for param_idx, param_set in enumerate(results[t_f]['parameter_sets']):
                    if param_name in param_set:
                        set_val = param_set[param_name]
                        if isinstance(set_val, (list, tuple, jnp.ndarray)):
                            set_val = float(set_val[0])
                        else:
                            set_val = float(set_val)
                        
                        if abs(set_val - param_val) < 1e-6:  # Match this parameter value
                            # Get posterior
                            log_post = results[t_f]['log_posterior_values'][param_idx]
                            if not np.isnan(log_post):
                                posteriors_for_val.append(log_post)
                                
                                # Calculate likelihood
                                try:
                                    from likelihood import PRIOR_REGISTRY
                                    prior_fn = PRIOR_REGISTRY.get(config['prior_type'])
                                    log_prior = prior_fn(param_set, config['prior_params'])
                                    likelihoods_for_val.append(log_post - log_prior)
                                except:
                                    likelihoods_for_val.append(np.nan)
                                
                                # Get gradient
                                grad_dict = results[t_f]['gradient_values'][param_idx]
                                if isinstance(grad_dict, dict) and param_name in grad_dict:
                                    grad = grad_dict[param_name]
                                    if isinstance(grad, (list, tuple, jnp.ndarray)):
                                        grad = float(jnp.linalg.norm(grad))
                                    else:
                                        grad = float(grad) if not np.isnan(grad) else np.nan
                                    
                                    if not np.isnan(grad):
                                        gradients_for_val.append(abs(grad))
                
                # Calculate means for this parameter value
                if len(likelihoods_for_val) > 0:
                    likelihood_means.append(np.mean(likelihoods_for_val))
                if len(posteriors_for_val) > 0:
                    posterior_means.append(np.mean(posteriors_for_val))
                if len(gradients_for_val) > 0:
                    gradient_means.append(np.mean(gradients_for_val))
            
            # Store time-averaged statistics
            times.append(t_f)
            mean_likelihoods.append(np.mean(likelihood_means) if likelihood_means else np.nan)
            std_likelihoods.append(np.std(likelihood_means) if len(likelihood_means) > 1 else 0)
            mean_posteriors.append(np.mean(posterior_means) if posterior_means else np.nan)
            std_posteriors.append(np.std(posterior_means) if len(posterior_means) > 1 else 0)
            mean_gradients.append(np.mean(gradient_means) if gradient_means else np.nan)
            std_gradients.append(np.std(gradient_means) if len(gradient_means) > 1 else 0)
        
        # Create plots
        times = np.array(times)
        
        # Plot likelihood vs time
        ax = axes[0, 0]
        valid_mask = ~np.isnan(mean_likelihoods)
        if np.sum(valid_mask) > 0:
            ax.errorbar(times[valid_mask], np.array(mean_likelihoods)[valid_mask], 
                       yerr=np.array(std_likelihoods)[valid_mask], 
                       marker='o', linestyle='-', capsize=5, alpha=0.8, linewidth=2)
        ax.set_xlabel('Time (t_f)')
        ax.set_ylabel('Mean Log Likelihood')
        ax.set_title(f'Mean Likelihood vs Time')
        ax.grid(True, alpha=0.3)
        
        # Plot posterior vs time
        ax = axes[0, 1]
        valid_mask = ~np.isnan(mean_posteriors)
        if np.sum(valid_mask) > 0:
            ax.errorbar(times[valid_mask], np.array(mean_posteriors)[valid_mask], 
                       yerr=np.array(std_posteriors)[valid_mask], 
                       marker='s', linestyle='-', capsize=5, alpha=0.8, linewidth=2, color='orange')
        ax.set_xlabel('Time (t_f)')
        ax.set_ylabel('Mean Log Posterior')
        ax.set_title(f'Mean Posterior vs Time')
        ax.grid(True, alpha=0.3)
        
        # Plot gradient vs time
        ax = axes[1, 0]
        valid_mask = ~np.isnan(mean_gradients)
        if np.sum(valid_mask) > 0:
            ax.errorbar(times[valid_mask], np.array(mean_gradients)[valid_mask], 
                       yerr=np.array(std_gradients)[valid_mask], 
                       marker='^', linestyle='-', capsize=5, alpha=0.8, linewidth=2, color='red')
            ax.set_yscale('log')
        ax.set_xlabel('Time (t_f)')
        ax.set_ylabel('Mean |Gradient|')
        ax.set_title(f'Mean Gradient Magnitude vs Time')
        ax.grid(True, alpha=0.3)
        
        # Information loss summary
        ax = axes[1, 1]
        if len(times) > 1:
            # Normalize metrics
            norm_grads = np.array(mean_gradients) / np.nanmax(mean_gradients)
            norm_posts = (np.array(mean_posteriors) - np.nanmin(mean_posteriors)) / (np.nanmax(mean_posteriors) - np.nanmin(mean_posteriors))
            
            valid_mask = ~(np.isnan(norm_grads) | np.isnan(norm_posts))
            if np.sum(valid_mask) > 0:
                ax.plot(times[valid_mask], norm_grads[valid_mask], 'r-o', label='Normalized Gradient', linewidth=2)
                ax.plot(times[valid_mask], norm_posts[valid_mask], 'b-s', label='Normalized Posterior', linewidth=2)
                ax.set_xlabel('Time (t_f)')
                ax.set_ylabel('Normalized Metric')
                ax.set_title('Information Loss Summary')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center', va='center')
        else:
            ax.text(0.5, 0.5, 'Insufficient time points', transform=ax.transAxes, ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'1d_analysis_{param_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_nan_detection_analysis(results, output_dir):
    """Create NaN detection and visualization plots."""
    print("Creating NaN detection analysis...")
    
    t_f_values = sorted(results.keys())
    
    # Count NaNs for each time point
    nan_counts = []
    inf_counts = []
    total_counts = []
    
    for t_f in t_f_values:
        if 'log_posterior_values' not in results[t_f]:
            nan_counts.append(0)
            inf_counts.append(0)
            total_counts.append(0)
            continue
        
        log_posts = results[t_f]['log_posterior_values']
        total_counts.append(len(log_posts))
        nan_counts.append(np.sum(np.isnan(log_posts)))
        inf_counts.append(np.sum(np.isinf(log_posts)))
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('NaN/Inf Detection Analysis', fontsize=16)
    
    # Plot NaN counts vs time
    ax = axes[0, 0]
    ax.plot(t_f_values, nan_counts, 'ro-', linewidth=2, markersize=6, label='NaN count')
    ax.set_xlabel('Time (t_f)')
    ax.set_ylabel('Number of NaN values')
    ax.set_title('NaN Occurrences vs Time')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot Inf counts vs time
    ax = axes[0, 1]
    ax.plot(t_f_values, inf_counts, 'bo-', linewidth=2, markersize=6, label='Inf count')
    ax.set_xlabel('Time (t_f)')
    ax.set_ylabel('Number of Inf values')
    ax.set_title('Inf Occurrences vs Time')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot success rate vs time
    ax = axes[1, 0]
    success_rates = [(total - nan - inf) / total * 100 if total > 0 else 0 
                    for total, nan, inf in zip(total_counts, nan_counts, inf_counts)]
    ax.plot(t_f_values, success_rates, 'go-', linewidth=2, markersize=6)
    ax.set_xlabel('Time (t_f)')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Evaluation Success Rate vs Time')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    # Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    total_evaluations = sum(total_counts)
    total_nans = sum(nan_counts)
    total_infs = sum(inf_counts)
    total_success = total_evaluations - total_nans - total_infs
    
    summary_text = f"""
    EVALUATION SUMMARY
    
    Total evaluations: {total_evaluations}
    Successful: {total_success} ({100*total_success/total_evaluations:.1f}%)
    NaN failures: {total_nans} ({100*total_nans/total_evaluations:.1f}%)
    Inf failures: {total_infs} ({100*total_infs/total_evaluations:.1f}%)
    
    Time range: {min(t_f_values):.2f} - {max(t_f_values):.2f}
    Time points analyzed: {len(t_f_values)}
    """
    
    ax.text(0.1, 0.9, summary_text
            
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

    # Make analysis

    return

if __name__ == "__main__":
    main()


