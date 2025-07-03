import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import yaml
import argparse
from pathlib import Path
import datetime

from model import model
from likelihood import get_log_posterior

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def calculate_blob_radius(positions, center, percentile=90):
    distances = jnp.linalg.norm(positions - center, axis=1)
    return jnp.percentile(distances, percentile)

def calculate_rms_radius(positions, center):
    distances_squared = jnp.sum((positions - center)**2, axis=1)
    return jnp.sqrt(jnp.mean(distances_squared))

def generate_parameter_perturbations(config, base_params):
    """Generate parameter perturbations around the base parameters."""
    analysis_params = config['analysis_params']
    param_bounds = analysis_params['param_bounds']
    n_points = analysis_params['n_points_per_param']
    
    parameter_sets = []
    param_names = list(param_bounds.keys())
    
    # Generate perturbations for each parameter
    for param_name, bounds in param_bounds.items():
        if param_name in config['prior_params']:
            if 'center' in param_name.lower():
                min_val = bounds['min'] if isinstance(bounds['min'], (int, float)) else bounds['min'][0]
                max_val = bounds['max'] if isinstance(bounds['max'], (int, float)) else bounds['max'][0]
                param_values = np.linspace(min_val, max_val, n_points)
                
                for val in param_values:
                    param_dict = dict(base_params)
                    param_dict[param_name] = jnp.array([val, val, val])
                    parameter_sets.append(param_dict)
            else:
                param_values = np.linspace(bounds['min'], bounds['max'], n_points)
                for val in param_values:
                    param_dict = dict(base_params)
                    param_dict[param_name] = val
                    parameter_sets.append(param_dict)
    
    return parameter_sets, param_names

def generate_reference_simulation(config):
    model_params = config['model_params'].copy()
    t_f_range = config['analysis_params']['t_f_range']
    blobs_params = model_params['blobs_params']
    G = model_params['G']
    length = model_params['length']
    softening = model_params['softening']
    dt = model_params['dt']
    
    # Generate mock data with all time points
    key = jax.random.PRNGKey(model_params.get('data_seed', 42))
    solver = model_params.get('solver', 'LeapfrogMidpoint')
    density_scaling = model_params.get('density_scaling', 'none')
    scaling_kwargs = model_params.get('scaling_kwargs', {})
    
    # Run simulation with ts parameter set to our analysis times
    mock_data = model(
        blobs_params=blobs_params,
        G=G,
        length=length,
        softening=softening,
        t_f=max(t_f_range),  # Run to maximum time
        dt=dt,
        ts=jnp.array(t_f_range),  # Save at our analysis times
        key=key,
        solver=solver,
        density_scaling=density_scaling,
        **scaling_kwargs
    )
    
    return {
        'solution': mock_data[4],
        'masses': mock_data[5] if len(mock_data) > 5 else None,
        'model_params': model_params,
        'blobs_params': blobs_params,
        't_f_range': t_f_range
    }

def extract_data_at_time_index(reference_sim, time_idx):
    """Extract density field data at a specific time index."""
    from jaxpm.painting import cic_paint
    from utils import apply_density_scaling
    
    solution = reference_sim['solution']
    masses = reference_sim['masses']
    model_params = reference_sim['model_params']
    
    # Get data directly from the solution at this time index
    actual_time = solution.ts[time_idx]
    pos_at_time = solution.ys[time_idx, 0]
    
    # Create density field
    length = model_params['length']
    grid_shape = (length, length, length)
    
    if masses is not None:
        raw_field = cic_paint(jnp.zeros(grid_shape), pos_at_time, weight=masses)
    else:
        raw_field = cic_paint(jnp.zeros(grid_shape), pos_at_time)
    
    # Apply density scaling
    density_scaling = model_params.get('density_scaling', 'none')
    scaling_kwargs = model_params.get('scaling_kwargs', {})
    observed_data = apply_density_scaling(raw_field, density_scaling, **scaling_kwargs)
    
    return observed_data, actual_time

def evaluate_informativeness_at_tf(time_idx, config, parameter_sets, param_names, reference_sim):
    """Evaluate informativeness metrics at a specific time index."""
    t_f = reference_sim['t_f_range'][time_idx]
    print(f"Evaluating informativeness at t_f = {t_f:.3f} (index {time_idx})...")
    
    # Extract observed data at this time index
    observed_data, actual_time = extract_data_at_time_index(reference_sim, time_idx)
    
    # Calculate radius evolution up to this time point
    radius_data = calculate_radius_evolution_up_to_time(reference_sim, time_idx)
    
    # Create log posterior function
    model_params = config['model_params'].copy()
    model_params['t_f'] = t_f
    
    log_posterior_fn = get_log_posterior(
        likelihood_type=config['likelihood_type'],
        data=observed_data,
        prior_params=config['prior_params'],
        prior_type=config['prior_type'],
        model_fn=model,
        init_params=reference_sim['blobs_params'],
        **config['likelihood_kwargs'],
        **{k: v for k, v in model_params.items() if k != 'blobs_params'}
    )
    
    # Evaluate gradients and likelihood for all parameter sets
    log_posterior_values = []
    gradient_values = []
    
    value_and_grad_fn = jax.value_and_grad(log_posterior_fn)
    
    print(f"  Evaluating {len(parameter_sets)} parameter combinations...")
    for i, params in enumerate(parameter_sets):
        try:
            log_post_val, grad_val = value_and_grad_fn(params)
            log_posterior_values.append(float(log_post_val))
            gradient_values.append(grad_val)
        except Exception as e:
            print(f"    Warning: Failed to evaluate parameter set {i}: {e}")
            log_posterior_values.append(np.nan)
            gradient_values.append({k: np.nan for k in param_names})
    
    log_posterior_values = np.array(log_posterior_values)
    
    # Calculate gradient norms and individual gradients
    grad_norms = []
    individual_grads = {param: [] for param in param_names}
    
    for grad_dict in gradient_values:
        if isinstance(grad_dict, dict):
            total_grad_norm = 0.0
            has_nan = False
            for key, grad_val in grad_dict.items():
                if key in param_names:
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
    
    return {
        't_f': t_f,
        'actual_time': actual_time,
        'time_idx': time_idx,
        'log_posterior_values': log_posterior_values,
        'gradient_values': gradient_values,
        'grad_norms': grad_norms,
        'individual_grads': individual_grads,
        'observed_data': observed_data,
        'radius_data': radius_data
    }

def calculate_radius_evolution_up_to_time(reference_sim, max_time_idx):
    """Calculate radius evolution up to a specific time index."""
    solution = reference_sim['solution']
    blobs_params = reference_sim['blobs_params']
    
    radius_data = {}
    
    for blob_idx, blob_params in enumerate(blobs_params):
        center = jnp.array(blob_params['pos_params']['center'])
        times = solution.ts[:max_time_idx + 1]  # Only up to the specified time
        percentile_radii = []
        rms_radii = []
        
        # Calculate particle range for this blob
        start_idx = sum(blob['n_part'] for blob in blobs_params[:blob_idx])
        end_idx = start_idx + blob_params['n_part']
        
        for t_idx in range(len(times)):
            pos_t = solution.ys[t_idx, 0]  # All positions at time t
            blob_pos = pos_t[start_idx:end_idx]  # This blob's positions
            
            perc_radius = calculate_blob_radius(blob_pos, center, percentile=90)
            rms_radius = calculate_rms_radius(blob_pos, center)
            
            percentile_radii.append(perc_radius)
            rms_radii.append(rms_radius)
        
        radius_data[f'blob{blob_idx}'] = {
            'times': times,
            'percentile_radii': jnp.array(percentile_radii),
            'rms_radii': jnp.array(rms_radii),
            'center': center
        }
    
    return radius_data

def calculate_radius_evolution(solution, blobs_params):
    """Calculate radius evolution for all blobs (full time range)."""
    radius_data = {}
    
    for blob_idx, blob_params in enumerate(blobs_params):
        center = jnp.array(blob_params['pos_params']['center'])
        times = solution.ts
        percentile_radii = []
        rms_radii = []
        
        # Calculate particle range for this blob
        start_idx = sum(blob['n_part'] for blob in blobs_params[:blob_idx])
        end_idx = start_idx + blob_params['n_part']
        
        for t_idx in range(len(times)):
            pos_t = solution.ys[t_idx, 0]  # All positions at time t
            blob_pos = pos_t[start_idx:end_idx]  # This blob's positions
            
            perc_radius = calculate_blob_radius(blob_pos, center, percentile=90)
            rms_radius = calculate_rms_radius(blob_pos, center)
            
            percentile_radii.append(perc_radius)
            rms_radii.append(rms_radius)
        
        radius_data[f'blob{blob_idx}'] = {
            'times': times,
            'percentile_radii': jnp.array(percentile_radii),
            'rms_radii': jnp.array(rms_radii),
            'center': center
        }
    
    return radius_data

def analyze_informativeness_vs_time(config):
    """Main analysis function that evaluates informativeness across different t_f values."""
    # Create timestamped output directory
    script_dir = Path(__file__).parent
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = script_dir / 'test_outputs' / 'information_loss' / f'analysis_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating information loss analysis in: {output_dir}")
    
    # Get time range to analyze
    t_f_range = config['analysis_params']['t_f_range']
    print(f"Analyzing {len(t_f_range)} time points: {t_f_range}")
    
    # Generate single reference simulation with all time points
    reference_sim = generate_reference_simulation(config)
    
    # Get base parameters (true values)
    base_params = {}
    blobs_params = config['model_params']['blobs_params']
    for blob_idx, blob in enumerate(blobs_params):
        if f'blob{blob_idx}_sigma' in config['prior_params']:
            base_params[f'blob{blob_idx}_sigma'] = blob['pos_params']['sigma']
        if f'blob{blob_idx}_center' in config['prior_params']:
            base_params[f'blob{blob_idx}_center'] = jnp.array(blob['pos_params']['center'])
    
    # Generate parameter perturbations
    parameter_sets, param_names = generate_parameter_perturbations(config, base_params)
    print(f"Generated {len(parameter_sets)} parameter perturbations for parameters: {param_names}")
    
    # Evaluate informativeness at each time index
    results = {}
    for time_idx, t_f in enumerate(t_f_range):
        results[t_f] = evaluate_informativeness_at_tf(time_idx, config, parameter_sets, param_names, reference_sim)
    
    # Add reference simulation info to results for plotting
    results['reference_sim'] = reference_sim
    
    # Create comprehensive plots
    create_informativeness_plots(results, param_names, base_params, output_dir, config)
    
    # Save comprehensive summary
    save_analysis_summary(results, param_names, output_dir, config)
    
    print(f"Information loss analysis completed!")
    print(f"Results saved to: {output_dir}")
    
    return results, output_dir

def create_informativeness_plots(results, param_names, base_params, output_dir, config):
    """Create comprehensive plots analyzing informativeness vs time."""
    
    t_f_values = sorted(results.keys())
    
    # 1. Gradient Norm Evolution vs t_f
    print("Creating gradient norm evolution plots...")
    create_gradient_norm_evolution_plot(results, param_names, t_f_values, output_dir)
    
    # 2. Parameter Sensitivity Heatmaps
    print("Creating parameter sensitivity heatmaps...")
    create_sensitivity_heatmaps(results, param_names, t_f_values, output_dir)
    
    # 3. Likelihood/Posterior Surfaces at Different t_f
    print("Creating likelihood surfaces...")
    create_likelihood_surfaces_vs_time(results, param_names, base_params, t_f_values, output_dir)
    
    # 4. Radius Evolution Comparison
    print("Creating radius evolution plots...")
    create_radius_evolution_plots(results, t_f_values, output_dir)
    
    # 5. Data Informativeness Metrics
    print("Creating informativeness metrics...")
    create_informativeness_metrics_plot(results, t_f_values, output_dir)

def create_gradient_norm_evolution_plot(results, param_names, t_f_values, output_dir):
    """Plot how gradient norms evolve with t_f for each parameter."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Gradient Norm Evolution vs Final Time', fontsize=16)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(param_names)))
    
    for i, param_name in enumerate(param_names):
        if i < 4:
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            mean_grad_norms = []
            std_grad_norms = []
            max_grad_norms = []
            
            for t_f in t_f_values:
                grads = results[t_f]['individual_grads'][param_name]
                valid_grads = grads[~np.isnan(grads)]
                
                if len(valid_grads) > 0:
                    mean_grad_norms.append(np.mean(np.abs(valid_grads)))
                    std_grad_norms.append(np.std(valid_grads))
                    max_grad_norms.append(np.max(np.abs(valid_grads)))
                else:
                    mean_grad_norms.append(np.nan)
                    std_grad_norms.append(np.nan)
                    max_grad_norms.append(np.nan)
            
            # Plot mean with error bars
            ax.errorbar(t_f_values, mean_grad_norms, yerr=std_grad_norms, 
                       marker='o', linewidth=2, markersize=6, capsize=5,
                       label='Mean ± Std', color=colors[i])
            
            # Plot maximum gradient
            ax.plot(t_f_values, max_grad_norms, '--', 
                   linewidth=2, label='Maximum', color=colors[i], alpha=0.7)
            
            ax.set_xlabel('Final Time (t_f)')
            ax.set_ylabel('Gradient Magnitude')
            ax.set_title(f'Gradient Evolution: {param_name}')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_norm_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_sensitivity_heatmaps(results, param_names, t_f_values, output_dir):
    """Create heatmaps showing parameter sensitivity across time."""
    
    # Calculate sensitivity matrix (mean absolute gradient for each param at each t_f)
    sensitivity_matrix = np.zeros((len(param_names), len(t_f_values)))
    
    for i, param_name in enumerate(param_names):
        for j, t_f in enumerate(t_f_values):
            grads = results[t_f]['individual_grads'][param_name]
            valid_grads = grads[~np.isnan(grads)]
            if len(valid_grads) > 0:
                sensitivity_matrix[i, j] = np.mean(np.abs(valid_grads))
            else:
                sensitivity_matrix[i, j] = np.nan
    
    # Create heatmap
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    im = ax.imshow(sensitivity_matrix, cmap='YlOrRd', aspect='auto', 
                   interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(range(len(t_f_values)))
    ax.set_xticklabels([f'{t_f:.2f}' for t_f in t_f_values])
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels(param_names)
    
    ax.set_xlabel('Final Time (t_f)')
    ax.set_ylabel('Parameters')
    ax.set_title('Parameter Sensitivity Heatmap\n(Mean Absolute Gradient)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean |Gradient|')
    
    # Add text annotations
    for i in range(len(param_names)):
        for j in range(len(t_f_values)):
            if not np.isnan(sensitivity_matrix[i, j]):
                text = ax.text(j, i, f'{sensitivity_matrix[i, j]:.1e}',
                             ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sensitivity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_likelihood_surfaces_vs_time(results, param_names, base_params, t_f_values, output_dir):
    """Create likelihood surfaces for parameter pairs at different t_f values."""
    
    # Focus on first two parameters for surface plots
    if len(param_names) >= 2:
        param1, param2 = param_names[0], param_names[1]
        
        # Select a subset of t_f values for visualization
        selected_tf = t_f_values[::max(1, len(t_f_values)//4)]  # Select ~4 time points
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Likelihood Surfaces: {param1} vs {param2}', fontsize=16)
        
        for i, t_f in enumerate(selected_tf[:4]):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            # Get parameter values and likelihood
            param1_vals = []
            param2_vals = []
            likelihood_vals = []
            
            # Extract parameter values from parameter sets
            for j, params in enumerate(results[t_f]['log_posterior_values']):
                if not np.isnan(params):
                    # Need to reconstruct parameter values - this is a simplification
                    # In practice, you'd need to store the parameter sets used
                    param1_vals.append(j)  # Placeholder
                    param2_vals.append(j)  # Placeholder
                    likelihood_vals.append(params)
            
            if len(likelihood_vals) > 0:
                # Create a simple scatter plot instead of surface for now
                scatter = ax.scatter(range(len(likelihood_vals)), likelihood_vals, 
                                   c=likelihood_vals, cmap='viridis', alpha=0.6)
                ax.set_xlabel('Parameter Index')
                ax.set_ylabel('Log Posterior')
                ax.set_title(f't_f = {t_f:.2f}')
                plt.colorbar(scatter, ax=ax, label='Log Posterior')
            else:
                ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes, 
                       ha='center', va='center')
                ax.set_title(f't_f = {t_f:.2f}')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'likelihood_surfaces_{param1}_vs_{param2}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

def create_radius_evolution_plots(results, t_f_values, output_dir):
    """Create plots showing radius evolution for different final times."""
    
    # Get radius data from the reference simulation (full evolution)
    reference_sim = results['reference_sim']
    full_radius_data = calculate_radius_evolution(reference_sim['solution'], reference_sim['blobs_params'])
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(full_radius_data)))
    
    for i, (blob_name, data) in enumerate(full_radius_data.items()):
        times = data['times']
        perc_radii = data['percentile_radii']
        rms_radii = data['rms_radii']
        
        # Plot 90th percentile radius
        axes[0].plot(times, perc_radii, 'o-', color=colors[i], 
                    label=f'{blob_name}', linewidth=2, markersize=2)
        
        # Plot RMS radius
        axes[1].plot(times, rms_radii, 'o-', color=colors[i], 
                    label=f'{blob_name}', linewidth=2, markersize=2)
    
    # Mark analysis time points
    for t_f in t_f_values:
        axes[0].axvline(t_f, color='red', linestyle='--', alpha=0.7, linewidth=1)
        axes[1].axvline(t_f, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('90th Percentile Radius')
    axes[0].set_title('Blob Radius Evolution (90th Percentile)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('RMS Radius')
    axes[1].set_title('Blob Radius Evolution (RMS)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add legend for the analysis time points
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', alpha=0.7, label='Analysis times')
    ]
    axes[0].legend(handles=legend_elements, loc='upper right')
    
    plt.suptitle('Blob Collapse Over Time\n(Red lines show analysis time points)', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'radius_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_informativeness_metrics_plot(results, t_f_values, output_dir):
    """Create plots of various informativeness metrics vs time."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Data Informativeness Metrics vs Final Time', fontsize=16)
    
    # Calculate metrics for each t_f
    mean_grad_norms = []
    max_grad_norms = []
    log_post_ranges = []
    data_variances = []
    
    for t_f in t_f_values:
        # Overall gradient norm metrics
        all_grad_norms = results[t_f]['grad_norms']
        valid_grads = all_grad_norms[~np.isnan(all_grad_norms)]
        
        if len(valid_grads) > 0:
            mean_grad_norms.append(np.mean(valid_grads))
            max_grad_norms.append(np.max(valid_grads))
        else:
            mean_grad_norms.append(np.nan)
            max_grad_norms.append(np.nan)
        
        # Posterior range (measure of landscape curvature)
        log_posts = results[t_f]['log_posterior_values']
        valid_posts = log_posts[~np.isnan(log_posts)]
        
        if len(valid_posts) > 1:
            log_post_ranges.append(np.max(valid_posts) - np.min(valid_posts))
        else:
            log_post_ranges.append(np.nan)
        
        # Data variance (measure of structural information)
        data = results[t_f]['observed_data']
        data_variances.append(np.var(data))
    
    # Plot 1: Gradient norm metrics
    axes[0, 0].semilogy(t_f_values, mean_grad_norms, 'o-', label='Mean', linewidth=2)
    axes[0, 0].semilogy(t_f_values, max_grad_norms, 's-', label='Maximum', linewidth=2)
    axes[0, 0].set_xlabel('Final Time (t_f)')
    axes[0, 0].set_ylabel('Gradient Norm')
    axes[0, 0].set_title('Gradient Magnitude vs Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Posterior landscape curvature
    axes[0, 1].plot(t_f_values, log_post_ranges, 'o-', color='orange', linewidth=2)
    axes[0, 1].set_xlabel('Final Time (t_f)')
    axes[0, 1].set_ylabel('Log Posterior Range')
    axes[0, 1].set_title('Posterior Landscape Curvature')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Data structural information
    axes[1, 0].plot(t_f_values, data_variances, 'o-', color='green', linewidth=2)
    axes[1, 0].set_xlabel('Final Time (t_f)')
    axes[1, 0].set_ylabel('Data Variance')
    axes[1, 0].set_title('Density Field Structural Information')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Combined informativeness score
    # Normalize metrics and combine them
    norm_grads = np.array(mean_grad_norms) / np.nanmax(mean_grad_norms)
    norm_ranges = np.array(log_post_ranges) / np.nanmax(log_post_ranges) 
    norm_vars = np.array(data_variances) / np.nanmax(data_variances)
    
    # Combined score (higher = more informative)
    informativeness_score = norm_grads * norm_ranges * norm_vars
    
    axes[1, 1].plot(t_f_values, informativeness_score, 'o-', color='red', linewidth=2)
    axes[1, 1].set_xlabel('Final Time (t_f)')
    axes[1, 1].set_ylabel('Informativeness Score')
    axes[1, 1].set_title('Combined Informativeness Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'informativeness_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_analysis_summary(results, param_names, output_dir, config):
    """Save comprehensive analysis summary."""
    
    summary_file = output_dir / 'information_loss_summary.txt'
    
    with open(summary_file, 'w') as f:
        f.write("=== INFORMATION LOSS ANALYSIS SUMMARY ===\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H%M%S')}\n")
        f.write(f"Output directory: {output_dir}\n\n")
        
        # Analysis configuration
        f.write("=== ANALYSIS CONFIGURATION ===\n")
        f.write(f"Parameters analyzed: {param_names}\n")
        f.write(f"Time points analyzed: {config['analysis_params']['t_f_range']}\n")
        f.write(f"Number of parameter perturbations: {config['analysis_params']['n_points_per_param']}\n")
        f.write(f"Likelihood type: {config['likelihood_type']}\n")
        f.write(f"Prior type: {config['prior_type']}\n\n")
        
        # Results for each time point
        f.write("=== RESULTS BY TIME POINT ===\n")
        for t_f in sorted(k for k in results.keys() if isinstance(k, (int, float))):
            actual_time = results[t_f]['actual_time']
            f.write(f"\nTime t_f = {t_f:.3f} (actual = {actual_time:.3f}):\n")
            
            # Gradient statistics
            grad_norms = results[t_f]['grad_norms']
            valid_grads = grad_norms[~np.isnan(grad_norms)]
            
            if len(valid_grads) > 0:
                f.write(f"  Gradient norm range: [{np.min(valid_grads):.3e}, {np.max(valid_grads):.3e}]\n")
                f.write(f"  Mean gradient norm: {np.mean(valid_grads):.3e}\n")
                f.write(f"  Median gradient norm: {np.median(valid_grads):.3e}\n")
            
            # Posterior statistics
            log_posts = results[t_f]['log_posterior_values']
            valid_posts = log_posts[~np.isnan(log_posts)]
            
            if len(valid_posts) > 0:
                f.write(f"  Posterior range: [{np.min(valid_posts):.3f}, {np.max(valid_posts):.3f}]\n")
                f.write(f"  Posterior span: {np.max(valid_posts) - np.min(valid_posts):.3f}\n")
            
            # Data characteristics
            data = results[t_f]['observed_data']
            f.write(f"  Data variance: {np.var(data):.3e}\n")
            f.write(f"  Data mean: {np.mean(data):.3e}\n")
            
            # Parameter-specific gradients
            f.write(f"  Parameter-specific gradient norms:\n")
            for param_name in param_names:
                grads = results[t_f]['individual_grads'][param_name]
                valid_param_grads = grads[~np.isnan(grads)]
                if len(valid_param_grads) > 0:
                    f.write(f"    {param_name}: mean={np.mean(np.abs(valid_param_grads)):.3e}, "
                           f"max={np.max(np.abs(valid_param_grads)):.3e}\n")
        
        # Key insights
        f.write("\n=== KEY INSIGHTS ===\n")
        
        # Find time of maximum informativeness
        t_f_list = sorted(k for k in results.keys() if isinstance(k, (int, float)))
        max_info_tf = None
        max_info_score = -np.inf
        
        for t_f in t_f_list:
            grad_norms = results[t_f]['grad_norms']
            valid_grads = grad_norms[~np.isnan(grad_norms)]
            if len(valid_grads) > 0:
                score = np.mean(valid_grads)
                if score > max_info_score:
                    max_info_score = score
                    max_info_tf = t_f
        
        if max_info_tf is not None:
            f.write(f"Time of maximum informativeness: t_f = {max_info_tf:.3f}\n")
        
        # Analyze trend
        grad_means = []
        for t_f in t_f_list:
            grad_norms = results[t_f]['grad_norms']
            valid_grads = grad_norms[~np.isnan(grad_norms)]
            if len(valid_grads) > 0:
                grad_means.append(np.mean(valid_grads))
            else:
                grad_means.append(np.nan)
        
        if len(grad_means) > 1:
            if grad_means[-1] < grad_means[0]:
                f.write("Overall trend: Informativeness DECREASES with time (as expected)\n")
            else:
                f.write("Overall trend: Informativeness INCREASES with time (unexpected)\n")
        
        # Efficiency note
        f.write(f"\nEFFICIENCY NOTE:\n")
        f.write(f"Single simulation run with ts parameter for all {len(t_f_list)} time points\n")
        f.write(f"Much more efficient than running {len(t_f_list)} separate simulations!\n")
        
        f.write("\n=== FILES GENERATED ===\n")
        f.write("- gradient_norm_evolution.png\n")
        f.write("- sensitivity_heatmap.png\n")
        f.write("- likelihood_surfaces_*.png\n")
        f.write("- radius_evolution.png\n")
        f.write("- informativeness_metrics.png\n")
        f.write("- information_loss_summary.txt\n")
    
    print(f"Summary saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Information loss analysis for N-body parameter inference')
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
    
    # Run analysis
    results, output_dir = analyze_informativeness_vs_time(config)
    
    print("Information loss analysis completed!")

if __name__ == "__main__":
    main()
