import os
import argparse
import numpy as np
import yaml
import matplotlib.pyplot as plt
import corner
from datetime import datetime

def load_samples_from_directory(directory):
    samples_path = os.path.join(directory, "samples.npz")
    config_path = os.path.join(directory, "config.yaml")
    truth_path = os.path.join(directory, "truth.npz")
    
    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"samples.npz not found in {directory}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml not found in {directory}")

    if not os.path.exists(truth_path):
        raise FileNotFoundError(f"truth.npz not found in {directory}")
    # Load samples
    samples_data = np.load(samples_path)
    samples = {key: samples_data[key] for key in samples_data.keys()}
    # Load truth parameters
    truth_data = np.load(truth_path)
    true_params = {key: truth_data[key] for key in truth_data.keys()}
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    directory_name = os.path.basename(directory)
    return {
        'samples': samples,
        'true_params': true_params,
        'config': config,
        'directory_name': directory_name
    }

def check_same_truth_parameters(true_params_list):
    if not true_params_list:
        return True
    first_truth = true_params_list[0]
    for truth in true_params_list[1:]:
        for key in first_truth:
            if not np.array_equal(first_truth[key], truth.get(key)):
                return False
    print("All truth parameters match across directories.")
    return True

def convert_to_chain_format(samples_list, param_order):
    if not samples_list:
        return {}, {}
    
    # Initialize chain samples dictionary
    chain_samples = {}
    flattened_samples = {}
    
    for param_name in param_order:
        chain_data = []
        flattened_data = []
        
        for samples in samples_list:
            if param_name in samples:
                param_samples = samples[param_name]
                chain_data.append(param_samples)
                flattened_data.append(param_samples)
        
        if chain_data:
            # Stack chains along first axis
            chain_samples[param_name] = np.stack(chain_data, axis=0)
            # Concatenate for flattened version
            flattened_samples[param_name] = np.concatenate(flattened_data, axis=0)
    
    return chain_samples, flattened_samples

def compute_rhat(chain_samples, param_name, n_samples):
    """
    Compute R-hat for a parameter across multiple chains up to n_samples.
    
    Parameters:
    -----------
    chain_samples : dict
        Dictionary with chain-separated samples
    param_name : str
        Parameter name to compute R-hat for
    n_samples : int
        Number of samples to use from each chain
    
    Returns:
    --------
    float : R-hat value
    """
    if param_name not in chain_samples:
        return None
    
    chains = chain_samples[param_name][:, :n_samples]  # (n_chains, n_samples) or (n_chains, n_samples, n_dims)

    n_chains, n_samples_actual = chains.shape
    if n_chains < 2 or n_samples_actual < 2:
        return None
    
    # Compute chain means and overall mean
    chain_means = np.mean(chains, axis=1)
    overall_mean = np.mean(chain_means)
    
    # Between-chain variance
    B = n_samples_actual * np.var(chain_means, ddof=1)
    
    # Within-chain variance
    chain_vars = np.var(chains, axis=1, ddof=1)
    W = np.mean(chain_vars)
    
    # R-hat calculation
    if W == 0:
        return 1.0
    
    var_hat = ((n_samples_actual - 1) * W + B) / n_samples_actual
    rhat = np.sqrt(var_hat / W)
    
    return rhat

def compute_ess(samples):
    """
    Compute effective sample size for a single chain using autocorrelation.
    
    Parameters:
    -----------
    samples : array
        1D array of samples
    
    Returns:
    --------
    float : ESS value
    """
    n = len(samples)
    if n < 4:
        return n
    
    # Center the samples
    samples_centered = samples - np.mean(samples)
    
    # Compute autocorrelation using FFT
    f_samples = np.fft.fft(samples_centered, n=2*n)
    autocorr = np.fft.ifft(f_samples * np.conj(f_samples)).real
    autocorr = autocorr[:n] / autocorr[0]  # Normalize
    
    # Find first negative autocorrelation or use cutoff
    cutoff = min(n//4, 100)  # Reasonable cutoff
    first_negative = np.where(autocorr < 0)[0]
    if len(first_negative) > 0:
        cutoff = min(cutoff, first_negative[0])
    
    # Compute integrated autocorrelation time
    tau_int = 1 + 2 * np.sum(autocorr[1:cutoff])
    tau_int = max(tau_int, 1.0)  # Ensure at least 1
    
    # ESS = N / (2 * tau_int)
    ess = n / (2 * tau_int)
    return max(ess, 1.0)

def plot_rhat_analysis(chain_samples, param_order, n_samples_interval, output_dir):
    """
    Plot R-hat convergence analysis for multiple chains.
    """
    if len(next(iter(chain_samples.values()))) < 2:
        print("R-hat analysis requires at least 2 chains. Skipping...")
        return None
    
    max_samples = next(iter(chain_samples.values())).shape[1]
    sample_points = np.arange(n_samples_interval, max_samples + 1, n_samples_interval)
    
    fig, axes = plt.subplots(len(param_order), 1, figsize=(10, 3 * len(param_order)), squeeze=False)
    axes = axes.flatten()
    
    for idx, param_name in enumerate(param_order):
        ax = axes[idx]
        rhats = []
        
        for n_samples in sample_points:
            rhat = compute_rhat(chain_samples, param_name, n_samples)
            rhats.append(rhat)
        
        # Plot R-hat evolution
        ax.plot(sample_points, rhats, 'b-', linewidth=2, label='R-hat')
        
        
        
        ax.set_xlabel('Number of samples')
        ax.set_ylabel('R-hat')
        ax.set_title(f'R-hat convergence: {param_name}')
        ax.grid(True, alpha=0.3)
        if min(rhats) > 2.0:
            ax.set_ylim(min(rhats) if rhats else 0.90, max(rhats) if rhats else 2.0)
        else:
            ax.set_ylim(0.95, min(2.0, max(rhats) * 1.1) if rhats else 2.0)
            # Add reference lines
            ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect convergence')
            ax.axhline(y=1.01, color='orange', linestyle='--', alpha=0.7, label='Excellent (< 1.01)')
            ax.axhline(y=1.1, color='red', linestyle='--', alpha=0.7, label='Poor (> 1.1)')
        ax.legend()
        
    plt.suptitle('R-hat Convergence Analysis', fontsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "rhat_analysis.png")
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"R-hat analysis saved to: {save_path}")
    
    return fig

def plot_ess_analysis(chain_samples, param_order, n_samples_interval, output_dir):
    """
    Plot ESS analysis for each chain.
    """
    max_samples = next(iter(chain_samples.values())).shape[1]
    sample_points = np.arange(n_samples_interval, max_samples + 1, n_samples_interval)
    n_chains = len(next(iter(chain_samples.values())))
    
    fig, axes = plt.subplots(len(param_order), 1, figsize=(10, 3 * len(param_order)), squeeze=False)
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_chains))
    
    for idx, param_name in enumerate(param_order):
        ax = axes[idx]
        
        if param_name not in chain_samples:
            continue
            
        chains = chain_samples[param_name]
        
        for chain_idx in range(n_chains):
            ess_values = []
            
            for n_samples in sample_points:
                # Get samples for this chain up to n_samples
                if chains.ndim == 3:  # Vector parameter - use first component
                    chain_data = chains[chain_idx, :n_samples, 0]
                else:  # Scalar parameter
                    chain_data = chains[chain_idx, :n_samples]
                
                ess = compute_ess(chain_data)
                ess_values.append(ess)
            
            ax.plot(sample_points, ess_values, color=colors[chain_idx], 
                   linewidth=2, label=f'Chain {chain_idx + 1}')
        
        # Add reference line (ideal ESS = number of samples)
        ax.plot(sample_points, sample_points, 'k--', alpha=0.5, label='Ideal (ESS = N)')
        
        ax.set_xlabel('Number of samples')
        ax.set_ylabel('Effective Sample Size')
        ax.set_title(f'ESS Analysis: {param_name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_yscale('log')  # Set y-axis to log scale
    
    plt.suptitle('Effective Sample Size Analysis', fontsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "ess_analysis.png")
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"ESS analysis saved to: {save_path}")
    
    return fig

def plot_multiple_chains(directories, output_dir, burnin=0):
    # Load data from all directories
    all_data = []
    chain_labels = []
    
    print(f"Loading samples from {len(directories)} directories...")
    
    for i, directory in enumerate(directories):
        try:
            data = load_samples_from_directory(directory)
            all_data.append(data)
            chain_labels.append(f"Chain {i+1} ({data['directory_name']})")
            print(f"  Loaded: {directory}")
        except Exception as e:
            print(f"  Error loading {directory}: {e}")
            continue
    
    if not all_data:
        print("No valid sample data found!")
        return
    
    # Extract samples and configs
    samples_list = [data['samples'] for data in all_data]
    configs = [data['config'] for data in all_data]
    true_params_list = [data['true_params'] for data in all_data]

    # Check if all truth parameters are the same
    if not check_same_truth_parameters(true_params_list):
        raise ValueError("Truth parameters do not match across all directories!")
    
    # Use first config for simulation parameters (assuming they're the same)
    reference_config = configs[0]
    model_params = reference_config.get('model_params', {})
    G = model_params.get('G')
    t_f = model_params.get('t_f')
    dt = model_params.get('dt')
    length = model_params.get('length')
    softening = model_params.get('softening')
    blobs_params = model_params.get('blobs_params', [])
    n_part = sum(blob['n_part'] for blob in blobs_params)
    burnin = configs[0].get('warmup', burnin)  # Use burnin from first config if available
    
    # Extract true parameters
    theta = true_params_list[0] 
    
    # Get parameter order
    param_order = list(theta.keys())
    
    # Convert to chain format
    chain_samples, flattened_samples = convert_to_chain_format(samples_list, param_order)
    
    # Determine method from config
    method = reference_config.get('sampler')

    # Output directory
    if output_dir is None:
        output_dir = "."
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    subfolder_name = f"analysis_{method}_{current_date}"
    output_dir = os.path.join(output_dir, subfolder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nPlotting parameters: {param_order}")
    print(f"Number of chains: {len(chain_labels)}")
    print(f"Burn-in samples: {burnin}")
    
    # Create trace plots
    print("\nGenerating trace plots...")
    trace_save_path = os.path.join(output_dir, "trace_plots.png")
    
    fig_trace, axes_trace = plot_trace_subplots(
        mcmc_samples=flattened_samples,
        theta=theta,
        G=G,
        t_f=t_f,
        dt=dt,
        softening=softening,
        length=length,
        n_part=n_part,
        method=method,
        param_order=param_order,
        save_path=trace_save_path,
        chain_samples=chain_samples
    )
    
    if fig_trace is not None:
        print(f"Trace plots saved to: {trace_save_path}")
        plt.close(fig_trace)
    
    # Create corner plot
    print("\nGenerating corner plot...")
    corner_save_path = os.path.join(output_dir, "corner_plot.png")
    
    fig_corner = plot_corner_after_burnin(
        mcmc_samples=flattened_samples,
        theta=theta,
        G=G,
        t_f=t_f,
        dt=dt,
        softening=softening,
        length=length,
        n_part=n_part,
        method=method,
        param_order=param_order,
        burnin=burnin,
        save_path=corner_save_path,
        chain_samples=chain_samples
    )
    
    if fig_corner is not None:
        print(f"Corner plot saved to: {corner_save_path}")
        plt.close(fig_corner)
    
    # Get analysis parameters from config
    analysis_params = reference_config.get('analysis', {})
    n_samples_interval = analysis_params.get('n_samples_interval', 100)
    
    # Add R-hat analysis
    if len(chain_labels) >= 2:
        print("\nGenerating R-hat analysis...")
        fig_rhat = plot_rhat_analysis(chain_samples, param_order, n_samples_interval, output_dir)
        if fig_rhat is not None:
            plt.close(fig_rhat)
    
    # Add ESS analysis
    print("\nGenerating ESS analysis...")
    fig_ess = plot_ess_analysis(chain_samples, param_order, n_samples_interval, output_dir)
    if fig_ess is not None:
        plt.close(fig_ess)
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Total chains: {len(chain_labels)}")
    for i, label in enumerate(chain_labels):
        first_param = next(iter(samples_list[i].values()))
        n_samples = len(first_param)
        print(f"  {label}: {n_samples} samples")
    
    if flattened_samples:
        first_param = next(iter(flattened_samples.values()))
        total_samples = len(first_param)
        print(f"Total samples (all chains): {total_samples}")
        print(f"Samples after burn-in: {total_samples - burnin * len(chain_labels)}")

def main():
    """
    Main function to run corner plots analysis for multiple MCMC chains.
    
    This script processes multiple directories containing MCMC samples and generates:
    1. Trace plots showing parameter evolution over iterations
    2. Corner plots showing posterior distributions and correlations
    3. R-hat convergence analysis
    4. Effective Sample Size (ESS) analysis
    
    Expected directory structure for each input directory:
    - samples.npz: Contains MCMC samples for all parameters
    - config.yaml: Contains simulation configuration and method info
    - truth.npz: Contains true parameter values for comparison
    
    Usage examples:
    
    1. Basic usage with config file:
       python corner_plots.py analysis_config.yaml
    
    2. With custom output directory and burnin:
       python corner_plots.py analysis_config.yaml -o results --burnin 1000
    """
    parser = argparse.ArgumentParser(description="Generate trace and corner plots from multiple MCMC chains")
    parser.add_argument("config_file", help="YAML config file containing directories to analyze")
    parser.add_argument("--output-dir", "-o", default="corner_plots_output", 
                       help="Output directory for plots (default: corner_plots_output)")
    parser.add_argument("--burnin", "-b", type=int, default=0, 
                       help="Number of burn-in samples to discard (default: 0)")
    
    args = parser.parse_args()
    
    # Load config file
    if not os.path.exists(args.config_file):
        print(f"Error: Config file {args.config_file} does not exist!")
        return
    
    try:
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return
    
    # Extract directories from config
    directories = config.get('directories', [])
    if not directories:
        print("Error: No directories specified in config file!")
        return
    
    # Override output_dir and burnin from config if specified
    output_dir = config.get('output_dir', args.output_dir)
    burnin = config.get('burnin', args.burnin)
    
    # Validate directories
    valid_directories = []
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist, skipping...")
            continue
        if not os.path.exists(os.path.join(directory, "samples.npz")):
            print(f"Warning: samples.npz not found in {directory}, skipping...")
            continue
        valid_directories.append(directory)
    
    if not valid_directories:
        print("Error: No valid directories with samples.npz found!")
        return
    
    print(f"Processing {len(valid_directories)} directories...")
    
    # Generate plots
    plot_multiple_chains(
        directories=valid_directories,
        output_dir=output_dir,
        burnin=burnin
    )
    
    print(f"\nAnalysis complete! Check {output_dir} for results.")

### Plotting functions for sampling experiments

def plot_trace_subplots(mcmc_samples, theta, G, t_f, dt, softening, length, n_part, method, param_order, save_path=None, chain_samples=None):
    """
    Plot trace plots for MCMC samples with blob parameters.
    Vector parameters are plotted with separate subplots for each component.
    Scalar parameters each get their own subplot.
    
    If chain_samples is provided, plots individual chains separately.
    Otherwise, plots the flattened samples.

    Parameters:
    -----------
    mcmc_samples : dict
        Dictionary with parameter names as keys and sample arrays as values (flattened)
    theta : dict
        Dictionary with true parameter values
    param_order : tuple
        Tuple of parameter names to plot
    chain_samples : dict, optional
        Dictionary with chain-separated samples (n_chains, n_samples, ...)
    """
    title = 'Sampling of the model parameters with ' + method
    param_info = f'G={G}, tf={t_f}, dt={dt}, L={length}, N={n_part}, softening={softening}'

    # Determine total number of subplots needed (scalar + vector components)
    subplot_info = []
    for param_name in param_order:
        if param_name in mcmc_samples:
            samples = mcmc_samples[param_name]
            if samples.ndim == 2:
                n_components = samples.shape[1]
                for j in range(n_components):
                    subplot_info.append((param_name, j))
            else:
                subplot_info.append((param_name, None))

    n_subplots = len(subplot_info)
    if n_subplots == 0:
        print("No parameters to plot")
        return None, None

    # Create figure with 2 columns: main trace plots and zoom plots
    fig = plt.figure(figsize=(20, 3 * n_subplots))
    
    # Add chain information to title if available
    if chain_samples is not None:
        n_chains = len(next(iter(chain_samples.values())))
        title += f' ({n_chains} chains)'
    
    plt.suptitle(f"{title}\n{param_info}", fontsize=16, y=0.98)
    
    # Adjust subplot spacing - increase top margin significantly
    plt.subplots_adjust(top=0.85, bottom=0.1, left=0.08, right=0.95, hspace=0.4, wspace=0.3)

    # Use a better color palette for multiple chains
    if chain_samples is not None:
        n_chains = len(next(iter(chain_samples.values())))
        if n_chains <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, n_chains))
        else:
            colors = plt.cm.rainbow(np.linspace(0, 1, n_chains))
    
    for idx, (param_name, comp_idx) in enumerate(subplot_info):
        # Main trace plot (left column)
        ax_main = plt.subplot(n_subplots, 2, 2*idx + 1)
        
        # Zoom plot (right column)
        ax_zoom = plt.subplot(n_subplots, 2, 2*idx + 2)
        
        true_val = None
        all_samples = []
        
        if chain_samples is not None:
            # Plot individual chains
            chain_data = chain_samples[param_name]
            n_chains = chain_data.shape[0]
            
            if comp_idx is not None:
                # Vector component
                for chain_idx in range(n_chains):
                    chain_samples_comp = chain_data[chain_idx, :, comp_idx]
                    all_samples.extend(chain_samples_comp)
                    ax_main.plot(chain_samples_comp, color=colors[chain_idx], 
                               label=f'Chain {chain_idx+1}', alpha=0.8, linewidth=1)
                    ax_zoom.plot(chain_samples_comp, color=colors[chain_idx], 
                               alpha=0.8, linewidth=1)
                
                # Add true value line if available
                if param_name in theta:
                    true_val_array = theta[param_name]
                    if isinstance(true_val_array, (list, tuple, np.ndarray)):
                        true_val_array = np.array(true_val_array)
                        if comp_idx < len(true_val_array):
                            true_val = true_val_array[comp_idx]
                            ax_main.axhline(true_val, color='red', linestyle='--', linewidth=2,
                                         label=f'True value = {true_val:.3f}')
                            ax_zoom.axhline(true_val, color='red', linestyle='--', linewidth=2)
                ax_main.set_ylabel(f'{param_name}[{comp_idx}]')
                ax_main.set_title(f'{param_name}[{comp_idx}] - Full Trace')
                ax_zoom.set_title(f'{param_name}[{comp_idx}] - Zoom around True Value')
            else:
                # Scalar parameter
                for chain_idx in range(n_chains):
                    chain_samples_scalar = chain_data[chain_idx, :]
                    all_samples.extend(chain_samples_scalar)
                    ax_main.plot(chain_samples_scalar, color=colors[chain_idx], 
                               label=f'Chain {chain_idx+1}', alpha=0.8, linewidth=1)
                    ax_zoom.plot(chain_samples_scalar, color=colors[chain_idx], 
                               alpha=0.8, linewidth=1)
                
                if param_name in theta:
                    true_val = theta[param_name]
                    if not isinstance(true_val, (list, tuple, np.ndarray)):
                        ax_main.axhline(true_val, color='red', linestyle='--', linewidth=2,
                                     label=f'True value = {true_val:.3f}')
                        ax_zoom.axhline(true_val, color='red', linestyle='--', linewidth=2)
                ax_main.set_ylabel(param_name)
                ax_main.set_title(f'{param_name} - Full Trace')
                ax_zoom.set_title(f'{param_name} - Zoom around True Value')
            
            # Only show legend for first few subplots to avoid clutter
            if idx < 3:
                ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        else:
            # Plot flattened samples (original behavior)
            samples = mcmc_samples[param_name]
            if comp_idx is not None:
                # Vector component
                samples_comp = samples[:, comp_idx]
                all_samples = samples_comp
                ax_main.plot(samples_comp, color=plt.cm.tab10(comp_idx), 
                           label=f'{param_name}[{comp_idx}]', alpha=0.8)
                ax_zoom.plot(samples_comp, color=plt.cm.tab10(comp_idx), alpha=0.8)
                # Add true value line if available
                if param_name in theta:
                    true_val_array = theta[param_name]
                    if isinstance(true_val_array, (list, tuple, np.ndarray)):
                        true_val_array = np.array(true_val_array)
                        if comp_idx < len(true_val_array):
                            true_val = true_val_array[comp_idx]
                            ax_main.axhline(true_val, color='red', linestyle='--',
                                         label=f'True value = {true_val:.3f}')
                            ax_zoom.axhline(true_val, color='red', linestyle='--')
                ax_main.set_ylabel(f'{param_name}[{comp_idx}]')
                ax_main.set_title(f'{param_name}[{comp_idx}] - Full Trace')
                ax_zoom.set_title(f'{param_name}[{comp_idx}] - Zoom around True Value')
                ax_main.legend()
            else:
                # Scalar parameter
                all_samples = samples
                ax_main.plot(samples, label=param_name)
                ax_zoom.plot(samples)
                if param_name in theta:
                    true_val = theta[param_name]
                    if not isinstance(true_val, (list, tuple, np.ndarray)):
                        ax_main.axhline(true_val, color='red', linestyle='--', 
                                     label=f'True value = {true_val:.3f}')
                        ax_zoom.axhline(true_val, color='red', linestyle='--')
                ax_main.set_ylabel(param_name)
                ax_main.set_title(f'{param_name} - Full Trace')
                ax_zoom.set_title(f'{param_name} - Zoom around True Value')
                ax_main.legend()
        print(true_val)
        # Set zoom limits around true value if available
        if true_val is not None and all_samples:
            all_samples = np.array(all_samples)
            # Zoom range of ±10% of the true value
            zoom_range = abs(true_val) * 0.05
            zoom_min = true_val - zoom_range
            zoom_max = true_val + zoom_range
            
            ax_zoom.set_ylim(zoom_min, zoom_max)
        
        ax_main.grid(True, alpha=0.3)
        ax_zoom.grid(True, alpha=0.3)
        ax_zoom.set_ylabel('Zoomed view')

    # Set x-label only for bottom plots
    ax_main.set_xlabel('Sample')
    ax_zoom.set_xlabel('Sample')
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig, None

def plot_corner_after_burnin(mcmc_samples, theta, G, t_f, dt, softening, length, n_part, method, param_order, burnin=0, save_path=None, chain_samples=None):
    """
    Plot corner plot for MCMC samples with blob parameters.
    Uses all samples (flattened across chains) for the corner plot.
    
    Parameters:
    -----------
    mcmc_samples : dict
        Dictionary with flattened samples across all chains
    chain_samples : dict, optional
        Dictionary with chain-separated samples (not used for corner plot, but affects title)
    """    
    # Prepare data for corner plot
    samples_list = []
    labels = []
    truths = []
    
    title = "Posterior distribution of initial distribution's parameters with " + method.upper()
    param_info = f'G={G}, tf={t_f}, dt={dt}, L={length}, N={n_part}, softening={softening}'
    
    # Add chain information to title if available
    if chain_samples is not None:
        n_chains = len(next(iter(chain_samples.values())))
        n_samples_per_chain = next(iter(chain_samples.values())).shape[1]
        title += f' ({n_chains} chains, {n_samples_per_chain} samples each)'
    
    title = f"{title}\n{param_info}"
    
    for param_name in param_order:
        if param_name in mcmc_samples:
            param_samples = mcmc_samples[param_name][burnin:]
            
            # Handle vector parameters
            if param_samples.ndim > 1 and param_samples.shape[1] > 1:
                # Treat each coordinate as a separate parameter
                for coord_idx in range(param_samples.shape[1]):
                    samples_list.append(param_samples[:, coord_idx])
                    labels.append(f"{param_name}[{coord_idx}]")
                    
                    if param_name in theta:
                        true_val = np.array(theta[param_name])
                        # Fix: Extract scalar value, not array
                        truths.append(float(true_val[coord_idx]) if true_val is not None and coord_idx < len(true_val) else None)
                    else:
                        truths.append(None)
            else:
                samples_list.append(param_samples.flatten())
                labels.append(param_name)
                
                if param_name in theta:
                    # Fix: Ensure scalar value for non-vector parameters
                    true_val = theta[param_name]
                    truths.append(float(true_val) if not isinstance(true_val, (list, tuple, np.ndarray)) else None)
                else:
                    truths.append(None)
    
    if not samples_list:
        print("No valid samples for corner plot")
        return None
    
    samples_array = np.column_stack(samples_list)
    
    # Create corner plot with better styling for multiple chains
    fig = corner.corner(
        samples_array,
        labels=labels,
        truths=truths,
        show_titles=True,
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 12},
        quantiles=[0.16, 0.5, 0.84],
        levels=(0.68, 0.95),
        plot_contours=True,
        fill_contours=True,
        bins=40,
        smooth=1.0,
        color='blue',
        truth_color='red',
        hist_kwargs={'density': True, 'alpha': 0.7}
    )
    
    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(top=0.92)  # Adjust for the suptitle

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    
    return fig

if __name__ == "__main__":
    main()



