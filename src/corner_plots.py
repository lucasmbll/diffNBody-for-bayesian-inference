import os
import argparse
import numpy as np
import yaml
import matplotlib.pyplot as plt
import corner

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

def plot_multiple_chains(directories, output_dir, burnin=0):
    if output_dir is None:
        output_dir = "."
    
    os.makedirs(output_dir, exist_ok=True)
    
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

    fig, axes = plt.subplots(n_subplots, 1, figsize=(12, 3 * n_subplots), squeeze=False)
    axes = axes.flatten()
    
    # Add chain information to title if available
    if chain_samples is not None:
        n_chains = len(next(iter(chain_samples.values())))
        title += f' ({n_chains} chains)'
    
    plt.suptitle(f"{title}\n{param_info}", fontsize=14)
    plt.subplots_adjust(top=0.92)

    # Use a better color palette for multiple chains
    if chain_samples is not None:
        n_chains = len(next(iter(chain_samples.values())))
        if n_chains <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, n_chains))
        else:
            colors = plt.cm.rainbow(np.linspace(0, 1, n_chains))
    
    for idx, (param_name, comp_idx) in enumerate(subplot_info):
        ax = axes[idx]
        
        if chain_samples is not None:
            # Plot individual chains
            chain_data = chain_samples[param_name]
            n_chains = chain_data.shape[0]
            
            if comp_idx is not None:
                # Vector component
                for chain_idx in range(n_chains):
                    chain_samples_comp = chain_data[chain_idx, :, comp_idx]
                    ax.plot(chain_samples_comp, color=colors[chain_idx], 
                           label=f'Chain {chain_idx+1}', alpha=0.8, linewidth=1)
                
                # Add true value line if available
                if param_name in theta:
                    true_val = theta[param_name]
                    if isinstance(true_val, (list, tuple, np.ndarray)):
                        true_val = np.array(true_val)
                        if comp_idx < len(true_val):
                            ax.axhline(true_val[comp_idx], color='red', linestyle='--', linewidth=2,
                                     label=f'True value = {true_val[comp_idx]:.3f}')
                ax.set_ylabel(f'{param_name}[{comp_idx}]')
                ax.set_title(f'{param_name}[{comp_idx}]')
            else:
                # Scalar parameter
                for chain_idx in range(n_chains):
                    chain_samples_scalar = chain_data[chain_idx, :]
                    ax.plot(chain_samples_scalar, color=colors[chain_idx], 
                           label=f'Chain {chain_idx+1}', alpha=0.8, linewidth=1)
                
                if param_name in theta:
                    true_val = theta[param_name]
                    if not isinstance(true_val, (list, tuple, np.ndarray)):
                        ax.axhline(true_val, color='red', linestyle='--', linewidth=2,
                                 label=f'True value = {true_val:.3f}')
                ax.set_ylabel(param_name)
                ax.set_title(param_name)
            
            # Only show legend for first few subplots to avoid clutter
            if idx < 3:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        else:
            # Plot flattened samples (original behavior)
            samples = mcmc_samples[param_name]
            if comp_idx is not None:
                # Vector component
                ax.plot(samples[:, comp_idx], color=plt.cm.tab10(comp_idx), 
                       label=f'{param_name}[{comp_idx}]', alpha=0.8)
                # Add true value line if available
                if param_name in theta:
                    true_val = theta[param_name]
                    if isinstance(true_val, (list, tuple, np.ndarray)):
                        true_val = np.array(true_val)
                        if comp_idx < len(true_val):
                            ax.axhline(true_val[comp_idx], color='red', linestyle='--',
                                     label=f'True value = {true_val[comp_idx]:.3f}')
                ax.set_ylabel(f'{param_name}[{comp_idx}]')
                ax.set_title(f'{param_name}[{comp_idx}]')
                ax.legend()
            else:
                # Scalar parameter
                ax.plot(samples, label=param_name)
                if param_name in theta:
                    true_val = theta[param_name]
                    if not isinstance(true_val, (list, tuple, np.ndarray)):
                        ax.axhline(true_val, color='red', linestyle='--', 
                                 label=f'True value = {true_val:.3f}')
                ax.set_ylabel(param_name)
                ax.set_title(param_name)
                ax.legend()
        
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Sample')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig, axes

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



