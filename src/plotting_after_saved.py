import numpy as np
import yaml
import argparse
import os
from plotting import plot_trace_subplots, plot_corner_after_burnin

def fix_sample_shape(samples_dict):
    """
    Fix sample shapes to ensure they are in (n_samples, n_dims) format.
    Also handles double chain results by combining chains.
    
    Parameters:
    -----------
    samples_dict : dict
        Dictionary of samples with potentially transposed shapes or multiple chains
        
    Returns:
    --------
    fixed_samples : dict
        Dictionary with corrected sample shapes
    chain_info : dict
        Information about chain structure
    """
    fixed_samples = {}
    chain_info = {'has_multiple_chains': False, 'n_chains': 1, 'original_shapes': {}}
    
    for param_name, samples in samples_dict.items():
        original_shape = samples.shape
        chain_info['original_shapes'][param_name] = original_shape
        
        if samples.ndim == 3:
            # Shape is likely (n_chains, n_samples, n_dims) or (n_samples, n_chains, n_dims)
            print(f"Detected 3D array for {param_name} with shape {samples.shape}")
            
            # Heuristic: if first dimension is small (<=10), it's likely n_chains
            if samples.shape[0] <= 10:
                # (n_chains, n_samples, n_dims) -> combine chains
                n_chains, n_samples, n_dims = samples.shape
                combined_samples = samples.reshape(-1, n_dims)
                print(f"Combined {n_chains} chains for {param_name}: {original_shape} -> {combined_samples.shape}")
                fixed_samples[param_name] = combined_samples
                chain_info['has_multiple_chains'] = True
                chain_info['n_chains'] = max(chain_info['n_chains'], n_chains)
            elif samples.shape[2] <= 10:
                # (n_samples, n_chains, n_dims) -> combine chains
                n_samples, n_chains, n_dims = samples.shape
                combined_samples = samples.transpose(1, 0, 2).reshape(-1, n_dims)
                print(f"Combined {n_chains} chains for {param_name}: {original_shape} -> {combined_samples.shape}")
                fixed_samples[param_name] = combined_samples
                chain_info['has_multiple_chains'] = True
                chain_info['n_chains'] = max(chain_info['n_chains'], n_chains)
            else:
                # If unclear, assume middle dimension is n_chains
                n_samples, n_chains, n_dims = samples.shape
                combined_samples = samples.transpose(1, 0, 2).reshape(-1, n_dims)
                print(f"Assumed middle dim is chains for {param_name}: {original_shape} -> {combined_samples.shape}")
                fixed_samples[param_name] = combined_samples
                chain_info['has_multiple_chains'] = True
                chain_info['n_chains'] = max(chain_info['n_chains'], n_chains)
                
        elif samples.ndim == 2 and samples.shape[0] <= 10 and samples.shape[1] > samples.shape[0]:
            # Shape is (n_dims, n_samples) - transpose to (n_samples, n_dims)
            print(f"Transposing {param_name} from {samples.shape} to {samples.T.shape}")
            fixed_samples[param_name] = samples.T
        else:
            fixed_samples[param_name] = samples
    
    return fixed_samples, chain_info

def plot_from_saved_samples(samples_path, config_path):
    """
    Load saved samples and generate corner and trace plots.
    Now handles double chain results.
    
    Parameters:
    -----------
    samples_path : str
        Path to the .npz file with saved samples
    config_path : str, optional
        Path to the original config file (only needed for model params like G, length, etc.)
    """
    # Load samples
    samples_data = np.load(samples_path)
    samples_dict = {key: samples_data[key] for key in samples_data.files}
    
    # Fix sample shapes if needed and get chain info
    samples_dict, chain_info = fix_sample_shape(samples_dict)
    
    # Load true parameters from init_params.yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    model_params = config["model_params"]
    G = model_params.get("G", 5.0)
    length = model_params.get("length", 64)
    softening = model_params.get("softening", 0.1)
    t_f = model_params.get("t_f", 1.0)
    dt = model_params.get("dt", 0.5)
    sampler = config.get("sampler", "nuts")
    burnin = config.get("num_warmup", 1000)
    
    # Extract prior and initial position information
    prior_params = config.get("prior_params", {})
    initial_position = config.get("initial_position", {})
    prior_type = config.get("prior_type", "unknown")
    
    blobs_params = model_params["blobs_params"]
    n_part = sum(blob['n_part'] for blob in blobs_params)
    
    # Extract true values from blobs_params
    def extract_true_values_from_blobs(blobs_params):
        true_values = {}
        for blob_idx, blob in enumerate(blobs_params):
            # Position parameters
            if blob['pos_type'] == 'gaussian':
                true_values[f"blob{blob_idx}_sigma"] = blob['pos_params']['sigma']
                true_values[f"blob{blob_idx}_center"] = blob['pos_params']['center']
            elif blob['pos_type'] == 'nfw':
                true_values[f"blob{blob_idx}_rs"] = blob['pos_params']['rs']
                true_values[f"blob{blob_idx}_c"] = blob['pos_params']['c']
                true_values[f"blob{blob_idx}_center"] = blob['pos_params']['center']
            
            # Velocity parameters
            if blob['vel_type'] == 'cold' and 'vel_dispersion' in blob['vel_params']:
                true_values[f"blob{blob_idx}_vel_dispersion"] = blob['vel_params']['vel_dispersion']
            elif blob['vel_type'] == 'virial' and 'virial_ratio' in blob['vel_params']:
                true_values[f"blob{blob_idx}_virial_ratio"] = blob['vel_params']['virial_ratio']
            elif blob['vel_type'] == 'circular' and 'vel_factor' in blob['vel_params']:
                true_values[f"blob{blob_idx}_vel_factor"] = blob['vel_params']['vel_factor']
        
        return true_values
    
    # Build theta dict
    all_true_values = extract_true_values_from_blobs(blobs_params)
    inferred_keys = list(samples_dict.keys())
    theta = {}
    for key in inferred_keys:
        theta[key] = all_true_values.get(key, None)
    
    param_order = tuple(inferred_keys)
    
    # Create additional info strings for suptitles
    def format_prior_info(param_name, prior_params, prior_type):
        """Format prior information for a parameter."""
        if param_name not in prior_params:
            return "No prior"
        
        prior_info = prior_params[param_name]
        if prior_type == "blob_gaussian":
            if isinstance(prior_info.get('mu'), list):
                mu_str = f"[{', '.join([f'{x:.1f}' for x in prior_info['mu']])}]"
            else:
                mu_str = f"{prior_info.get('mu', 'N/A')}"
            return f"Prior: N({mu_str}, {prior_info.get('sigma', 'N/A')})"
        elif prior_type == "blob_uniform":
            if isinstance(prior_info.get('low'), list):
                low_str = f"[{', '.join([f'{x:.1f}' for x in prior_info['low']])}]"
                high_str = f"[{', '.join([f'{x:.1f}' for x in prior_info['high']])}]"
            else:
                low_str = f"{prior_info.get('low', 'N/A')}"
                high_str = f"{prior_info.get('high', 'N/A')}"
            return f"Prior: U({low_str}, {high_str})"
        else:
            return f"Prior: {prior_type}"
    
    def format_initial_pos(param_name, initial_position):
        """Format initial position for a parameter."""
        if param_name not in initial_position:
            return "No init"
        
        init_val = initial_position[param_name]
        if isinstance(init_val, list):
            return f"Init: [{', '.join([f'{x:.1f}' for x in init_val])}]"
        else:
            return f"Init: {init_val}"
    
    # Build info strings for each parameter
    param_info = {}
    for param_name in inferred_keys:
        prior_str = format_prior_info(param_name, prior_params, prior_type)
        init_str = format_initial_pos(param_name, initial_position)
        param_info[param_name] = f"{prior_str} | {init_str}"
    
    output_dir = os.path.dirname(samples_path)
    
    print(f"Found parameters: {list(theta.keys())}")
    print(f"True values: {theta}")
    print(f"Chain info: {chain_info}")
    print(f"Sample shapes after fixing: {[(k, v.shape) for k, v in samples_dict.items()]}")
    
    # Generate trace plot
    print("Generating trace plot...")
    fig, axes = plot_trace_subplots(
        samples_dict,
        theta=theta,
        G=G, t_f=t_f, dt=dt, softening=softening, length=length, n_part=n_part,
        method=sampler,
        param_order=param_order
    )
    
    # Add chain info to trace plot suptitle
    chain_info_str = ""
    if chain_info['has_multiple_chains']:
        chain_info_str = f" | Combined {chain_info['n_chains']} chains"
    
    # Add prior and initial position info to trace plot suptitle
    trace_info_lines = []
    for param_name in param_order:
        trace_info_lines.append(f"{param_name}: {param_info[param_name]}")
    
    # Get existing suptitle and add info
    existing_suptitle = fig._suptitle.get_text() if fig._suptitle else "Trace Plot"
    new_suptitle = f"{existing_suptitle}{chain_info_str}\n" + " | ".join(trace_info_lines[:2])  # Limit to first 2 params to avoid overcrowding
    if len(trace_info_lines) > 2:
        new_suptitle += f"\n" + " | ".join(trace_info_lines[2:])
    
    fig.suptitle(new_suptitle, fontsize=10)
    
    trace_path = os.path.join(output_dir, "trace_plot_regenerated.png")
    fig.savefig(trace_path, bbox_inches='tight')
    print(f"Trace plot saved to: {trace_path}")
    
    # Generate corner plot
    print("Generating corner plot...")
    fig = plot_corner_after_burnin(
        samples_dict,
        theta=theta,
        G=G, t_f=t_f, dt=dt, softening=softening, length=length, n_part=n_part,
        method=sampler,
        burnin=burnin,
        param_order=param_order
    )
    
    # Add chain info to corner plot suptitle
    # Add prior and initial position info to corner plot suptitle
    corner_info_lines = []
    for param_name in param_order:
        corner_info_lines.append(f"{param_name}: {param_info[param_name]}")
    
    # Get existing suptitle and add info
    existing_suptitle = fig._suptitle.get_text() if fig._suptitle else "Corner Plot"
    new_suptitle = f"{existing_suptitle}{chain_info_str}\n" + " | ".join(corner_info_lines[:2])  # Limit to first 2 params
    if len(corner_info_lines) > 2:
        new_suptitle += f"\n" + " | ".join(corner_info_lines[2:])
    
    fig.suptitle(new_suptitle, fontsize=10)
    
    corner_path = os.path.join(output_dir, "corner_plot_regenerated.png")
    fig.savefig(corner_path, bbox_inches='tight')
    print(f"Corner plot saved to: {corner_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from saved MCMC samples")
    parser.add_argument("--samples", type=str, required=True, help="Path to .npz file with samples")
    parser.add_argument("--config", type=str, required=True, help="Path to original config file (optional, for model params)")
    
    args = parser.parse_args()
    plot_from_saved_samples(args.samples, args.config)