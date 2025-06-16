import numpy as np
import yaml
import argparse
import os
from plotting import plot_trace_subplots, plot_corner_after_burnin

def fix_sample_shape(samples_dict):
    """
    Fix sample shapes to ensure they are in (n_samples, n_dims) format.
    
    Parameters:
    -----------
    samples_dict : dict
        Dictionary of samples with potentially transposed shapes
        
    Returns:
    --------
    fixed_samples : dict
        Dictionary with corrected sample shapes
    """
    fixed_samples = {}
    
    for param_name, samples in samples_dict.items():
        if samples.ndim == 2 and samples.shape[0] <= 10 and samples.shape[1] > samples.shape[0]:
            # Shape is (n_dims, n_samples) - transpose to (n_samples, n_dims)
            print(f"Transposing {param_name} from {samples.shape} to {samples.T.shape}")
            fixed_samples[param_name] = samples.T
        else:
            fixed_samples[param_name] = samples
    
    return fixed_samples

def plot_from_saved_samples(samples_path, init_params_path, config_path=None, output_dir=None):
    """
    Load saved samples and generate corner and trace plots.
    
    Parameters:
    -----------
    samples_path : str
        Path to the .npz file with saved samples
    init_params_path : str
        Path to the init_params.yaml file with true parameter values
    config_path : str, optional
        Path to the original config file (only needed for model params like G, length, etc.)
    output_dir : str, optional
        Directory to save plots. If None, saves in same directory as samples
    """
    # Load samples
    samples_data = np.load(samples_path)
    samples_dict = {key: samples_data[key] for key in samples_data.files}
    
    # Fix sample shapes if needed
    samples_dict = fix_sample_shape(samples_dict)
    
    # Load true parameters from init_params.yaml
    with open(init_params_path, "r") as f:
        init_params_data = yaml.safe_load(f)
    
    blobs_params = init_params_data["blobs_params"]
    
    # If config is provided, use it for model parameters, otherwise use defaults
    if config_path:
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
    else:
        # Use default values
        G = 5.0
        length = 64
        softening = 0.1
        t_f = 1.0
        dt = 0.5
        sampler = "nuts"
        burnin = 1000
    
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
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(samples_path)
    
    print(f"Found parameters: {list(theta.keys())}")
    print(f"True values: {theta}")
    print(f"Sample shapes after fixing: {[(k, v.shape) for k, v in samples_dict.items()]}")
    
    # Generate trace plot
    print("Generating trace plot...")
    fig, _ = plot_trace_subplots(
        samples_dict,
        theta=theta,
        G=G, t_f=t_f, dt=dt, softening=softening, length=length, n_part=n_part,
        method=sampler,
        param_order=param_order
    )
    trace_path = os.path.join(output_dir, "trace_plot_regenerated.png")
    fig.savefig(trace_path)
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
    corner_path = os.path.join(output_dir, "corner_plot_regenerated.png")
    fig.savefig(corner_path)
    print(f"Corner plot saved to: {corner_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from saved MCMC samples")
    parser.add_argument("--samples", type=str, required=True, help="Path to .npz file with samples")
    parser.add_argument("--init_params", type=str, required=True, help="Path to init_params.yaml file")
    parser.add_argument("--config", type=str, help="Path to original config file (optional, for model params)")
    parser.add_argument("--output", type=str, help="Output directory for plots (optional)")
    
    args = parser.parse_args()
    plot_from_saved_samples(args.samples, args.init_params, args.config, args.output)