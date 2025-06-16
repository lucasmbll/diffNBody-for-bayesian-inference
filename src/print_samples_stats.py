import numpy as np
import argparse

def print_sample_statistics(samples_path):
    """
    Print basic statistics of saved MCMC samples.
    
    Parameters:
    -----------
    samples_path : str
        Path to the .npz file with saved samples
    """
    # Load samples
    samples_data = np.load(samples_path)
    
    print("="*60)
    print(f"MCMC SAMPLES STATISTICS")
    print("="*60)
    print(f"File: {samples_path}")
    print(f"Parameters found: {len(samples_data.files)}")
    print()
    
    for param_name in samples_data.files:
        samples = samples_data[param_name]
        
        print(f"Parameter: {param_name}")
        print(f"  Shape: {samples.shape}")
        
        # Handle different shapes - check if samples are in (n_samples, n_dims) or (n_dims, n_samples) format
        if samples.ndim == 1:
            # Scalar parameter
            n_samples = samples.shape[0]
            print(f"  Total samples: {n_samples}")
            print(f"  Mean: {np.mean(samples):.6f}")
            print(f"  Std:  {np.std(samples):.6f}")
            print(f"  Min:  {np.min(samples):.6f}")
            print(f"  Max:  {np.max(samples):.6f}")
            print(f"  Median: {np.median(samples):.6f}")
            print(f"  25th percentile: {np.percentile(samples, 25):.6f}")
            print(f"  75th percentile: {np.percentile(samples, 75):.6f}")
        
        elif samples.ndim == 2:
            # Vector parameter - need to determine which dimension is samples
            # If first dimension is small (like 3 for coordinates), it's likely (n_dims, n_samples)
            if samples.shape[0] <= 10 and samples.shape[1] > samples.shape[0]:
                # Shape is (n_dims, n_samples) - transpose it
                samples_transposed = samples.T  # Now (n_samples, n_dims)
                n_samples, n_dims = samples_transposed.shape
                print(f"  Total samples: {n_samples}")
                print(f"  Dimensions: {n_dims}")
                
                for i in range(n_dims):
                    component = samples_transposed[:, i]
                    print(f"    Component [{i}]:")
                    print(f"      Mean: {np.mean(component):.6f}")
                    print(f"      Std:  {np.std(component):.6f}")
                    print(f"      Min:  {np.min(component):.6f}")
                    print(f"      Max:  {np.max(component):.6f}")
                    print(f"      Median: {np.median(component):.6f}")
                    print(f"      25th percentile: {np.percentile(component, 25):.6f}")
                    print(f"      75th percentile: {np.percentile(component, 75):.6f}")
            else:
                # Shape is (n_samples, n_dims)
                n_samples, n_dims = samples.shape
                print(f"  Total samples: {n_samples}")
                print(f"  Dimensions: {n_dims}")
                
                for i in range(n_dims):
                    component = samples[:, i]
                    print(f"    Component [{i}]:")
                    print(f"      Mean: {np.mean(component):.6f}")
                    print(f"      Std:  {np.std(component):.6f}")
                    print(f"      Min:  {np.min(component):.6f}")
                    print(f"      Max:  {np.max(component):.6f}")
                    print(f"      Median: {np.median(component):.6f}")
                    print(f"      25th percentile: {np.percentile(component, 25):.6f}")
                    print(f"      75th percentile: {np.percentile(component, 75):.6f}")
        
        print("-" * 40)
    
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print basic statistics of MCMC samples")
    parser.add_argument("samples", type=str, help="Path to .npz file with samples")
    
    args = parser.parse_args()
    print_sample_statistics(args.samples)