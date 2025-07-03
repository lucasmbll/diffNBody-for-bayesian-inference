# test_utils.py
import yaml
import jax.numpy as jnp

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

## Radius functions

def calculate_blob_radius(positions, center, percentile=90):
    """
    Calculate the radius containing a given percentile of particles.
    
    Parameters:
    -----------
    positions : jnp.array
        Particle positions (N, 3)
    center : jnp.array
        Center position (3,)
    percentile : float
        Percentile for radius calculation (default: 90th percentile)
        
    Returns:
    --------
    radius : float
        Radius containing the specified percentile of particles
    """
    # Calculate distances from center
    distances = jnp.linalg.norm(positions - center, axis=1)
    
    # Return the percentile radius
    return jnp.percentile(distances, percentile)

def calculate_rms_radius(positions, center):
    """
    Calculate the RMS radius of the blob.
    
    Parameters:
    -----------
    positions : jnp.array
        Particle positions (N, 3)
    center : jnp.array
        Center position (3,)
        
    Returns:
    --------
    rms_radius : float
        RMS radius of the blob
    """
    distances_squared = jnp.sum((positions - center)**2, axis=1)
    return jnp.sqrt(jnp.mean(distances_squared))

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