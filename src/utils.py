# utils.py

import jax.numpy as jnp
from jax.scipy.special import erf

def calculate_energy(pos, vel, G, length, softening, m_part):
    """Calculate kinetic, potential, and total energy of the system"""
    n_particles = pos.shape[0]
    
    # Kinetic energy
    ke = 0.5 * m_part * jnp.sum(vel**2)
    
    # Potential energy with periodic boundaries
    dx = pos[:, None, :] - pos[None, :, :]
    dx = dx - length * jnp.round(dx / length)  # periodic boundaries
    r2 = jnp.sum(dx**2, axis=-1) + softening**2  # softening squared
    r = jnp.sqrt(r2)
    
    # Upper triangular part to avoid double counting, exclude diagonal
    mask = jnp.triu(jnp.ones((n_particles, n_particles)), k=1)
    pe = -G * m_part * m_part * jnp.sum(mask / r)
    
    total_energy = ke + pe
    
    return ke, pe, total_energy

def blob_enclosed_mass_gaussian(r, M, sigma):
    """
    Calculate the enclosed mass for a Gaussian blob.
    
    Parameters:
    -----------
    r : jnp.array
        Radial distances
    M : float
        Total mass
    sigma : float
        Standard deviation of the Gaussian
        
    Returns:
    --------
    M_enclosed : jnp.array
        Enclosed mass at each radius r
    """
    x = r / (jnp.sqrt(2) * sigma)
    term1 = erf(x)
    term2 = jnp.sqrt(2 / jnp.pi) * x * jnp.exp(-x**2)
    return M * (term1 - term2)


def blob_enclosed_mass_nfw(r, M, rs, c):
        """
        Calculate the enclosed mass for an NFW profile.

        Parameters:
        -----------
        r : jnp.array
            Radial distances
        M : float
            Total mass
        rs : float
            Scale radius
        c : float
            Concentration parameter

        Returns:
        --------
        M_enclosed : jnp.array
            Enclosed mass at each radius r
        """
        x = r / rs
        norm = jnp.log(1 + c) - c / (1 + c)
        enclosed = (jnp.log(1 + x) - x / (1 + x)) / norm
        # Ensure no negative or nan values for r=0
        enclosed = jnp.where(r > 0, enclosed, 0.0)
        return M * enclosed

def apply_density_scaling(density_field, scaling_type="none", **scaling_kwargs):
    """
    Apply scaling transformation to density field.
    
    Parameters:
    -----------
    density_field : jnp.array
        Input density field to scale
    scaling_type : str
        Type of scaling to apply:
        - "none": No scaling (default)
        - "log": Logarithmic scaling: log(field + offset)
        - "sqrt": Square root scaling: sqrt(field)
        - "normalize": Normalize to [0, 1]
        - "standardize": Standardize to mean=0, std=1
        - "power": Power scaling: field^power
    scaling_kwargs : dict
        Additional parameters for scaling:
        - log_offset: offset for log scaling (default: 1e-8)
        - power: exponent for power scaling (default: 0.5)
    
    Returns:
    --------
    scaled_field : jnp.array
        Scaled density field
    """
    if scaling_type == "none":
        # No scaling: f(x) = x
        return density_field
    
    elif scaling_type == "log":
        # Logarithmic scaling: f(x) = log(x + offset)
        log_offset = float(scaling_kwargs.get("log_offset", 1e-8))
        return jnp.log(density_field + log_offset)
    
    elif scaling_type == "sqrt":
        # Square root scaling: f(x) = sqrt(max(x, 0))
        return jnp.sqrt(jnp.maximum(density_field, 0.0))
    
    elif scaling_type == "normalize":
        # Normalize to [0, 1]: f(x) = (x - min(x)) / (max(x) - min(x))
        field_min = jnp.min(density_field)
        field_max = jnp.max(density_field)
        field_range = field_max - field_min
        field_range = jnp.clip(field_max - field_min, 1e-8, None)  # Avoid division by zero
        return (density_field - field_min) / field_range
    
    elif scaling_type == "standardize":
        # Standardize to mean 0, std 1: f(x) = (x - mean(x)) / std(x)
        field_mean = jnp.mean(density_field)
        field_std = jnp.std(density_field)
        if field_std > 0:
            return (density_field - field_mean) / field_std
        else:
            return density_field - field_mean
    
    elif scaling_type == "power":
        # Power scaling: f(x) = max(x, 0) ** power
        power = float(scaling_kwargs.get("power", 0.5))
        return jnp.power(jnp.maximum(density_field, 0.0), power)
    
    else:
        raise ValueError(f"Unknown scaling_type: {scaling_type}. Available options: "
                        "'none', 'log', 'sqrt', 'normalize', 'standardize', 'power'")
    
# We use the approximation from Robotham & Howlett (2018)
def approx_inverse_nfw_cdf(x, rs, c):
    p = 0.3333 - 0.2*jnp.log10(c) + 0.01*c
    a = 1.0 / c 
    q = x * (1 - a) + a
    r = rs * (q**(-1.0/p) - 1.0) / (1.0 - q**(-1.0/p) / c)
    return r

def extract_true_values_from_blobs(blobs_params):
        """Extract true parameter values from blobs_params configuration."""
        true_values = {}
        
        for blob_idx, blob in enumerate(blobs_params):
            # Extract position parameters
            if blob['pos_type'] == 'gaussian':
                true_values[f"blob{blob_idx}_sigma"] = blob['pos_params']['sigma']
                true_values[f"blob{blob_idx}_center"] = blob['pos_params']['center']
            elif blob['pos_type'] == 'nfw':
                true_values[f"blob{blob_idx}_rs"] = blob['pos_params']['rs']
                true_values[f"blob{blob_idx}_c"] = blob['pos_params']['c']
                true_values[f"blob{blob_idx}_center"] = blob['pos_params']['center']
            
            # Extract velocity parameters
            if blob['vel_type'] == 'cold':
                if 'vel_dispersion' in blob['vel_params']:
                    true_values[f"blob{blob_idx}_vel_dispersion"] = blob['vel_params']['vel_dispersion']
            elif blob['vel_type'] == 'virial':
                if 'virial_ratio' in blob['vel_params']:
                    true_values[f"blob{blob_idx}_virial_ratio"] = blob['vel_params']['virial_ratio']
            elif blob['vel_type'] == 'circular':
                if 'vel_factor' in blob['vel_params']:
                    true_values[f"blob{blob_idx}_vel_factor"] = blob['vel_params']['vel_factor']
        
        return true_values

def format_prior_info(param_name, prior_params, prior_type):
    if param_name not in prior_params:
        return "No prior"
    prior_info = prior_params[param_name]
    if prior_type == "blob_gaussian":
        if isinstance(prior_info.get('mu'), list):
            mu_str = f"[{', '.join([f'{x:.2f}' for x in prior_info['mu']])}]"
        else:
            mu_str = f"{prior_info.get('mu', 'N/A')}"
        return f"Prior: N({mu_str}, {prior_info.get('sigma', 'N/A')})"
    elif prior_type == "blob_uniform":
        if isinstance(prior_info.get('low'), list):
            low_str = f"[{', '.join([f'{x:.2f}' for x in prior_info['low']])}]"
            high_str = f"[{', '.join([f'{x:.2f}' for x in prior_info['high']])}]"
        else:
            low_str = f"{prior_info.get('low', 'N/A')}"
            high_str = f"{prior_info.get('high', 'N/A')}"
        return f"Prior: U({low_str}, {high_str})"
    else:
        return f"Prior: {prior_type}"

def format_initial_pos(param_name, initial_position):
    if param_name not in initial_position:
        return "No init"
    init_val = initial_position[param_name]
    if isinstance(init_val, list):
        return f"Init: [{', '.join([f'{x:.2f}' for x in init_val])}]"
    else:
        return f"Init: {init_val}"
