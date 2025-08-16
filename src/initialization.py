# initialization.py - Functions for initializing blob positions and velocities

import jax
import jax.numpy as jnp
from jax.scipy.special import erf
import numpy as np
from functools import partial
from utils import approx_inverse_nfw_cdf, blob_enclosed_mass_gaussian, blob_enclosed_mass_nfw, blob_enclosed_mass_plummer

# --- Position initialization functions ---
@partial(jax.jit, static_argnames=['n_part'])
def initialize_gaussian_positions(n_part, center, sigma, length, key):
    positions = jax.random.normal(key, (n_part, 3)) * sigma + center
    positions = positions % length
    return positions

@partial(jax.jit, static_argnames=['n_part'])
def initialize_nfw_positions(n_part, center, rs, c, length, key):
    key1, key2, key3 = jax.random.split(key, 3)
    u = jax.random.uniform(key1, (n_part,))
    r = approx_inverse_nfw_cdf(u, rs, c)
    phi = jax.random.uniform(key2, (n_part,)) * 2 * jnp.pi
    cos_theta = jax.random.uniform(key3, (n_part,)) * 2 - 1
    sin_theta = jnp.sqrt(1 - cos_theta**2)
    
    # Convert to Cartesian coordinates
    x = r * sin_theta * jnp.cos(phi)
    y = r * sin_theta * jnp.sin(phi)
    z = r * cos_theta
    positions = jnp.stack([x, y, z], axis=1) + center
    
    # Apply periodic boundary conditions
    positions = positions % length
    
    return positions

@partial(jax.jit, static_argnames=['n_part'])
def initialize_plummer_positions(n_part, center, rs, length, key):
    # Generate random numbers for the CDF sampling
    key1, key2, key3 = jax.random.split(key, 3)
    u = jax.random.uniform(key1, (n_part,)) 
    # Sample radial coordinate using inverse CDF for Plummer sphere
    # For Plummer sphere: M(r) = M_total * r^3 / (r^2 + rs^2)^(3/2)
    # Inverse CDF: r = rs * (u^(-2/3) - 1)^(-1/2)
    r = rs / jnp.sqrt(u**(-2.0/3.0) - 1.0)
    # Generate random angles
    phi = jax.random.uniform(key2, (n_part,)) * 2 * jnp.pi
    cos_theta = jax.random.uniform(key3, (n_part,)) * 2 - 1
    sin_theta = jnp.sqrt(1 - cos_theta**2)
    # Convert to Cartesian coordinates
    x = r * sin_theta * jnp.cos(phi)
    y = r * sin_theta * jnp.sin(phi)
    z = r * cos_theta
    positions = jnp.stack([x, y, z], axis=1) + center
    # Apply periodic boundary conditions
    positions = positions % length
    return positions

# --- Velocity initialization functions ---

@jax.jit
def initialize_cold_velocities(positions, vel_dispersion, vel_key):
    n_part = positions.shape[0]
    velocities = jax.random.normal(vel_key, (n_part, 3)) * vel_dispersion
    return velocities

@jax.jit
def initialize_virial_velocities(positions, virial_ratio, G, m_part, vel_key):
    n_part = positions.shape[0]
    # Calculate the potential energy (assuming all particles in this blob have the same mass)
    dx = positions[:, None, :] - positions[None, :, :]
    r = jnp.sqrt(jnp.sum(dx**2, axis=-1) + 1e-6)
    # Avoid self-interaction
    mask = jnp.ones((n_part, n_part)) - jnp.eye(n_part)
    W = -G * m_part * m_part * jnp.sum(mask / r) / 2.0
    # Calculate the velocity scale to achieve the desired virial ratio
    # Virial ratio = 2T/|W|, so T = virial_ratio * |W| / 2
    # T = 0.5 * m * sum(v²), so v_scale = sqrt(virial_ratio * |W| / (m * n_part))
    v_scale = jnp.sqrt(virial_ratio * jnp.abs(W) / (m_part * n_part))
    # Generate random velocity directions
    vel_directions = jax.random.normal(vel_key, (n_part, 3))
    vel_directions = vel_directions / jnp.sqrt(jnp.sum(vel_directions**2, axis=1, keepdims=True))
    # Scale velocities to achieve the desired virial ratio
    velocities = vel_directions * v_scale
    # Ensure zero net momentum
    net_momentum = jnp.sum(velocities, axis=0)
    velocities = velocities - net_momentum / n_part
    return velocities

@jax.jit
def initialize_circular_velocities_proxy(positions, vel_factor, vel_dispersion, G, m_part, softening, vel_key):
    n_part = positions.shape[0]
    center = jnp.mean(positions, axis=0)     # Calculate the center of mass
    M = n_part * m_part     # Calculate total mass
    rel_pos = positions - center     # Calculate relative positions from center
    r = jnp.linalg.norm(rel_pos, axis=1)     # Calculate radial distances
    M_enclosed = M * r / (jnp.max(r) + 1e-6)      # Simple approximation for enclosed mass: M(r) ∝ r for small r
    v_circ = jnp.sqrt(G * M_enclosed / jnp.sqrt(r**2 + softening**2))   # Calculate circular velocity magnitude
    z_axis = jnp.array([0.0, 0.0, 1.0])     # Compute direction perpendicular to radius vector in the x-y plane
    perp = jnp.cross(rel_pos, z_axis)
    perp_norm = jnp.linalg.norm(perp, axis=1, keepdims=True)     # Safe normalization of perpendicular vector
    perp_normalized = perp / (perp_norm + 1e-6) 
    velocities = v_circ[:, None] * perp_normalized
    order_magnitude = jnp.mean(jnp.linalg.norm(velocities, axis=1))
    noise = jax.random.normal(vel_key, velocities.shape) * vel_dispersion * order_magnitude  # Add random dispersion proportional to the amplitude of circular velocities
    velocities = (velocities + noise) * vel_factor
    return velocities

@jax.jit
def initialize_circular_velocities_gaussian(positions, vel_factor, vel_dispersion, G, m_part, sigma, softening, vel_key):
    n_part = positions.shape[0]
    # Calculate the center of mass (should be close to the blob center)
    center = jnp.mean(positions, axis=0)
    # Calculate total mass
    M = n_part * m_part
    # Calculate relative positions from center
    rel_pos = positions - center
    # Calculate radial distances
    r = jnp.linalg.norm(rel_pos, axis=1)
    # Calculate enclosed mass using the Gaussian profile
    M_enclosed = blob_enclosed_mass_gaussian(r, M, sigma)
    # Calculate circular velocity magnitude
    v_circ = jnp.sqrt(G * M_enclosed / jnp.sqrt(r**2 + softening**2)) 
    # Compute direction perpendicular to radius vector in the x-y plane
    z_axis = jnp.array([0.0, 0.0, 1.0])
    perp = jnp.cross(rel_pos, z_axis)
    # Safe normalization of perpendicular vector
    perp_norm = jnp.linalg.norm(perp, axis=1, keepdims=True)
    perp_normalized = perp / (perp_norm + 1e-6)
    velocities = v_circ[:, None] * perp_normalized
    order_magnitude = jnp.mean(jnp.linalg.norm(velocities, axis=1))
    # Add random dispersion proportional to the amplitude of circular velocities
    noise = jax.random.normal(vel_key, velocities.shape) * vel_dispersion * order_magnitude
    velocities = (velocities + noise) * vel_factor
    return velocities

@jax.jit
def initialize_circular_velocities_nfw(positions, vel_factor, vel_dispersion, G, m_part, c, rs, softening, vel_key):
    n_part = positions.shape[0]
    center = jnp.mean(positions, axis=0)
    M = n_part * m_part
    rel_pos = positions - center
    r = jnp.linalg.norm(rel_pos, axis=1)
    M_enclosed = blob_enclosed_mass_nfw(r, M, rs, c)
    v_circ = jnp.sqrt(G * M_enclosed / jnp.sqrt(r**2 + softening**2)) 
    z_axis = jnp.array([0.0, 0.0, 1.0])
    perp = jnp.cross(rel_pos, z_axis)
    perp_norm = jnp.linalg.norm(perp, axis=1, keepdims=True)
    perp_normalized = perp / (perp_norm + 1e-6)
    velocities = v_circ[:, None] * perp_normalized
    order_magnitude = jnp.mean(jnp.linalg.norm(velocities, axis=1))
    noise = jax.random.normal(vel_key, velocities.shape) * vel_dispersion * order_magnitude
    velocities = (velocities + noise) * vel_factor
    return velocities

@jax.jit
def initialize_circular_velocities_plummer(positions, vel_factor, vel_dispersion, G, m_part, rs, softening, vel_key):
    n_part = positions.shape[0]
    center = jnp.mean(positions, axis=0)
    M = n_part * m_part
    rel_pos = positions - center
    r = jnp.linalg.norm(rel_pos, axis=1)
    M_enclosed = blob_enclosed_mass_plummer(r, M, rs)
    v_circ = jnp.sqrt(G * M_enclosed / jnp.sqrt(r**2 + softening**2)) 
    z_axis = jnp.array([0.0, 0.0, 1.0])
    perp = jnp.cross(rel_pos, z_axis)
    perp_norm = jnp.linalg.norm(perp, axis=1, keepdims=True)
    perp_normalized = perp / (perp_norm + 1e-6)
    velocities = v_circ[:, None] * perp_normalized
    order_magnitude = jnp.mean(jnp.linalg.norm(velocities, axis=1))
    noise = jax.random.normal(vel_key, velocities.shape) * vel_dispersion * order_magnitude
    velocities = (velocities + noise) * vel_factor
    return velocities

# --- Combined initialization function ---

def initialize_blob(params_blob, fixed_params_blob, other_params_blob, params_infos_blob, length, G, softening, key):
    # Split the key for position and velocity initialization
    pos_key, vel_key = jax.random.split(key)
    
    # Extract parameter info
    fixed_param_order = params_infos_blob['fixed_param_order']
    changing_param_order = params_infos_blob['changing_param_order']
    
    # Extract center coordinates
    if 'center' in changing_param_order:
        # Single scalar format
        center_idx = changing_param_order.index('center')
        center_scalar = params_blob[center_idx]
        center = jnp.array([center_scalar, center_scalar, center_scalar])
    elif 'center_x' in changing_param_order:
        # Traditional 3D format
        center_x_idx = changing_param_order.index('center_x')
        center = params_blob[center_x_idx:center_x_idx+3]
    elif 'center' in fixed_param_order:
        # Single scalar format (fixed)
        center_idx = fixed_param_order.index('center')
        center_scalar = fixed_params_blob[center_idx]
        center = jnp.array([center_scalar, center_scalar, center_scalar])
    else:
        # Traditional 3D format (fixed)
        center_x_idx = fixed_param_order.index('center_x')
        center = fixed_params_blob[center_x_idx:center_x_idx+3]
    if 'm_part' in fixed_param_order:
        m_part_idx = fixed_param_order.index('m_part')
        m_part = fixed_params_blob[m_part_idx]
    else:
        m_part_idx = changing_param_order.index('m_part')
        m_part = params_blob[m_part_idx]
    pos_type = other_params_blob['pos_type']
    n_part = other_params_blob['n_part']
    if pos_type == 'gaussian':
        if 'sigma' in changing_param_order:
            sigma_idx = changing_param_order.index('sigma')
            sigma = params_blob[sigma_idx]
        else:
            sigma_idx = fixed_param_order.index('sigma')
            sigma = fixed_params_blob[sigma_idx]
        positions = initialize_gaussian_positions(n_part, center, sigma, length, pos_key)
    
    elif pos_type == 'nfw':
        if 'rs' in changing_param_order:
            rs_idx = changing_param_order.index('rs')
            rs = params_blob[rs_idx]
        else:
            rs_idx = fixed_param_order.index('rs')
            rs = fixed_params_blob[rs_idx]
        
        if 'c' in changing_param_order:
            c_idx = changing_param_order.index('c')
            c = params_blob[c_idx]
        else:
            c_idx = fixed_param_order.index('c')
            c = fixed_params_blob[c_idx]
        positions = initialize_nfw_positions(n_part, rs, c, length, pos_key)

    elif pos_type == 'plummer':
        if 'rs' in changing_param_order:
            rs_idx = changing_param_order.index('rs')
            rs = params_blob[rs_idx]
        else:
            rs_idx = fixed_param_order.index('rs')
            rs = fixed_params_blob[rs_idx]
        positions = initialize_plummer_positions(n_part, rs, length, pos_key)
    
    else:
        raise ValueError(f"Unknown position type: {pos_type}")
    
    # Build vel_params from other_params
    
    vel_type = other_params_blob['vel_type']
    
    if vel_type == 'cold':
        if 'vel_dispersion' in changing_param_order:
            vel_dispersion_idx = changing_param_order.index('vel_dispersion')
            vel_dispersion = params_blob[vel_dispersion_idx]
        else:
            vel_dispersion_idx = fixed_param_order.index('vel_dispersion')
            vel_dispersion = fixed_params_blob[vel_dispersion_idx]
        velocities = initialize_cold_velocities(positions, vel_dispersion, vel_key)
    elif vel_type == 'virial':
        if 'vel_dispersion' in changing_param_order:
            virial_ratio_idx = changing_param_order.index('virial_ratio')
            virial_ratio = params_blob[virial_ratio_idx]
        else:
            virial_ratio_idx = fixed_param_order.index('virial_ratio')
            virial_ratio = fixed_params_blob[virial_ratio_idx]
        velocities = initialize_virial_velocities(positions, virial_ratio, G, m_part, vel_key)
    elif vel_type == 'circular':
        distrib = other_params_blob.get('distrib', False)
        if 'vel_factor' in changing_param_order:
            vel_factor_idx = changing_param_order.index('vel_factor')
            vel_factor = params_blob[vel_factor_idx]
        else:
            vel_factor_idx = fixed_param_order.index('vel_factor')
            vel_factor = fixed_params_blob[vel_factor_idx]
        if 'vel_dispersion' in changing_param_order:
            vel_dispersion_idx = changing_param_order.index('vel_dispersion')
            vel_dispersion = params_blob[vel_dispersion_idx]
        else:
            vel_dispersion_idx = fixed_param_order.index('vel_dispersion')
            vel_dispersion = fixed_params_blob[vel_dispersion_idx]
        if distrib:
            if pos_type == 'gaussian':
                velocities = initialize_circular_velocities_gaussian(positions, vel_factor, vel_dispersion, G, m_part, sigma, softening, vel_key)
            elif pos_type == 'nfw':
                velocities = initialize_circular_velocities_nfw(positions, vel_factor, vel_dispersion, G, m_part, c, rs, softening, vel_key)
            elif pos_type == 'plummer':
                velocities = initialize_circular_velocities_plummer(positions, vel_factor, vel_dispersion, G, m_part, rs, softening, vel_key)
        else:
            velocities = initialize_circular_velocities_proxy(positions, vel_factor, vel_dispersion, G, m_part, softening, vel_key)
    else:
        raise ValueError(f"Unknown velocity type: {vel_type}")
    
    # Create mass array for this blob
    masses = jnp.full(n_part, m_part)
    
    return positions, velocities, masses

def initialize_blobs(params, fixed_params, other_params, params_infos, length, G, softening, key):
    # Split the key for each blob
    num_blobs = len(other_params)
    keys = jax.random.split(key, num_blobs)
    
    # Initialize each blob
    all_positions = []
    all_velocities = []
    all_masses = []
    for i in range(num_blobs):
        pos, vel, masses = initialize_blob(params[i], fixed_params[i], other_params[i], params_infos[i], length, G, softening, keys[i])
        all_positions.append(pos)
        all_velocities.append(vel)
        all_masses.append(masses)
    
    # Concatenate all positions, velocities, and masses
    positions = jnp.concatenate(all_positions, axis=0)
    velocities = jnp.concatenate(all_velocities, axis=0)
    masses = jnp.concatenate(all_masses, axis=0)
    
    return positions, velocities, masses