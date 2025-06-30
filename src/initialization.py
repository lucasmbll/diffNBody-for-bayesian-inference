# initialization.py - Functions for initializing blob positions and velocities

import jax
import jax.numpy as jnp
from jax.scipy.special import erf
import numpy as np
from utils import approx_inverse_nfw_cdf, blob_enclosed_mass_gaussian, blob_enclosed_mass_nfw, blob_enclosed_mass_plummer



# --- Position initialization functions ---

def initialize_gaussian_positions(n_part, pos_params, length, key):
    """
    Initialize a blob with a Gaussian distribution.
    
    Parameters:
    -----------
    n_part : int
        Number of particles of the blob
    pos_params : dict
        Dictionary with parameters: 
        - sigma: standard deviation of the Gaussian
        - center: center position of the blob (3D array)
    length : float
        Size of the simulation box
    key : jax.random.PRNGKey
        Random key for initialization
    
    Returns:
    --------
    positions : jnp.array
        Array of shape (n_part, 3) with particle positions
    """
    sigma = pos_params['sigma']
    center = pos_params['center']
    
    center = jnp.array(center, dtype=jnp.float32)

    positions = jax.random.normal(key, (n_part, 3)) * sigma + center
    
    # Apply periodic boundary conditions
    positions = positions % length
    
    return positions

def initialize_nfw_positions(n_part, pos_params, length, key):
    """
    Initialize particle positions with an NFW profile.
    
    Parameters:
    -----------
    n_part : int
        Number of particles
    pos_params : dict
        Dictionary with parameters:
        - rs: scale radius
        - c: concentration parameter
        - center: center position of the halo (3D array)
    length : float
        Size of the simulation box
    key : jax.random.PRNGKey
        Random key for initialization
    
    Returns:
    --------
    positions : jnp.array
        Array of shape (n_part, 3) with particle positions
    """
    rs = pos_params['rs'] # scale radius
    c = pos_params['c'] # concentration parameter
    center = pos_params['center']
    center = jnp.array(center, dtype=jnp.float32)
    
    # Generate random numbers for the CDF sampling
    key1, key2, key3 = jax.random.split(key, 3)
    u = jax.random.uniform(key1, (n_part,))
    
    # Sample radial coordinate using the inverse CDF
    # NFW enclosed mass: M(r) ∝ log(1+r/rs) - (r/rs)/(1+r/rs)
    # Inverting this to get r from a uniform random variable is done numerically
    
    r = approx_inverse_nfw_cdf(u, rs, c)
    
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


def initialize_plummer_positions(n_part, pos_params, length, key):
    """
    Initialize particle positions with a Plummer sphere profile.
    
    Parameters:
    -----------
    n_part : int
        Number of particles
    pos_params : dict
        Dictionary with parameters:
        - rs: scale radius (Plummer radius)
        - center: center position of the sphere (3D array)
    length : float
        Size of the simulation box
    key : jax.random.PRNGKey
        Random key for initialization
    
    Returns:
    --------
    positions : jnp.array
        Array of shape (n_part, 3) with particle positions
    """
    rs = pos_params['rs']  # Plummer radius
    center = pos_params['center']
    center = jnp.array(center, dtype=jnp.float32)
    
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

def initialize_cold_velocities(positions, vel_params):
    """
    Initialize velocities with near-zero values (cold start).
    
    Parameters:
    -----------
    positions : jnp.array
        Array of shape (n_part, 3) with particle positions
    vel_params : dict
        Dictionary with parameters:
        - vel_dispersion: small velocity dispersion (close to 0)
    m_part : float
        Mass per particle
        
    Returns:
    --------
    velocities : jnp.array
        Array of shape (n_part, 3) with particle velocities
    """
    n_part = positions.shape[0]
    vel_dispersion = vel_params.get('vel_dispersion', 1e-6)
    
    # Generate small random velocities
    key = jax.random.PRNGKey(vel_params.get('seed', 0))
    velocities = jax.random.normal(key, (n_part, 3)) * vel_dispersion
    
    return velocities

def initialize_virial_velocities(positions, vel_params, G, m_part):
    """
    Initialize velocities to achieve a given virial ratio.
    
    Parameters:
    -----------
    positions : jnp.array
        Array of shape (n_part, 3) with particle positions
    vel_params : dict
        Dictionary with parameters:
        - virial_ratio: target virial ratio (2T/|W|)
        - seed: random seed for velocity directions
    G : float
        Gravitational constant
    m_part : float
        Mass per particle for this blob
        
    Returns:
    --------
    velocities : jnp.array
        Array of shape (n_part, 3) with particle velocities
    """
    n_part = positions.shape[0]
    virial_ratio = vel_params.get('virial_ratio', 1.0)
    
    # Calculate the center of mass
    com = jnp.mean(positions, axis=0)
    
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
    key = jax.random.PRNGKey(vel_params.get('seed', 0))
    vel_directions = jax.random.normal(key, (n_part, 3))
    vel_directions = vel_directions / jnp.sqrt(jnp.sum(vel_directions**2, axis=1, keepdims=True))
    
    # Scale velocities to achieve the desired virial ratio
    velocities = vel_directions * v_scale
    
    # Ensure zero net momentum
    net_momentum = jnp.sum(velocities, axis=0)
    velocities = velocities - net_momentum / n_part
    
    return velocities

def initialize_circular_velocities_proxy(positions, vel_params, G, m_part, softening):
    """
    Initialize velocities to circular orbits around the center of mass with a proxy method.
    
    Parameters:
    -----------
    positions : jnp.array
        Array of shape (n_part, 3) with particle positions
    vel_params : dict
        Dictionary with parameters:
        - vel_factor: optional factor to multiply circular velocities (default: 1.0)
    G : float
        Gravitational constant
    m_part : float
        Mass per particle
        
    Returns:
    --------
    velocities : jnp.array
        Array of shape (n_part, 3) with particle velocities
    """
    n_part = positions.shape[0]
    vel_factor = vel_params.get('vel_factor', 1.0)
    
    # Calculate the center of mass
    center = jnp.mean(positions, axis=0)
    
    # Calculate total mass
    M = n_part * m_part
    
    # Calculate relative positions from center
    rel_pos = positions - center
    
    # Calculate radial distances
    r = jnp.linalg.norm(rel_pos, axis=1)
    
    # Simple approximation for enclosed mass: M(r) ∝ r for small r
    M_enclosed = M * r / (jnp.max(r) + 1e-6)  
    
    # Calculate circular velocity magnitude
    v_circ = jnp.sqrt(G * M_enclosed / jnp.sqrt(r**2 + softening**2)) * vel_factor
    
    # Compute direction perpendicular to radius vector in the x-y plane
    z_axis = jnp.array([0.0, 0.0, 1.0])
    perp = jnp.cross(rel_pos, z_axis)
    
    # SAFETY MEASURE 3: Safe normalization of perpendicular vector
    perp_norm = jnp.linalg.norm(perp, axis=1, keepdims=True)
    perp_normalized = perp / (perp_norm + 1e-6) 
    
    # Set velocities
    velocities = v_circ[:, None] * perp_normalized
    
    return velocities

def initialize_circular_velocities_gaussian(positions, vel_params, G, m_part, pos_params, softening):
    """
    Initialize velocities to circular orbits for a Gaussian blob.
    
    Parameters:
    -----------
    positions : jnp.array
        Array of shape (n_part, 3) with particle positions in the blob
    vel_params : dict
        Dictionary with parameters:
        - vel_factor: optional factor to multiply circular velocities (default: 1.0)
    G : float
        Gravitational constant
    m_part : float
        Mass per particle
    pos_params : dict
        Position parameters including sigma for Gaussian distribution
        
    Returns:
    --------
    velocities : jnp.array
        Array of shape (n_part, 3) with particle velocities
    """
    n_part = positions.shape[0]
    vel_factor = vel_params.get('vel_factor', 1.0)
    sigma = pos_params['sigma']
    
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
    v_circ = jnp.sqrt(G * M_enclosed / jnp.sqrt(r**2 + softening**2)) * vel_factor
    
    # Compute direction perpendicular to radius vector in the x-y plane
    z_axis = jnp.array([0.0, 0.0, 1.0])
    perp = jnp.cross(rel_pos, z_axis)
    
    # SAFETY MEASURE 3: Safe normalization of perpendicular vector
    perp_norm = jnp.linalg.norm(perp, axis=1, keepdims=True)
    perp_normalized = perp / (perp_norm + 1e-6)
    
    # Set velocities
    velocities = v_circ[:, None] * perp_normalized
    
    return velocities

def initialize_circular_velocities_nfw(positions, vel_params, G, m_part, pos_params, softening):
    """
    Initialize velocities to circular orbits for an NFW halo (analytic mass).

    Parameters:
    -----------
    positions : jnp.array
        Array of shape (n_part, 3) with particle positions
    vel_params : dict
        Dictionary with parameters:
        - vel_factor: optional factor to multiply circular velocities (default: 1.0)
    G : float
        Gravitational constant
    m_part : float
        Mass per particle
    pos_params : dict
        Position parameters including rs and c for NFW distribution

    Returns:
    --------
    velocities : jnp.array
        Array of shape (n_part, 3) with particle velocities
    """
    n_part = positions.shape[0]
    vel_factor = vel_params.get('vel_factor', 1.0)
    rs = pos_params['rs']
    c = pos_params['c']

    center = jnp.mean(positions, axis=0)
    M = n_part * m_part
    rel_pos = positions - center
    r = jnp.linalg.norm(rel_pos, axis=1)

    M_enclosed = blob_enclosed_mass_nfw(r, M, rs, c)
    
    v_circ = jnp.sqrt(G * M_enclosed / jnp.sqrt(r**2 + softening**2)) * vel_factor

    z_axis = jnp.array([0.0, 0.0, 1.0])
    perp = jnp.cross(rel_pos, z_axis)
    
    perp_norm = jnp.linalg.norm(perp, axis=1, keepdims=True)
    perp_normalized = perp / (perp_norm + 1e-6)
    
    velocities = v_circ[:, None] * perp_normalized
       
    return velocities


def initialize_circular_velocities_plummer(positions, vel_params, G, m_part, pos_params, softening):
    """
    Initialize velocities to circular orbits for a Plummer sphere.

    Parameters:
    -----------
    positions : jnp.array
        Array of shape (n_part, 3) with particle positions
    vel_params : dict
        Dictionary with parameters:
        - vel_factor: optional factor to multiply circular velocities (default: 1.0)
    G : float
        Gravitational constant
    m_part : float
        Mass per particle
    pos_params : dict
        Position parameters including rs for Plummer sphere

    Returns:
    --------
    velocities : jnp.array
        Array of shape (n_part, 3) with particle velocities
    """
    n_part = positions.shape[0]
    vel_factor = vel_params.get('vel_factor', 1.0)
    rs = pos_params['rs']

    center = jnp.mean(positions, axis=0)
    M = n_part * m_part
    rel_pos = positions - center
    r = jnp.linalg.norm(rel_pos, axis=1)

    M_enclosed = blob_enclosed_mass_plummer(r, M, rs)
    
    v_circ = jnp.sqrt(G * M_enclosed / jnp.sqrt(r**2 + softening**2)) * vel_factor

    z_axis = jnp.array([0.0, 0.0, 1.0])
    perp = jnp.cross(rel_pos, z_axis)
    
    perp_norm = jnp.linalg.norm(perp, axis=1, keepdims=True)
    perp_normalized = perp / (perp_norm + 1e-6)
    
    velocities = v_circ[:, None] * perp_normalized
       
    return velocities

# --- Combined initialization function ---

def initialize_blob(blob_params, length, G, softening, key):
    """
    Initialize a single blob with given parameters.
    
    Parameters:
    -----------
    blob_params : dict
        Dictionary with blob parameters:
        - n_part: number of particles
        - m_part: mass per particle for this blob (optional, defaults to 1.0)
        - pos_type: position distribution type ('gaussian', 'nfw', or 'plummer')
        - pos_params: position distribution parameters
        - vel_type: velocity distribution type ('cold', 'virial', or 'circular')
        - vel_params: velocity distribution parameters
    length : float
        Size of the simulation box
    G : float
        Gravitational constant
    softening : float
        Softening length
    key : jax.random.PRNGKey
        Random key for initialization
        
    Returns:
    --------
    positions : jnp.array
        Array of shape (n_part, 3) with particle positions
    velocities : jnp.array
        Array of shape (n_part, 3) with particle velocities
    masses : jnp.array
        Array of shape (n_part,) with particle masses
    """
    n_part = blob_params['n_part']
    m_part = blob_params.get('m_part', 1.0)  # Default mass per particle
    pos_type = blob_params['pos_type']
    pos_params = blob_params['pos_params']
    vel_type = blob_params['vel_type']
    vel_params = blob_params['vel_params']
    
    # Split the key for position and velocity initialization
    pos_key, vel_key = jax.random.split(key)
    
    # Initialize positions
    if pos_type == 'gaussian':
        positions = initialize_gaussian_positions(n_part, pos_params, length, pos_key)
    elif pos_type == 'nfw':
        positions = initialize_nfw_positions(n_part, pos_params, length, pos_key)
    elif pos_type == 'plummer':
        positions = initialize_plummer_positions(n_part, pos_params, length, pos_key)
    else:
        raise ValueError(f"Unknown position type: {pos_type}")
    
    if vel_type == 'cold':
        velocities = initialize_cold_velocities(positions, vel_params)
    elif vel_type == 'virial':
        velocities = initialize_virial_velocities(positions, vel_params, G, m_part)
    elif vel_type == 'circular':
        distrib = vel_params.get('distrib', False)
        # For circular velocities, we use the specific function for the position distribution
        if distrib:
            if pos_type == 'gaussian':
                velocities = initialize_circular_velocities_gaussian(positions, vel_params, G, m_part, pos_params, softening)
            elif pos_type == 'nfw':
                velocities = initialize_circular_velocities_nfw(positions, vel_params, G, m_part, pos_params, softening)
            elif pos_type == 'plummer':
                velocities = initialize_circular_velocities_plummer(positions, vel_params, G, m_part, pos_params, softening)
        else:
            velocities = initialize_circular_velocities_proxy(positions, vel_params, G, m_part, softening)
    else:
        raise ValueError(f"Unknown velocity type: {vel_type}")
    
    # Create mass array for this blob
    masses = jnp.full(n_part, m_part)
    
    return positions, velocities, masses

def initialize_blobs(blobs_params, length, G, softening, key=None):
    """
    Initialize multiple blobs with given parameters.
    
    Parameters:
    -----------
    blobs_params : list of dict
        List of dictionaries with blob parameters
    length : float
        Size of the simulation box
    G : float
        Gravitational constant
    softening : float
        Softening length
    key : jax.random.PRNGKey, optional
        Random key for initialization
        
    Returns:
    --------
    positions : jnp.array
        Array of shape (total_n_part, 3) with particle positions
    velocities : jnp.array
        Array of shape (total_n_part, 3) with particle velocities
    masses : jnp.array
        Array of shape (total_n_part,) with particle masses
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    # Split the key for each blob
    keys = jax.random.split(key, len(blobs_params))
    
    # Initialize each blob
    all_positions = []
    all_velocities = []
    all_masses = []
    for i, blob_params in enumerate(blobs_params):
        pos, vel, masses = initialize_blob(blob_params, length, G, softening, keys[i])
        all_positions.append(pos)
        all_velocities.append(vel)
        all_masses.append(masses)
    
    # Concatenate all positions, velocities, and masses
    positions = jnp.concatenate(all_positions, axis=0)
    velocities = jnp.concatenate(all_velocities, axis=0)
    masses = jnp.concatenate(all_masses, axis=0)
    
    return positions, velocities, masses

