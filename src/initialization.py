# initialization.py - Functions for initializing blob positions and velocities

import jax
import jax.numpy as jnp
from jax.scipy.special import erf
import numpy as np

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

    # Generate positions
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
    
    from utils import approx_inverse_nfw_cdf
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

# --- Velocity initialization functions ---

def initialize_cold_velocities(positions, vel_params, m_part):
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
        Mass per particle
        
    Returns:
    --------
    velocities : jnp.array
        Array of shape (n_part, 3) with particle velocities
    """
    n_part = positions.shape[0]
    virial_ratio = vel_params.get('virial_ratio', 1.0)
    
    # Calculate the center of mass
    com = jnp.mean(positions, axis=0)
    
    # Calculate the potential energy
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

def initialize_circular_velocities_proxy(positions, vel_params, G, m_part):
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
    
    # SAFETY MEASURE 1: Safe guard against very small radii
    r_min = 0.1  # Increased from 0.01 for more safety
    r_safe = jnp.maximum(r, r_min)

    # Simple approximation for enclosed mass: M(r) ∝ r for small r
    M_enclosed = M * r_safe / jnp.max(r_safe)
    
    # Calculate circular velocity magnitude
    v_circ = jnp.sqrt(G * M_enclosed / r_safe) * vel_factor
    
    # SAFETY MEASURE 2: Cap maximum velocity
    v_max = 10.0  # Maximum allowed velocity
    v_circ = jnp.minimum(v_circ, v_max)
    
    # Compute direction perpendicular to radius vector in the x-y plane
    z_axis = jnp.array([0.0, 0.0, 1.0])
    perp = jnp.cross(rel_pos, z_axis)
    
    # SAFETY MEASURE 3: Safe normalization of perpendicular vector
    perp_norm = jnp.linalg.norm(perp, axis=1, keepdims=True)
    perp_norm_safe = jnp.maximum(perp_norm, 1e-6)  # Minimum norm threshold
    perp_normalized = perp / perp_norm_safe
    
    # Set velocities
    velocities = v_circ[:, None] * perp_normalized
    
    # SAFETY MEASURE 4: Zero velocity for particles too close to center or with problematic geometry
    is_too_close = r < r_min
    has_tiny_perp = perp_norm[:, 0] < 1e-6  # Particles too close to z-axis
    is_problematic = is_too_close | has_tiny_perp
    
    # Apply safety condition: zero velocity for problematic particles
    velocities = jnp.where(
        is_problematic[:, None],  # Condition broadcasted to 3D
        jnp.zeros_like(velocities),  # Zero velocity if problematic
        velocities  # Original velocity otherwise
    )
    
    return velocities

def initialize_circular_velocities_gaussian(positions, vel_params, G, m_part, pos_params):
    """
    Initialize velocities to circular orbits for a Gaussian blob.
    
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
    
    # SAFETY MEASURE 1: Safe guard against very small radii
    r_min = 0.1 * sigma  # Scale with blob size
    r_safe = jnp.maximum(r, r_min)
    
    # Calculate enclosed mass using the Gaussian profile
    from utils import blob_enclosed_mass_gaussian
    M_enclosed = blob_enclosed_mass_gaussian(r_safe, M, sigma)
    
    # SAFETY MEASURE: Ensure minimum enclosed mass
    M_enclosed = jnp.maximum(M_enclosed, 0.01 * M)
    
    # Calculate circular velocity magnitude
    v_circ = jnp.sqrt(G * M_enclosed / r_safe) * vel_factor
    
    # SAFETY MEASURE 2: Cap maximum velocity
    v_max = 10.0
    v_circ = jnp.minimum(v_circ, v_max)
    
    # Compute direction perpendicular to radius vector in the x-y plane
    z_axis = jnp.array([0.0, 0.0, 1.0])
    perp = jnp.cross(rel_pos, z_axis)
    
    # SAFETY MEASURE 3: Safe normalization of perpendicular vector
    perp_norm = jnp.linalg.norm(perp, axis=1, keepdims=True)
    perp_norm_safe = jnp.maximum(perp_norm, 1e-6)
    perp_normalized = perp / perp_norm_safe
    
    # Set velocities
    velocities = v_circ[:, None] * perp_normalized
    
    # SAFETY MEASURE 4: Zero velocity for particles too close to center or with problematic geometry
    is_too_close = r < r_min
    has_tiny_perp = jnp.linalg.norm(perp, axis=1) < 1e-6
    has_extreme_velocity = v_circ > v_max * 0.99
    is_problematic = is_too_close | has_tiny_perp | has_extreme_velocity
    
    # Apply safety condition: zero velocity for problematic particles
    velocities = jnp.where(
        is_problematic[:, None],
        jnp.zeros_like(velocities),
        velocities
    )
    
    return velocities

    

def initialize_circular_velocities_nfw(positions, vel_params, G, m_part, pos_params):
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
    
    # SAFETY MEASURE 1: Safe guard against very small radii
    r_min = 0.01 * rs  # Scale with NFW scale radius
    r_max = c * rs     # Virial radius
    r_safe = jnp.clip(r, r_min, r_max)

    from utils import blob_enclosed_mass_nfw
    M_enclosed = blob_enclosed_mass_nfw(r_safe, M, rs, c)
    
    # SAFETY MEASURE: Ensure minimum enclosed mass
    M_enclosed = jnp.maximum(M_enclosed, 0.01 * M)
    
    v_circ = jnp.sqrt(G * M_enclosed / r_safe) * vel_factor
    
    # SAFETY MEASURE 2: Cap maximum velocity
    v_max = 10.0
    v_circ = jnp.minimum(v_circ, v_max)

    z_axis = jnp.array([0.0, 0.0, 1.0])
    perp = jnp.cross(rel_pos, z_axis)
    
    # SAFETY MEASURE 3: Safe normalization of perpendicular vector
    perp_norm = jnp.linalg.norm(perp, axis=1, keepdims=True)
    perp_norm_safe = jnp.maximum(perp_norm, 1e-6)
    perp_normalized = perp / perp_norm_safe
    
    velocities = v_circ[:, None] * perp_normalized
    
    # SAFETY MEASURE 4: Zero velocity for particles too close to center or with problematic geometry
    is_too_close = r < r_min
    is_too_far = r > r_max
    has_tiny_perp = jnp.linalg.norm(perp, axis=1) < 1e-6
    has_extreme_velocity = v_circ > v_max * 0.99
    is_problematic = is_too_close | is_too_far | has_tiny_perp | has_extreme_velocity
    
    # Apply safety condition: zero velocity for problematic particles
    velocities = jnp.where(
        is_problematic[:, None],
        jnp.zeros_like(velocities),
        velocities
    )

    return velocities

# --- Combined initialization function ---

def initialize_blob(blob_params, length, G, m_part, key):
    """
    Initialize a single blob with given parameters.
    
    Parameters:
    -----------
    blob_params : dict
        Dictionary with blob parameters:
        - n_part: number of particles
        - pos_type: position distribution type ('gaussian' or 'nfw')
        - pos_params: position distribution parameters
        - vel_type: velocity distribution type ('cold', 'virial', or 'circular')
        - vel_params: velocity distribution parameters
    length : float
        Size of the simulation box
    G : float
        Gravitational constant
    m_part : float
        Mass per particle
    key : jax.random.PRNGKey
        Random key for initialization
        
    Returns:
    --------
    positions : jnp.array
        Array of shape (n_part, 3) with particle positions
    velocities : jnp.array
        Array of shape (n_part, 3) with particle velocities
    """
    n_part = blob_params['n_part']
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
    else:
        raise ValueError(f"Unknown position type: {pos_type}")
    
    if vel_type == 'cold':
        velocities = initialize_cold_velocities(positions, vel_params, m_part)
    elif vel_type == 'virial':
        velocities = initialize_virial_velocities(positions, vel_params, G, m_part)
    elif vel_type == 'circular':
        distrib = vel_params.get('distrib', False)
        # For circular velocities, we use the specific function for the position distribution
        if distrib:
            if pos_type == 'gaussian':
                velocities = initialize_circular_velocities_gaussian(positions, vel_params, G, m_part, pos_params)
            
            elif pos_type == 'nfw':
                velocities = initialize_circular_velocities_nfw(positions, vel_params, G, m_part, pos_params)
        else:
            velocities = initialize_circular_velocities_proxy(positions, vel_params, G, m_part)
    else:
        raise ValueError(f"Unknown velocity type: {vel_type}")
    
    return positions, velocities

def initialize_blobs(blobs_params, length, G, m_part, key=None):
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
    m_part : float
        Mass per particle
    key : jax.random.PRNGKey, optional
        Random key for initialization
        
    Returns:
    --------
    positions : jnp.array
        Array of shape (total_n_part, 3) with particle positions
    velocities : jnp.array
        Array of shape (total_n_part, 3) with particle velocities
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    # Split the key for each blob
    keys = jax.random.split(key, len(blobs_params))
    
    # Initialize each blob
    all_positions = []
    all_velocities = []
    for i, blob_params in enumerate(blobs_params):
        pos, vel = initialize_blob(blob_params, length, G, m_part, keys[i])
        all_positions.append(pos)
        all_velocities.append(vel)
    
    # Concatenate all positions and velocities
    positions = jnp.concatenate(all_positions, axis=0)
    velocities = jnp.concatenate(all_velocities, axis=0)
    
    return positions, velocities, blobs_params

