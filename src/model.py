# src/model.py

import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, LeapfrogMidpoint, SaveAt
from jaxpm.painting import cic_paint
from jax.scipy.special import erf

def pairwise_forces(pos, G, softening, length, m_part):
    # dx_ij = pos_i - pos_j with minimal-image, branchless
    dx = pos[:,None,:] - pos[None,:,:] # (N,N,3)
    dx = dx - length*jnp.round(dx/length) # periodic, no comparison so better for XLA
    r2 = jnp.sum(dx**2, axis=-1) + softening**2 # (N,N)
    inv_r3 = jnp.where(jnp.eye(pos.shape[0]), 0., r2**-1.5)
    F = jnp.einsum('ij,ijc->ic', inv_r3, dx)  # (N,3)
    F = -G * F * m_part
    return F

def make_diffrax_ode(softening, G, length, m_part):
    def nbody_ode(t, state, args):
        pos, vel = state # tuple (position, velocities)
        forces = pairwise_forces(pos, G=G, softening=softening, length=length, m_part=m_part)
        dpos = vel #drift
        dvel = forces / m_part #kick
        return jnp.stack([dpos, dvel])
    return nbody_ode

def blob_enclosed_mass_gaussian(r, M, sigma):
    x = r / (jnp.sqrt(2) * sigma)
    term1 = erf(x)
    term2 = jnp.sqrt(2 / jnp.pi) * x * jnp.exp(-x**2)
    return M * (term1 - term2)

def blob_circular_velocity_gaussian(pos, G, M, sigma): # circular velocity in the x-y plane
    center = jnp.mean(pos, axis=0)
    rel_pos = pos - center
    r = jnp.linalg.norm(rel_pos, axis=1) + 1e-8
    M_enclosed = blob_enclosed_mass_gaussian(r, M, sigma)
    v_circ = jnp.sqrt(G * M_enclosed / r)
    # Compute direction perpendicular to radius vector using z-axis
    z_axis = jnp.array([0.0, 0.0, 1.0])
    perp = jnp.cross(rel_pos, z_axis)
    perp /= jnp.linalg.norm(perp, axis=1, keepdims=True) + 1e-8
    return v_circ[:, None] * perp  # shape (N, 3)

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
        if field_range > 0:
            return (density_field - field_min) / field_range
        else:
            return density_field
    
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

def gaussian_model(
    parameters,
    n_part,
    G,
    length,
    softening,
    t_f,
    dt,
    m_part=1.0,  # Mass per particle
    ts=None,
    key=None,
    random_vel=True,  # New parameter to control random velocity initialization
    density_scaling="none",  # New parameter for density field scaling
    **scaling_kwargs  # Additional scaling parameters
):
    """
    Generate simulated output density field for given Gaussian ICs.
    Parameters:
        parameters: array-like, [sigma, mean, vel_sigma] or [sigma, mean, vel_sigma, v_circ]
        n_part: int, number of particles
        grid_shape: tuple, shape of density field
        G: float, gravitational constant
        length: float, box size
        softening: float, softening length
        t_f: float, final time
        dt: float, time step
        ts: 1D array, time points to save at (optional)
        key: jax.random.PRNGKey, random key (optional)
        v_circ: float, circular velocity (optional, can be passed as kwarg)
        random_vel: bool, whether to add random velocity component (default: True)
        density_scaling: str, type of scaling to apply to density fields
        **scaling_kwargs: additional parameters for scaling
    Returns:
        input_field: jnp.array, scaled input density field
        init_pos: jnp.array, initial positions
        final_pos: jnp.array, final positions  
        output_field: jnp.array, scaled output density field
        sol: diffrax solution object
    """
    
    sigma, mean, vel_sigma = parameters
    
    grid_shape = (length, length, length)

    # Random keys for positions/velocities (if key is None, use default)
    if key is None:
        key = jax.random.PRNGKey(0)
    pos_key, vel_key = jax.random.split(key)

    init_pos = jax.random.normal(pos_key, (n_part, 3)) * sigma + mean

    # Initialize velocities based on random_vel parameter
    if random_vel:
        # Traditional: random Gaussian velocities + circular component
        init_vel = jax.random.normal(vel_key, (n_part, 3)) * vel_sigma
    else:
        # Pure circular: no random component, only circular velocity
        init_vel = jnp.zeros((n_part, 3))

    
    M = n_part * m_part
    circ_vel = blob_circular_velocity_gaussian(init_pos, G, M, sigma)
        
    init_vel = init_vel + circ_vel

    # Create raw density fields
    raw_input_field = cic_paint(jnp.zeros(grid_shape), init_pos)
    y0 = jnp.stack([init_pos, init_vel], axis=0)

    if ts is None:
        ts = jnp.linspace(0, t_f, int(t_f // dt) + 1)

    # Calculate max_steps with safety factor
    estimated_steps = int((t_f - 0.0) / dt)
    max_steps = max(estimated_steps * 2, 10000)  # 2x safety factor, minimum 10k steps

    term = ODETerm(make_diffrax_ode(softening=softening, G=G, length=length, m_part=m_part))
    solver = LeapfrogMidpoint()
    sol = diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=t_f,
        dt0=dt,
        y0=y0,
        saveat=SaveAt(ts=ts),
        max_steps=max_steps
    )
    final_pos = sol.ys[-1, 0]
    raw_output_field = cic_paint(jnp.zeros(grid_shape), final_pos)
    
    # Apply scaling to both input and output fields
    input_field = apply_density_scaling(raw_input_field, density_scaling, **scaling_kwargs)
    output_field = apply_density_scaling(raw_output_field, density_scaling, **scaling_kwargs)
    
    return input_field, init_pos, final_pos, output_field, sol

def gaussian_2blobs(
    parameters,
    n_part,
    G,
    length,
    softening,
    t_f,
    dt,
    m_part=1.0,
    ts=None,
    key=None,
    random_vel=True,  # New parameter to control random velocity initialization
    density_scaling="none",  # New parameter for density field scaling
    **scaling_kwargs  # Additional scaling parameters
):
    """
    Generate simulated output density field for two Gaussian blobs as ICs.
    Parameters:
        parameters: array-like, [sigma1, mean1, sigma2, mean2, vel_sigma]
        n_part: int, number of particles
        random_vel: bool, whether to add random velocity component (default: True)
        density_scaling: str, type of scaling to apply to density fields
        **scaling_kwargs: additional parameters for scaling
        ...
    """

    sigma1, mean1, sigma2, mean2, vel_sigma = parameters
    
    grid_shape = (length, length, length)
    if key is None:
        key = jax.random.PRNGKey(0)
    key1, key2, vel_key = jax.random.split(key, 3)
    
    n1 = n_part // 2
    n2 = n_part - n1
    pos1 = jax.random.normal(key1, (n1, 3)) * sigma1 + mean1
    pos2 = jax.random.normal(key2, (n2, 3)) * sigma2 + mean2
    
    # Initialize velocities based on random_vel parameter
    if random_vel:
        vel1 = jax.random.normal(vel_key, (n1, 3)) * vel_sigma
        vel2 = jax.random.normal(vel_key, (n2, 3)) * vel_sigma
    else:
        # Pure circular: no random component, only circular velocity
        vel1 = jnp.zeros((n1, 3))
        vel2 = jnp.zeros((n2, 3))
    
    # Add circular velocities for each blob separately
    n1_part = pos1.shape[0]  # Number of particles in blob 1 
    M1 = n1_part * m_part   # Mass of blob 1
    circ_vel_1 = blob_circular_velocity_gaussian(pos1, G, M1, sigma1)   
    vel1 += circ_vel_1
    
    n2_part = pos2.shape[0]  # Number of particles in blob 2 
    M2 = n2_part * m_part   # Mass of blob 2
    circ_vel_2 = blob_circular_velocity_gaussian(pos2, G, M2, sigma2)   
    vel2 += circ_vel_2
    
    init_pos = jnp.concatenate([pos1, pos2], axis=0)
    init_vel = jnp.concatenate([vel1, vel2], axis=0)
    
    # Create raw density fields
    raw_input_field = cic_paint(jnp.zeros(grid_shape), init_pos)
    y0 = jnp.stack([init_pos, init_vel], axis=0) 
    
    if ts is None:
        ts = jnp.linspace(0, t_f, int(t_f // dt) + 1)
       
    # Calculate max_steps with safety factor
    estimated_steps = int((t_f - 0.0) / dt)
    max_steps = max(estimated_steps * 2, 10000)  # 2x safety factor, minimum 10k steps
       
    term = ODETerm(make_diffrax_ode(softening=softening, G=G, length=length, m_part=m_part))
    solver = LeapfrogMidpoint()
    sol = diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=t_f,
        dt0=dt,
        y0=y0,
        saveat=SaveAt(ts=ts),
        max_steps=max_steps
    )
    final_pos = sol.ys[-1, 0]
    raw_output_field = cic_paint(jnp.zeros(grid_shape), final_pos)

    # Apply scaling to both input and output fields
    input_field = apply_density_scaling(raw_input_field, density_scaling, **scaling_kwargs)
    output_field = apply_density_scaling(raw_output_field, density_scaling, **scaling_kwargs)
    return input_field, init_pos, final_pos, output_field, sol


