# src/model.py

import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, LeapfrogMidpoint, SaveAt
from jaxpm.painting import cic_paint
from initialization import initialize_blobs
from utils import apply_density_scaling

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

def model(
    blobs_params,
    G,
    length,
    softening,
    t_f,
    dt,
    m_part,
    ts=None,
    key=None,
    density_scaling="none",
    **scaling_kwargs
):
    """
    Unified model function that uses blob parameters to initialize and simulate N-body system.
    
    Parameters:
    -----------
    blobs_params : list of dict
        List of dictionaries with blob parameters
    G : float
        Gravitational constant
    length : float
        Box size
    softening : float
        Softening length
    t_f : float
        Final time
    dt : float
        Time step
    m_part : float
        Mass per particle
    ts : array-like, optional
        Time points to save at
    key : jax.random.PRNGKey, optional
        Random key for initialization
    density_scaling : str
        Type of scaling to apply to density fields
    **scaling_kwargs : dict
        Additional parameters for scaling
        
    Returns:
    --------
    input_field : jnp.array
        Scaled input density field
    init_pos : jnp.array
        Initial positions
    final_pos : jnp.array
        Final positions
    output_field : jnp.array
        Scaled output density field
    sol : diffrax solution object
        Solution of the N-body simulation
    init_params : list of dict
        Initial parameters used for the blobs (for inference)
    """
    grid_shape = (length, length, length)

    # Initialize positions and velocities for all blobs
    if key is None:
        key = jax.random.PRNGKey(0)
    
    init_pos, init_vel, init_params = initialize_blobs(blobs_params, length, G, m_part, key)
    
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
    
    return input_field, init_pos, final_pos, output_field, sol, init_params

