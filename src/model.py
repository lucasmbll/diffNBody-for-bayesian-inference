# src/model.py

import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt
from jaxpm.painting import cic_paint
from initialization import initialize_blobs
from utils import apply_density_scaling

def pairwise_forces(pos, masses, G, softening, length):
    """
    Calculate pairwise forces with per-particle masses.
    
    Parameters:
    -----------
    pos : jnp.array
        Particle positions (N, 3)
    masses : jnp.array
        Particle masses (N,)
    G : float
        Gravitational constant
    softening : float
        Softening length
    length : float
        Box size for periodic boundaries
        
    Returns:
    --------
    F : jnp.array
        Forces on each particle (N, 3)
    """
    # dx_ij = pos_i - pos_j with minimal-image, branchless
    dx = pos[:,None,:] - pos[None,:,:] # (N,N,3)
    dx = dx - length*jnp.round(dx/length) # periodic, no comparison so better for XLA
    r2 = jnp.sum(dx**2, axis=-1) + softening**2 # (N,N)
    inv_r3 = jnp.where(jnp.eye(pos.shape[0]), 0., r2**-1.5)
    
    # Include masses in force calculation: F_ij = -G * m_i * m_j * r_ij / |r_ij|^3
    mass_matrix = masses[:, None] * masses[None, :]  # (N, N)
    F = jnp.einsum('ij,ijc->ic', -G * mass_matrix * inv_r3, dx)  # (N,3)
    
    return F

def make_diffrax_ode(softening, G, length, masses):
    @jax.jit
    def nbody_ode(t, state, args):
        pos, vel = state # tuple (position, velocities)
        forces = pairwise_forces(pos, masses, G=G, softening=softening, length=length)
        dpos = vel #drift
        dvel = forces / masses[:, None] #kick (divide by mass for each particle)
        return jnp.stack([dpos, dvel])
    return nbody_ode

def model(
    blobs_params,
    G,
    length,
    softening,
    t_f,
    dt,
    ts=None,
    key=None,
    density_scaling="none",
    solver="LeapfrogMidpoint",
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
    masses : jnp.array
        Particle masses
    """
    grid_shape = (length, length, length)

    # Initialize positions, velocities, and masses for all blobs
    if key is None:
        key = jax.random.PRNGKey(0)
    
    init_pos, init_vel, masses = initialize_blobs(blobs_params, length, G, softening, key)
    
    # Create raw density fields (weighted by mass for proper density)
    raw_input_field = cic_paint(jnp.zeros(grid_shape), init_pos, weight=masses)
    y0 = jnp.stack([init_pos, init_vel], axis=0)

    if ts is None:
        ts = jnp.linspace(0, t_f, int(t_f // dt) + 1)

    # Calculate max_steps with safety factor
    estimated_steps = int((t_f - 0.0) / dt)
    max_steps = max(estimated_steps * 2, 10000)  # 2x safety factor, minimum 10k steps

    term = ODETerm(make_diffrax_ode(softening=softening, G=G, length=length, masses=masses))

    if solver == "LeapfrogMidpoint":
        from diffrax import LeapfrogMidpoint
        solver = LeapfrogMidpoint()
    elif solver == "Dopri5":
        from diffrax import Dopri5
        solver = Dopri5()
    elif solver == "Dopri8":
        from diffrax import Dopri8
        solver = Dopri8()
    elif solver == "Tsit5":
        from diffrax import Tsit5
        solver = Tsit5()
    elif solver == "Heun":
        from diffrax import Heun
        solver = Heun()
    elif solver == "Midpoint":
        from diffrax import Midpoint
        solver = Midpoint()
    elif solver == "Euler":
        from diffrax import Euler
        solver = Euler()
    else:
        raise ValueError(f"Unknown solver: {solver}")
    
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
    raw_output_field = cic_paint(jnp.zeros(grid_shape), final_pos, weight=masses)
    
    # Apply scaling to both input and output fields
    input_field = apply_density_scaling(raw_input_field, density_scaling, **scaling_kwargs)
    output_field = apply_density_scaling(raw_output_field, density_scaling, **scaling_kwargs)
    
    return input_field, init_pos, final_pos, output_field, sol, masses

