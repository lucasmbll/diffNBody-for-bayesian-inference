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
    params,
    fixed_params,
    other_params,
    params_infos,
    G,
    length,
    softening,
    t_f,
    dt,
    key,
    density_scaling,
    velocity_scaling,
    solver="LeapfrogMidpoint",
):
    grid_shape = (length, length, length)
    # Initialize positions, velocities, and masses for all blobs    
    init_pos, init_vel, masses = initialize_blobs(params, fixed_params, other_params, params_infos, length, G, softening, key)
    
    # Create density fields (weighted by mass for proper density)
    initial_density_field = cic_paint(jnp.zeros(grid_shape), init_pos, weight=masses)
    initial_density_field = apply_density_scaling(initial_density_field, density_scaling)

    # Create velocity fields (weighted by mass and initial velocity)
    initial_vx_field = cic_paint(jnp.zeros(grid_shape), init_pos, weight=masses * init_vel[:, 0])
    initial_vx_field = apply_density_scaling(initial_vx_field, velocity_scaling)
    
    initial_vy_field = cic_paint(jnp.zeros(grid_shape), init_pos, weight=masses * init_vel[:, 1])
    initial_vy_field = apply_density_scaling(initial_vy_field, velocity_scaling)

    initial_vz_field = cic_paint(jnp.zeros(grid_shape), init_pos, weight=masses * init_vel[:, 2])
    initial_vz_field = apply_density_scaling(initial_vz_field, velocity_scaling)

    initial_phase_field = jnp.stack([initial_density_field, initial_vx_field, initial_vy_field, initial_vz_field], axis=-1)    
    

    y0 = jnp.stack([init_pos, init_vel], axis=0)
    
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
    final_vel = sol.ys[-1, 1]
    
    
    final_density_field = cic_paint(jnp.zeros(grid_shape), final_pos, weight=masses)
    final_density_field = apply_density_scaling(final_density_field, density_scaling)
    
    # Apply scaling to both input and output fields
    
    final_vx_field = cic_paint(jnp.zeros(grid_shape), final_pos, weight=masses * final_vel[:, 0])
    final_vx_field = apply_density_scaling(final_vx_field, velocity_scaling)
    final_vy_field = cic_paint(jnp.zeros(grid_shape), final_pos, weight=masses * final_vel[:, 1])
    final_vy_field = apply_density_scaling(final_vy_field, velocity_scaling)
    final_vz_field = cic_paint(jnp.zeros(grid_shape), final_pos, weight=masses * final_vel[:, 2])
    final_vz_field = apply_density_scaling(final_vz_field, velocity_scaling)

    final_phase_field = jnp.stack([final_density_field, final_vx_field, final_vy_field, final_vz_field], axis=-1)
    
    return initial_phase_field, final_phase_field, sol.ts, sol.ys, masses

