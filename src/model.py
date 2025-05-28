# src/model.py

import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, LeapfrogMidpoint, SaveAt
from jaxpm.painting import cic_paint

def pairwise_forces(pos, G, softening, length):
    # dx_ij = pos_i - pos_j with minimal-image, branchless
    dx = pos[:,None,:] - pos[None,:,:] # (N,N,3)
    dx = dx - length*jnp.round(dx/length) # periodic, no comparison so better for XLA
    r2 = jnp.sum(dx**2, axis=-1) + softening**2 # (N,N)
    inv_r3 = jnp.where(jnp.eye(pos.shape[0]), 0., r2**-1.5)
    F = jnp.einsum('ij,ijc->ic', inv_r3, dx)  # (N,3)
    F = -G*F
    return F

def make_diffrax_ode(softening, G, length):
    def nbody_ode(t, state, args):
        pos, vel = state # tuple (position, velocities)
        forces = pairwise_forces(pos, G=G, softening=softening, length=length)
        dpos = vel #drift
        dvel = forces #kick
        return jnp.stack([dpos, dvel])
    return nbody_ode

def gaussian_model(
    parameters,
    n_part=1000,
    G=5.0,
    length=64,
    softening=0.1,
    t_f=1.0,
    dt=0.5,
    ts=None,
    key=None
):
    """
    Generate simulated output density field for given Gaussian ICs.
    Parameters:
        parameters: array-like, [sigma, mean, vel_sigma]
        n_part: int, number of particles
        grid_shape: tuple, shape of density field
        G: float, gravitational constant
        length: float, box size
        softening: float, softening length
        t_f: float, final time
        dt: float, time step
        ts: 1D array, time points to save at (optional)
        key: jax.random.PRNGKey, random key (optional)
    Returns:
        output_field: jnp.array, density field after simulation
    """
    sigma, mean, vel_sigma = parameters
    grid_shape = (length, length, length)

    # Random keys for positions/velocities (if key is None, use default)
    if key is None:
        key = jax.random.PRNGKey(0)
    pos_key, vel_key = jax.random.split(key)

    init_pos = jax.random.normal(pos_key, (n_part, 3)) * sigma + mean
    init_vel = jax.random.normal(vel_key, (n_part, 3)) * vel_sigma
    input_field = cic_paint(jnp.zeros(grid_shape), init_pos)
    y0 = jnp.stack([init_pos, init_vel], axis=0)

    if ts is None:
        ts = jnp.linspace(0, t_f, int(t_f // dt) + 1)

    term = ODETerm(make_diffrax_ode(softening=softening, G=G, length=length))
    solver = LeapfrogMidpoint()
    sol = diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=t_f,
        dt0=dt,
        y0=y0,
        saveat=SaveAt(ts=ts)
    )
    final_pos = sol.ys[-1, 0]
    output_field = cic_paint(jnp.zeros(grid_shape), final_pos)
    return input_field, init_pos, final_pos, output_field, sol
