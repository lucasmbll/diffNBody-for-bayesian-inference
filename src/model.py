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
        parameters: array-like, [sigma, mean, vel_sigma, v_circ]
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
    # Unpack parameters
    if len(parameters) == 3:
        sigma, mean, vel_sigma = parameters
        v_circ = 0.0
    else:
        sigma, mean, vel_sigma, v_circ = parameters
    grid_shape = (length, length, length)

    # Random keys for positions/velocities (if key is None, use default)
    if key is None:
        key = jax.random.PRNGKey(0)
    pos_key, vel_key = jax.random.split(key)

    init_pos = jax.random.normal(pos_key, (n_part, 3)) * sigma + mean
    init_vel = jax.random.normal(vel_key, (n_part, 3)) * vel_sigma

    # Add circular velocity in the x-y plane around the mean
    if v_circ != 0.0:
        rel_pos = init_pos - mean
        r = jnp.linalg.norm(rel_pos[:, :2], axis=1, keepdims=True) + 1e-8
        # Perpendicular direction in x-y plane
        perp = jnp.concatenate([-rel_pos[:, 1:2], rel_pos[:, 0:1]], axis=1)
        perp = perp / r
        circ_vel = jnp.zeros_like(init_vel).at[:, :2].set(perp * v_circ)
        init_vel = init_vel + circ_vel

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

def gaussian_2blobs(
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
    Generate simulated output density field for two Gaussian blobs as ICs.
    Parameters:
        parameters: array-like, [sigma1, mean1, sigma2, mean2, vel_sigma, v_circ1, v_circ2]
        n_part: int, number of particles
        ...
    """
    # Unpack parameters
    if len(parameters) == 5:
        sigma1, mean1, sigma2, mean2, vel_sigma = parameters
        v_circ1 = 0.0
        v_circ2 = 0.0
    else:
        sigma1, mean1, sigma2, mean2, vel_sigma, v_circ1, v_circ2 = parameters
    grid_shape = (length, length, length)
    if key is None:
        key = jax.random.PRNGKey(0)
    key1, key2, vel_key = jax.random.split(key, 3)
    n1 = n_part // 2
    n2 = n_part - n1
    pos1 = jax.random.normal(key1, (n1, 3)) * sigma1 + mean1
    pos2 = jax.random.normal(key2, (n2, 3)) * sigma2 + mean2
    # Random velocities
    vel1 = jax.random.normal(vel_key, (n1, 3)) * vel_sigma
    vel2 = jax.random.normal(vel_key, (n2, 3)) * vel_sigma
    # Add circular velocity for blob 1 (x-y plane)
    if v_circ1 != 0.0:
        rel1 = pos1 - mean1
        r1 = jnp.linalg.norm(rel1[:, :2], axis=1, keepdims=True) + 1e-8
        perp1 = jnp.concatenate([-rel1[:, 1:2], rel1[:, 0:1]], axis=1)
        perp1 = perp1 / r1
        circ1 = jnp.zeros_like(vel1).at[:, :2].set(perp1 * v_circ1)
        vel1 = vel1 + circ1
    # Add circular velocity for blob 2 (x-y plane)
    if v_circ2 != 0.0:
        rel2 = pos2 - mean2
        r2 = jnp.linalg.norm(rel2[:, :2], axis=1, keepdims=True) + 1e-8
        perp2 = jnp.concatenate([-rel2[:, 1:2], rel2[:, 0:1]], axis=1)
        perp2 = perp2 / r2
        circ2 = jnp.zeros_like(vel2).at[:, :2].set(perp2 * v_circ2)
        vel2 = vel2 + circ2
    init_pos = jnp.concatenate([pos1, pos2], axis=0)
    init_vel = jnp.concatenate([vel1, vel2], axis=0)
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

