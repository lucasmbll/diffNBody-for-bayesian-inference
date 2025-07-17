# utils.py

import jax
import jax.numpy as jnp
from jax.scipy.special import erf
import numpy as np
from functools import partial

def calculate_energy_variable_mass(pos, vel, masses, G, length, softening):
    """Calculate kinetic, potential, and total energy of the system with variable masses"""
    n_particles = pos.shape[0]
    # Kinetic energy (each particle has its own mass)
    ke = 0.5 * jnp.sum(masses * jnp.sum(vel**2, axis=1))
    # Potential energy with periodic boundaries and variable masses
    dx = pos[:, None, :] - pos[None, :, :]
    dx = dx - length * jnp.round(dx / length)  # periodic boundaries
    r2 = jnp.sum(dx**2, axis=-1) + softening**2  # softening squared
    r = jnp.sqrt(r2)
    # Mass matrix for pairwise interactions
    mass_matrix = masses[:, None] * masses[None, :]
    # Upper triangular part to avoid double counting, exclude diagonal
    mask = jnp.triu(jnp.ones((n_particles, n_particles)), k=1)
    pe = -G * jnp.sum(mask * mass_matrix / r)
    total_energy = ke + pe
    return ke, pe, total_energy

def blob_enclosed_mass_gaussian(r, M, sigma):
    x = r / (jnp.sqrt(2) * sigma)
    term1 = erf(x)
    term2 = jnp.sqrt(2 / jnp.pi) * x * jnp.exp(-x**2)
    return M * (term1 - term2)

def blob_enclosed_mass_nfw(r, M, rs, c):
        x = r / rs
        norm = jnp.log(1 + c) - c / (1 + c)
        enclosed = (jnp.log(1 + x) - x / (1 + x)) / norm
        # Ensure no negative or nan values for r=0
        enclosed = jnp.where(r > 0, enclosed, 0.0)
        return M * enclosed

def blob_enclosed_mass_plummer(r, M, rs):
    x = r / rs
    enclosed = x**3 / (1 + x**2)**(3.0/2.0)
    # Ensure no negative values for r=0
    enclosed = jnp.where(r > 0, enclosed, 0.0)
    return M * enclosed


@partial(jax.jit, static_argnames=['scaling_type'])
def apply_density_scaling(density_field, scaling_type="none"):
    if scaling_type == "none":
        # No scaling: f(x) = x
        return density_field
    
    elif scaling_type == "log":
        return jnp.log(density_field + 1e-8)

    elif scaling_type == "asinh":
        return jnp.asinh(density_field)
    
    elif scaling_type == "signed_log":
        return jnp.sign(density_field) * jnp.log(jnp.abs(density_field) + 1e-8)
    
    elif scaling_type == "sqrt":
        # Square root scaling: f(x) = sqrt(x)
        return jnp.sqrt(density_field)
    
    else:
        raise ValueError(f"Unknown scaling_type: {scaling_type}. Available options: "
                        "'none', 'log', 'sqrt', 'asinh', 'signed_log'.")
    
# We use the approximation from Robotham & Howlett (2018)
@partial(jax.jit, static_argnames=['x'])
def approx_inverse_nfw_cdf(x, rs, c):
    p = 0.3333 - 0.2*jnp.log10(c) + 0.01*c
    a = 1.0 / c 
    q = x * (1 - a) + a
    r = rs * (q**(-1.0/p) - 1.0) / (1.0 - q**(-1.0/p) / c)
    return r


def smooth_trajectory(traj, window):
        if window < 2:
            return traj
        kernel = np.ones(window) / window
        traj_padded = np.pad(traj, ((window//2, window-1-window//2), (0,0)), mode='edge')
        smoothed = np.vstack([
            np.convolve(traj_padded[:, dim], kernel, mode='valid')
            for dim in range(traj.shape[1])
        ]).T
        return smoothed

def extract_blob_parameters(blob_config, mode, infered_params):
    # Initialize parameter collectors
    changing_params = []
    fixed_params = []
    other_params = {}
    param_info = {'fixed_param_order': [], 'changing_param_order': []}
    
    # Process position parameters
    pos_params = blob_config['pos_params']
    pos_type = blob_config['pos_type']
    other_params['pos_type'] = pos_type
    
    if pos_type == 'gaussian':
        # Handle center - support both single scalar and 3D coordinate
        if mode in ['sampling', 'mle', 'grid'] and 'center' in infered_params:
            center_param = pos_params['center']
            if isinstance(center_param, (list, tuple)) and len(center_param) == 3:
                # Traditional 3D format: [x, y, z]
                changing_params.extend(center_param)
                param_info['changing_param_order'].extend(['center_x', 'center_y', 'center_z'])
            else:
                # Single scalar format: X -> [X, X, X]
                changing_params.append(center_param)
                param_info['changing_param_order'].append('center')
        else:
            center_param = pos_params['center']
            if isinstance(center_param, (list, tuple)) and len(center_param) == 3:
                fixed_params.extend(center_param)
                param_info['fixed_param_order'].extend(['center_x', 'center_y', 'center_z'])
            else:
                fixed_params.append(center_param)
                param_info['fixed_param_order'].append('center')
            
        # Handle sigma
        if mode in ['sampling', 'mle', 'grid'] and 'sigma' in infered_params:
                changing_params.append(pos_params['sigma'])
                param_info['changing_param_order'].append('sigma')
        else:
            fixed_params.append(pos_params['sigma'])
            param_info['fixed_param_order'].append('sigma')
    
    elif pos_type == 'nfw':
        # Handle scale radius and concentration
        if mode in ['sampling', 'mle', 'grid'] and 'rs' in infered_params:
                changing_params.append(pos_params['rs'])
                param_info['changing_param_order'].append('rs')
        else:
            fixed_params.append(pos_params['rs'])
            param_info['fixed_param_order'].append('rs')

        if mode in ['sampling', 'mle', 'grid'] and 'c' in infered_params:
                changing_params.append(pos_params['c'])
                param_info['changing_param_order'].append('c')
        else:
            fixed_params.append(pos_params['c'])
            param_info['fixed_param_order'].append('c')
        
        # Handle center - support both single scalar and 3D coordinate
        if mode in ['sampling', 'mle', 'grid'] and 'center' in infered_params:
            center_param = pos_params['center']
            if isinstance(center_param, (list, tuple)) and len(center_param) == 3:
                # Traditional 3D format: [x, y, z]
                changing_params.extend(center_param)
                param_info['changing_param_order'].extend(['center_x', 'center_y', 'center_z'])
            else:
                # Single scalar format: X -> [X, X, X]
                changing_params.append(center_param)
                param_info['changing_param_order'].append('center')
        else:
            center_param = pos_params['center']
            if isinstance(center_param, (list, tuple)) and len(center_param) == 3:
                fixed_params.extend(center_param)
                param_info['fixed_param_order'].extend(['center_x', 'center_y', 'center_z'])
            else:
                fixed_params.append(center_param)
                param_info['fixed_param_order'].append('center')

    elif pos_type == 'plummer':
        # Handle scale radius
        if mode in ['sampling', 'mle', 'grid'] and 'rs' in infered_params:
                changing_params.append(pos_params['rs'])
                param_info['changing_param_order'].append('rs')
        else:
            fixed_params.append(pos_params['rs'])
            param_info['fixed_param_order'].append('rs')
        
        # Handle center - support both single scalar and 3D coordinate
        if mode in ['sampling', 'mle', 'grid'] and 'center' in infered_params:
            center_param = pos_params['center']
            if isinstance(center_param, (list, tuple)) and len(center_param) == 3:
                # Traditional 3D format: [x, y, z]
                changing_params.extend(center_param)
                param_info['changing_param_order'].extend(['center_x', 'center_y', 'center_z'])
            else:
                # Single scalar format: X -> [X, X, X]
                changing_params.append(center_param)
                param_info['changing_param_order'].append('center')
        else:
            center_param = pos_params['center']
            if isinstance(center_param, (list, tuple)) and len(center_param) == 3:
                fixed_params.extend(center_param)
                param_info['fixed_param_order'].extend(['center_x', 'center_y', 'center_z'])
            else:
                fixed_params.append(center_param)
                param_info['fixed_param_order'].append('center')
    
    # Process velocity parameters
    vel_params = blob_config['vel_params']
    vel_type = blob_config['vel_type']
    other_params['vel_type'] = vel_type
    
    if vel_type == 'circular':
        if mode in ['sampling', 'mle', 'grid'] and 'vel_factor' in infered_params:
                changing_params.append(vel_params['vel_factor'])
                param_info['changing_param_order'].append('vel_factor')
        else:
            fixed_params.append(vel_params['vel_factor'])
            param_info['fixed_param_order'].append('vel_factor')
        other_params['distrib'] = vel_params['distrib']    
    
    # Add cold velocity case and virial velocity case
    # Add non-varying parameters
    other_params['n_part'] = blob_config['n_part']
    fixed_params.append(blob_config['m_part'])
    param_info['fixed_param_order'].append('m_part')
    
    return jnp.array(changing_params, dtype=jnp.float32), jnp.array(fixed_params, dtype=jnp.float32), other_params, param_info

def blobs_params_init(blobs_params, prior_params, initial_position, mode):
    all_params = []
    all_fixed = []
    all_other_params = []
    all_info = []
    
    for blob_idx, blob in enumerate(blobs_params):
        if mode == 'sampling':
            infered_params = extract_params_to_infer(blob, blob_idx, prior_params, initial_position)
        elif mode == 'mle':
            infered_params = extract_params_to_optimize(blob, blob_idx, initial_position)
        elif mode == 'grid': 
            infered_params = extract_params_to_evaluate(blob, blob_idx, prior_params)
        else:
            infered_params = None
        params, fixed, others, info = extract_blob_parameters(blob, mode, infered_params)
        all_params.append(params)
        all_fixed.append(fixed)
        all_other_params.append(others)
        all_info.append(info)
    
    # Stack all parameter arrays
    params = jnp.stack(all_params)
    fixed = jnp.stack(all_fixed)
    
    return params, fixed, all_other_params, all_info

def extract_params_to_infer(blob, blob_idx, prior_params, initial_position):
    # Initialize set for parameters to infer
    params_to_infer = set()
    
    # Position parameters
    if blob['pos_type'] == 'gaussian':
        sigma_key = f"blob{blob_idx}_sigma"
        center_key = f"blob{blob_idx}_center"  # Check single scalar first
        center_x_key = f"blob{blob_idx}_center_x"  # Then check 3D format
        
        if (sigma_key in prior_params and sigma_key in initial_position):
            params_to_infer.add('sigma')
        
        # Check for single scalar center parameter
        if (center_key in prior_params and center_key in initial_position):
            params_to_infer.add('center')
        # Check for 3D center parameter
        elif (center_x_key in prior_params and center_x_key in initial_position):
            params_to_infer.add('center')
            
    elif blob['pos_type'] == 'nfw':
        rs_key = f"blob{blob_idx}_rs"
        c_key = f"blob{blob_idx}_c"
        center_key = f"blob{blob_idx}_center"  # Check single scalar first
        center_x_key = f"blob{blob_idx}_center_x"
        
        if (rs_key in prior_params and 
            rs_key in initial_position):
            params_to_infer.add('rs')
            
        if (c_key in prior_params and 
            c_key in initial_position):
            params_to_infer.add('c')
            
        # Check for single scalar center parameter
        if (center_key in prior_params and center_key in initial_position):
            params_to_infer.add('center')
        # Check for 3D center parameter
        elif (center_x_key in prior_params and center_x_key in initial_position):
            params_to_infer.add('center')
            
    elif blob['pos_type'] == 'plummer':
        rs_key = f"blob{blob_idx}_rs"
        center_key = f"blob{blob_idx}_center"  # Check single scalar first
        center_x_key = f"blob{blob_idx}_center_x"
        
        if (rs_key in prior_params and 
            rs_key in initial_position):
            params_to_infer.add('rs')
            
        # Check for single scalar center parameter
        if (center_key in prior_params and center_key in initial_position):
            params_to_infer.add('center')
        # Check for 3D center parameter
        elif (center_x_key in prior_params and center_x_key in initial_position):
            params_to_infer.add('center')
    
    # Velocity parameters
    if blob['vel_type'] == 'circular':
        vel_key = f"blob{blob_idx}_vel_factor"
        if (vel_key in prior_params and 
            vel_key in initial_position):
            params_to_infer.add('vel_factor')
            
    elif blob['vel_type'] == 'cold':
        vel_key = f"blob{blob_idx}_vel_dispersion"
        if (vel_key in prior_params and 
            vel_key in initial_position):
            params_to_infer.add('vel_dispersion')
            
    elif blob['vel_type'] == 'virial':
        vel_key = f"blob{blob_idx}_virial_ratio"
        if (vel_key in prior_params and 
            vel_key in initial_position):
            params_to_infer.add('virial_ratio')
    
    return list(params_to_infer)

def extract_params_to_evaluate(blob, blob_idx, prior_params):
    # Initialize set for parameters to infer
    params_to_infer = set()
    
    # Position parameters
    if blob['pos_type'] == 'gaussian':
        sigma_key = f"blob{blob_idx}_sigma"
        center_key = f"blob{blob_idx}_center"  # Check single scalar first
        center_x_key = f"blob{blob_idx}_center_x"  # Then check 3D format
        
        if (sigma_key in prior_params):
            params_to_infer.add('sigma')
            
        if (center_key in prior_params):
            params_to_infer.add('center')

        elif (center_x_key in prior_params):
            params_to_infer.add('center')
            
    elif blob['pos_type'] == 'nfw':
        rs_key = f"blob{blob_idx}_rs"
        c_key = f"blob{blob_idx}_c"
        center_key = f"blob{blob_idx}_center"  # Check single scalar first
        center_x_key = f"blob{blob_idx}_center_x"  # Then check 3D format
        
        if (rs_key in prior_params):
            params_to_infer.add('rs')
            
        if (c_key in prior_params):
            params_to_infer.add('c')
            
        if (center_key in prior_params):
            params_to_infer.add('center')

        elif (center_x_key in prior_params):
            params_to_infer.add('center')
            
    elif blob['pos_type'] == 'plummer':
        rs_key = f"blob{blob_idx}_rs"
        center_key = f"blob{blob_idx}_center"  # Check single scalar first
        center_x_key = f"blob{blob_idx}_center_x"  # Then check 3D format
        
        if (rs_key in prior_params):
            params_to_infer.add('rs')
            
        if (center_key in prior_params):
            params_to_infer.add('center')

        elif (center_x_key in prior_params):
            params_to_infer.add('center')
    
    # Velocity parameters
    if blob['vel_type'] == 'circular':
        vel_key = f"blob{blob_idx}_vel_factor"
        if (vel_key in prior_params):
            params_to_infer.add('vel_factor')
            
    elif blob['vel_type'] == 'cold':
        vel_key = f"blob{blob_idx}_vel_dispersion"
        if (vel_key in prior_params):
            params_to_infer.add('vel_dispersion')
            
    elif blob['vel_type'] == 'virial':
        vel_key = f"blob{blob_idx}_virial_ratio"
        if (vel_key in prior_params):
            params_to_infer.add('virial_ratio')
    
    return list(params_to_infer)

def extract_params_to_optimize(blob, blob_idx, initial_position):
    # Initialize set for parameters to infer
    params_to_infer = set()
    
    # Position parameters
    if blob['pos_type'] == 'gaussian':
        sigma_key = f"blob{blob_idx}_sigma"
        center_key = f"blob{blob_idx}_center"  # Check single scalar first
        center_x_key = f"blob{blob_idx}_center_x"  # Then check 3D format
        
        if sigma_key in initial_position:
            params_to_infer.add('sigma')
            
        if center_key in initial_position:
            params_to_infer.add('center')
        
        elif center_x_key in initial_position:
            params_to_infer.add('center')
            
    elif blob['pos_type'] == 'nfw':
        rs_key = f"blob{blob_idx}_rs"
        c_key = f"blob{blob_idx}_c"
        center_key = f"blob{blob_idx}_center"  # Check single scalar first
        center_x_key = f"blob{blob_idx}_center_x"  # Then check 3D format
        
        if rs_key in initial_position:
            params_to_infer.add('rs')
            
        if c_key in initial_position:
            params_to_infer.add('c')
            
        if center_key in initial_position:
            params_to_infer.add('center')
        
        elif center_x_key in initial_position:
            params_to_infer.add('center')
            
    elif blob['pos_type'] == 'plummer':
        rs_key = f"blob{blob_idx}_rs"
        center_key = f"blob{blob_idx}_center"  # Check single scalar first
        center_x_key = f"blob{blob_idx}_center_x"  # Then check 3D format
        
        if rs_key in initial_position:
            params_to_infer.add('rs')
            
        if center_key in initial_position:
            params_to_infer.add('center')
        
        elif center_x_key in initial_position:
            params_to_infer.add('center')
    
    # Velocity parameters
    if blob['vel_type'] == 'circular':
        vel_key = f"blob{blob_idx}_vel_factor"
        if vel_key in initial_position:
            params_to_infer.add('vel_factor')
            
    elif blob['vel_type'] == 'cold':
        vel_key = f"blob{blob_idx}_vel_dispersion"
        if vel_key in initial_position:
            params_to_infer.add('vel_dispersion')
            
    elif blob['vel_type'] == 'virial':
        vel_key = f"blob{blob_idx}_virial_ratio"
        if vel_key in initial_position:
            params_to_infer.add('virial_ratio')
    
    return list(params_to_infer)

def params_init(params_infos, initial_position):
    """Initialize sampling parameters array from initial positions in config"""
    all_initial_params = []
    n_blobs = len(params_infos)
    params_order = params_infos[0]['changing_param_order']
    
    for blob_idx in range(n_blobs):
        blob_initial_params = []
        for param_name in params_order:
            if param_name == 'center':
                # Single scalar format
                key = f"blob{blob_idx}_center"
                if key in initial_position:
                    blob_initial_params.append(initial_position[key])
                else:
                    raise ValueError(f"Parameter {key} not found in initial_position.")
            elif param_name in ['center_x', 'center_y', 'center_z']:
                # Traditional 3D format
                key = f"blob{blob_idx}_{param_name}"
                if key in initial_position:
                    blob_initial_params.append(initial_position[key])
                else:
                    raise ValueError(f"Parameter {key} not found in initial_position.")
            else:
                # Other parameters
                key = f"blob{blob_idx}_{param_name}"
                if key in initial_position:
                    blob_initial_params.append(initial_position[key])
                else:
                    raise ValueError(f"Parameter {key} not found in initial_position.")
        
        all_initial_params.append(jnp.array(blob_initial_params, dtype=jnp.float32))
    
    return jnp.stack(all_initial_params)

def prior_params_extract(prior_type, prior_params, params_infos):
    """Extract prior parameters from prior_params dict based on params_infos"""
    all_prior_params = []
    params_labels = []
    n_blobs = len(params_infos)
    params_order = params_infos[0]['changing_param_order']
    for blob_idx in range(n_blobs):
        blob_prior_params = []
        for param_name in params_order:
            # Check if the parameter exists in initial_position
            if f"blob{blob_idx}_{param_name}" in prior_params:
                params_labels.append(f"blob{blob_idx}_{param_name}")
                if prior_type == "blob_gaussian":
                    # For Gaussian priors, we expect a dict with 'mu' and 'sigma'
                    if isinstance(prior_params[f"blob{blob_idx}_{param_name}"], dict):
                        blob_prior_params.append(prior_params[f"blob{blob_idx}_{param_name}"]['mu'])
                        blob_prior_params.append(prior_params[f"blob{blob_idx}_{param_name}"]['sigma'])
                    else:
                        raise ValueError(f"Expected Gaussian prior for blob{blob_idx}_{param_name}, but found {prior_params[f'blob{blob_idx}_{param_name}']}")
                elif prior_type == "blob_uniform":
                    # For Uniform priors, we expect a dict with 'low' and 'high'
                    if isinstance(prior_params[f"blob{blob_idx}_{param_name}"], dict):
                        blob_prior_params.append(prior_params[f"blob{blob_idx}_{param_name}"]['low'])
                        blob_prior_params.append(prior_params[f"blob{blob_idx}_{param_name}"]['high'])
                    else:
                        raise ValueError(f"Expected Uniform prior for blob{blob_idx}_{param_name}, but found {prior_params[f'blob{blob_idx}_{param_name}']}")
            else:
                raise ValueError(f"Parameter blob{blob_idx}_{param_name} not found in initial_position.")
        all_prior_params.append(jnp.array(blob_prior_params, dtype=jnp.float32))
    return jnp.stack(all_prior_params), params_labels


    ## TO BE CHECKED 


def tune_step_size(sampler_class, log_posterior, initial_position, rng_key, num_trials=200, step_sizes=None):
    import jax
    if step_sizes is None:
        step_sizes = np.logspace(-4, 0, 10)  # e.g. [1e-4, 1e-3, ..., 1.0]
    best_rate = -1
    best_step = None
    for step_size in step_sizes:
        sampler = sampler_class(log_posterior, step_size)
        kernel = jax.jit(sampler.step)
        state = sampler.init(initial_position)
        keys = jax.random.split(rng_key, num_trials)
        n_accepts = 0
        for key in keys:
            new_state, info = kernel(key, state)
            if info.is_accepted:
                n_accepts += 1
            state = new_state
        acceptance_rate = n_accepts / num_trials
        print(f"step_size={step_size:.5f} => acceptance_rate={acceptance_rate:.3f}")
        # You can implement your selection criteria here.
        if 0.2 < acceptance_rate < 0.5 and acceptance_rate > best_rate:
            best_rate = acceptance_rate
            best_step = step_size
    print(f"Selected step_size={best_step} with acceptance_rate={best_rate}")
    return best_step