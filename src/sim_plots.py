# src/plotting.py

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless servers
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jaxpm.painting import cic_paint
import numpy as np
from utils import calculate_energy_variable_mass, blob_enclosed_mass_gaussian, blob_enclosed_mass_nfw, blob_enclosed_mass_plummer, lagrangian_radius



### Plotting functions for DiffNBody simulations

def plot_density_fields_and_positions(G, tf, dt, length, n_part, initial_field, init_pos, final_field, final_pos, density_scaling, solver, kpc_per_pixel):
    """
    Plot density fields and particle positions in various projections.

    Parameters:
    -----------
    input_field : array
        The input phase field - already scaled
    init_pos : array
        Initial particle positions (N x 3 array)
    final_pos : array
        Final particle positions (N x 3 array)
    output_field : array
        The output density field (3D array) - already scaled
    density_scaling : str
        Type of density scaling applied
    """
    
    initial_density = initial_field[..., 0]
    final_density = final_field[..., 0]

    # Create figure with a grid layout
    fig = plt.figure(figsize=(40, 10))
    gs = plt.GridSpec(2, 7, figure=fig) 

    title = 'Simulation with parameters:'
    param_info = f'G={G}, tf={tf:.2f} Gyr, dt={dt:.2f} Gyr, L={length*kpc_per_pixel:.1f} kpc, N={n_part}, density_scaling={density_scaling}, solver={solver}'

    title += f'\n{param_info}'
    # Place suptitle at the very top
    fig.suptitle(title, fontsize=25)
    fig.subplots_adjust(top=0.92)  # Adjust for the suptitle

    # Determine colorbar label based on scaling
    if density_scaling == "log":
        cbar_label = r"Log Density [$M_{\odot}$ kpc$^{-3}$]"
    elif density_scaling == "sqrt":
        cbar_label = r"$\sqrt{\text{Density}}$ [$M_{\odot}^{1/2}$ kpc$^{-3/2}$]"
    elif density_scaling == "normalize":
        cbar_label = "Normalized Density"
    elif density_scaling == "standardize":
        cbar_label = "Standardized Density"
    elif density_scaling == "power":
        cbar_label = "Power-scaled Density"
    else:
        cbar_label = r"Density [$M_{\odot}$ kpc$^{-3}$]"

    # Helper for axis limits in kpc
    limits_kpc = [0, length * kpc_per_pixel]

    # First row: Initial positions/density
    ax0 = fig.add_subplot(gs[0, 0], projection='3d')
    ax0.scatter(init_pos[:, 0] * kpc_per_pixel, init_pos[:, 1] * kpc_per_pixel, init_pos[:, 2] * kpc_per_pixel, c='r', marker='o', alpha=0.5, s=1)
    ax0.set_title('Initial Particle Positions (3D)')
    ax0.set_xlabel('X [kpc]'); ax0.set_ylabel('Y [kpc]'); ax0.set_zlabel('Z [kpc]')
    ax0.set_xlim(limits_kpc)
    ax0.set_ylim(limits_kpc)
    ax0.set_zlim(limits_kpc)
    ax0.set_aspect('equal')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.scatter(init_pos[:, 0] * kpc_per_pixel, init_pos[:, 1] * kpc_per_pixel, c='r', marker='o', alpha=0.5, s=1)
    ax1.set_title('Initial Positions (X-Y)')
    ax1.set_xlabel('X [kpc]'); ax1.set_ylabel('Y [kpc]')
    ax1.set_xlim(limits_kpc)
    ax1.set_ylim(limits_kpc)
    ax1.set_aspect('equal')

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(init_pos[:, 0] * kpc_per_pixel, init_pos[:, 2] * kpc_per_pixel, c='r', marker='o', alpha=0.5, s=1)
    ax2.set_title('Initial Positions (X-Z)')
    ax2.set_xlabel('X [kpc]'); ax2.set_ylabel('Z [kpc]')
    ax2.set_xlim(limits_kpc)
    ax2.set_ylim(limits_kpc)
    ax2.set_aspect('equal')
    
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.scatter(init_pos[:, 1] * kpc_per_pixel, init_pos[:, 2] * kpc_per_pixel, c='r', marker='o', alpha=0.5, s=1)
    ax3.set_title('Initial Positions (Y-Z)')
    ax3.set_xlabel('Y [kpc]'); ax3.set_ylabel('Z [kpc]')
    ax3.set_xlim(limits_kpc)
    ax3.set_ylim(limits_kpc)
    ax3.set_aspect('equal')

    ax4 = fig.add_subplot(gs[0, 4])
    im4 = ax4.imshow(jnp.sum(initial_density, axis=0), cmap='inferno', origin='lower',
                     extent=[0, length * kpc_per_pixel, 0, length * kpc_per_pixel], aspect='equal')
    ax4.set_title(f'Input {cbar_label.split("[")[0].strip()} Field (Projection X-Y)')
    ax4.set_xlabel('X [kpc]')
    ax4.set_ylabel('Y [kpc]')
    ax4.set_xlim(limits_kpc)
    ax4.set_ylim(limits_kpc)
    cbar4 = fig.colorbar(im4, ax=ax4, orientation='vertical')
    cbar4.set_label(cbar_label)

    ax5 = fig.add_subplot(gs[0, 5])
    im5 = ax5.imshow(jnp.sum(initial_density, axis=1), cmap='inferno', origin='lower',
                     extent=[0, length * kpc_per_pixel, 0, length * kpc_per_pixel], aspect='equal')
    ax5.set_title(f'Input {cbar_label.split("[")[0].strip()} Field (Projection X-Z)')
    ax5.set_xlabel('X [kpc]')
    ax5.set_ylabel('Z [kpc]')
    ax5.set_xlim(limits_kpc)
    ax5.set_ylim(limits_kpc)
    cbar5 = fig.colorbar(im5, ax=ax5, orientation='vertical')
    cbar5.set_label(cbar_label)

    ax6 = fig.add_subplot(gs[0, 6])
    im6 = ax6.imshow(jnp.sum(initial_density, axis=2), cmap='inferno', origin='lower',
                     extent=[0, length * kpc_per_pixel, 0, length * kpc_per_pixel], aspect='equal')
    ax6.set_title(f'Input {cbar_label.split("[")[0].strip()} Field (Projection Y-Z)')
    ax6.set_xlabel('Y [kpc]')
    ax6.set_ylabel('Z [kpc]')
    ax6.set_xlim(limits_kpc)
    ax6.set_ylim(limits_kpc)
    cbar6 = fig.colorbar(im6, ax=ax6, orientation='vertical')
    cbar6.set_label(cbar_label)
    
    # Second row: Final positions/density
    
    ax7 = fig.add_subplot(gs[1, 0], projection='3d')
    ax7.scatter(final_pos[:, 0] * kpc_per_pixel, final_pos[:, 1] * kpc_per_pixel, final_pos[:, 2] * kpc_per_pixel, c='r', marker='o', alpha=0.5, s=1)
    ax7.set_title('Final Particle Positions (3D)')
    ax7.set_xlabel('X [kpc]'); ax7.set_ylabel('Y [kpc]'); ax7.set_zlabel('Z [kpc]')
    ax7.set_xlim(limits_kpc)
    ax7.set_ylim(limits_kpc)
    ax7.set_zlim(limits_kpc)
    ax7.set_aspect('equal')

    ax8 = fig.add_subplot(gs[1, 1])
    ax8.scatter(final_pos[:, 0] * kpc_per_pixel, final_pos[:, 1] * kpc_per_pixel, c='r', marker='o', alpha=0.5, s=1)
    ax8.set_title('Final Positions (X-Y)')
    ax8.set_xlabel('X [kpc]'); ax8.set_ylabel('Y [kpc]')
    ax8.set_xlim(limits_kpc)
    ax8.set_ylim(limits_kpc)
    ax8.set_aspect('equal')

    ax9 = fig.add_subplot(gs[1, 2])
    ax9.scatter(final_pos[:, 0] * kpc_per_pixel, final_pos[:, 2] * kpc_per_pixel, c='r', marker='o', alpha=0.5, s=1)
    ax9.set_title('Final Positions (X-Z)')
    ax9.set_xlabel('X [kpc]'); ax9.set_ylabel('Z [kpc]')
    ax9.set_xlim(limits_kpc)
    ax9.set_ylim(limits_kpc)
    ax9.set_aspect('equal')

    ax10 = fig.add_subplot(gs[1, 3])
    ax10.scatter(final_pos[:, 1] * kpc_per_pixel, final_pos[:, 2] * kpc_per_pixel, c='r', marker='o', alpha=0.5, s=1)
    ax10.set_title('Final Positions (Y-Z)')
    ax10.set_xlabel('Y [kpc]'); ax10.set_ylabel('Z [kpc]')
    ax10.set_xlim(limits_kpc)
    ax10.set_ylim(limits_kpc)
    ax10.set_aspect('equal')

    ax11 = fig.add_subplot(gs[1, 4])
    im11 = ax11.imshow(jnp.sum(final_density, axis=0), cmap='inferno', origin='lower',
                     extent=[0, length * kpc_per_pixel, 0, length * kpc_per_pixel], aspect='equal')
    ax11.set_title(f'Final {cbar_label.split("[")[0].strip()} Field (Projection X-Y)')
    ax11.set_xlabel('X [kpc]')
    ax11.set_ylabel('Y [kpc]')
    ax11.set_xlim(limits_kpc)
    ax11.set_ylim(limits_kpc)
    cbar11 = fig.colorbar(im11, ax=ax11, orientation='vertical')
    cbar11.set_label(cbar_label)

    ax12 = fig.add_subplot(gs[1, 5])
    im12 = ax12.imshow(jnp.sum(final_density, axis=1), cmap='inferno', origin='lower',
                     extent=[0, length * kpc_per_pixel, 0, length * kpc_per_pixel], aspect='equal')
    ax12.set_title(f'Final {cbar_label.split("[")[0].strip()} Field (Projection X-Z)')
    ax12.set_xlabel('X [kpc]')
    ax12.set_ylabel('Z [kpc]')
    ax12.set_xlim(limits_kpc)
    ax12.set_ylim(limits_kpc)
    cbar12 = fig.colorbar(im12, ax=ax12, orientation='vertical')
    cbar12.set_label(cbar_label)

    ax13 = fig.add_subplot(gs[1, 6])
    im13 = ax13.imshow(jnp.sum(final_density, axis=2), cmap='inferno', origin='lower',
                     extent=[0, length * kpc_per_pixel, 0, length * kpc_per_pixel], aspect='equal')
    ax13.set_title(f'Final {cbar_label.split("[")[0].strip()} Field (Projection Y-Z)')
    ax13.set_xlabel('Y [kpc]')
    ax13.set_ylabel('Z [kpc]')
    ax13.set_xlim(limits_kpc)
    ax13.set_ylim(limits_kpc)
    cbar13 = fig.colorbar(im13, ax=ax13, orientation='vertical')
    cbar13.set_label(cbar_label)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_velocity_distributions(initial_field, final_field, sol_ys, G, tf, dt, length, n_part, solver, quiver_stride=5):
    # Calculate velocity norms for initial and final velocities
    init_vel = sol_ys[0, 1]  # Initial velocities
    init_pos = sol_ys[0, 0]  # Initial positions
    init_vel_norm = jnp.sqrt(jnp.sum(init_vel**2, axis=1))
    initial_velocity_field_x = initial_field[..., 1]  
    initial_velocity_field_y = initial_field[..., 2]
    initial_velocity_field_z = initial_field[..., 3]

    final_vel = sol_ys[-1, 1]  # Final velocities
    final_pos = sol_ys[-1, 0]  # Final positions
    final_vel_norm = jnp.sqrt(jnp.sum(final_vel**2, axis=1))
    final_velocity_field_x = final_field[..., 1]
    final_velocity_field_y = final_field[..., 2]
    final_velocity_field_z = final_field[..., 3]

    fig = plt.figure(figsize=(24, 35))
    gs = plt.GridSpec(8, 4, figure=fig) 
    limits = [0, length]


    title = 'Velocity Distribution Comparison with parameters:'
    param_info = f'G={G}, tf={tf}, dt={dt}, L={length}, N={n_part}, solver={solver}'
    
    title += f'\n{param_info}'
    # Place suptitle at the very top
    fig.suptitle(title, fontsize=22)
    fig.subplots_adjust(top=0.92)

    stride = quiver_stride
    idx = np.arange(0, n_part, stride)

    # First 4 rows : initial velocity
    # 1st row : x
    ax00 = fig.add_subplot(gs[0, 0])
    im00 = ax00.imshow(jnp.sum(initial_velocity_field_x, axis=0), cmap='inferno', origin='lower',
                     extent=[0, length, 0, length], aspect='equal')
    ax00.set_title(f'Initial Vx Field (Projection X-Y)')
    ax00.set_xlabel('X')
    ax00.set_ylabel('Y')
    ax00.set_xlim(limits)
    ax00.set_ylim(limits)
    fig.colorbar(im00, ax=ax00, orientation='vertical')

    ax01 = fig.add_subplot(gs[0, 1])
    im01 = ax01.imshow(jnp.sum(initial_velocity_field_x, axis=1), cmap='inferno', origin='lower',
                     extent=[0, length, 0, length], aspect='equal')
    ax01.set_title(f'Initial Vx Field (Projection X-Z)')
    ax01.set_xlabel('X')
    ax01.set_ylabel('Z')
    ax01.set_xlim(limits)
    ax01.set_ylim(limits)
    fig.colorbar(im01, ax=ax01, orientation='vertical')

    ax02 = fig.add_subplot(gs[0, 2])
    im02 = ax02.imshow(jnp.sum(initial_velocity_field_x, axis=2), cmap='inferno', origin='lower',
                     extent=[0, length, 0, length], aspect='equal')
    ax02.set_title(f'Initial Vx Field (Projection Y-Z)')
    ax02.set_xlabel('Y')
    ax02.set_ylabel('Z')
    ax02.set_xlim(limits)
    ax02.set_ylim(limits)
    fig.colorbar(im02, ax=ax02, orientation='vertical')

    ax03 = fig.add_subplot(gs[0, 3])
    ax03.hist(init_vel[:, 0], bins=50, color='blue', alpha=0.7, density=True)
    ax03.set_xlabel('Vx', fontsize=12)
    ax03.set_ylabel('Density', fontsize=12)
    ax03.set_title('Initial Vx Distribution', fontsize=14)
    ax03.grid(True, alpha=0.3)
    ax03.text(0.05, 0.95, 
        f"Mean={jnp.mean(init_vel[:, 0]):.2f}\nStd={jnp.std(init_vel[:, 0]):.2f}",
        transform=ax03.transAxes, 
        bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment='top')
    
    # 2nd row : y

    ax10 = fig.add_subplot(gs[1, 0])
    im10 = ax10.imshow(jnp.sum(initial_velocity_field_y, axis=0), cmap='inferno', origin='lower',
                     extent=[0, length, 0, length], aspect='equal')
    ax10.set_title(f'Initial Vy Field (Projection X-Y)')
    ax10.set_xlabel('X')
    ax10.set_ylabel('Y')
    ax10.set_xlim(limits)
    ax10.set_ylim(limits)
    fig.colorbar(im10, ax=ax10, orientation='vertical')

    ax11 = fig.add_subplot(gs[1, 1])
    im11 = ax11.imshow(jnp.sum(initial_velocity_field_y, axis=1), cmap='inferno', origin='lower',
                     extent=[0, length, 0, length], aspect='equal')
    ax11.set_title(f'Initial Vy Field (Projection X-Z)')
    ax11.set_xlabel('X')
    ax11.set_ylabel('Z')
    ax11.set_xlim(limits)
    ax11.set_ylim(limits)
    fig.colorbar(im11, ax=ax11, orientation='vertical')

    ax12 = fig.add_subplot(gs[1, 2])
    im12 = ax12.imshow(jnp.sum(initial_velocity_field_y, axis=2), cmap='inferno', origin='lower',
                     extent=[0, length, 0, length], aspect='equal')
    ax12.set_title(f'Initial Vy Field (Projection Y-Z)')
    ax12.set_xlabel('Y')
    ax12.set_ylabel('Z')
    ax12.set_xlim(limits)
    ax12.set_ylim(limits)
    fig.colorbar(im12, ax=ax12, orientation='vertical')

    ax13 = fig.add_subplot(gs[1, 3])
    ax13.hist(init_vel[:, 1], bins=50, color='blue', alpha=0.7, density=True)
    ax13.set_xlabel('Vy', fontsize=12)
    ax13.set_ylabel('Density', fontsize=12)
    ax13.set_title('Initial Vy Distribution', fontsize=14)
    ax13.grid(True, alpha=0.3)
    ax13.text(0.05, 0.95, 
        f"Mean={jnp.mean(init_vel[:, 1]):.2f}\nStd={jnp.std(init_vel[:, 1]):.2f}",
        transform=ax13.transAxes, 
        bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment='top')
    

    # 3rd row : z
    
    ax20 = fig.add_subplot(gs[2, 0])
    im20 = ax20.imshow(jnp.sum(initial_velocity_field_z, axis=0), cmap='inferno', origin='lower',
                     extent=[0, length, 0, length], aspect='equal')
    ax20.set_title(f'Initial Vz Field (Projection X-Y)')
    ax20.set_xlabel('X')
    ax20.set_ylabel('Y')
    ax20.set_xlim(limits)
    ax20.set_ylim(limits)
    fig.colorbar(im20, ax=ax20, orientation='vertical')

    ax21 = fig.add_subplot(gs[2, 1])
    im21 = ax21.imshow(jnp.sum(initial_velocity_field_z, axis=1), cmap='inferno', origin='lower',
                     extent=[0, length, 0, length], aspect='equal')
    ax21.set_title(f'Initial Vz Field (Projection X-Z)')
    ax21.set_xlabel('X')
    ax21.set_ylabel('Z')
    ax21.set_xlim(limits)
    ax21.set_ylim(limits)
    fig.colorbar(im21, ax=ax21, orientation='vertical')

    ax22 = fig.add_subplot(gs[2, 2])
    im22 = ax22.imshow(jnp.sum(initial_velocity_field_z, axis=2), cmap='inferno', origin='lower',
                     extent=[0, length, 0, length], aspect='equal')
    ax22.set_title(f'Initial Vz Field (Projection Y-Z)')
    ax22.set_xlabel('Y')
    ax22.set_ylabel('Z')
    ax22.set_xlim(limits)
    ax22.set_ylim(limits)
    fig.colorbar(im22, ax=ax22, orientation='vertical')


    ax23 = fig.add_subplot(gs[2, 3])
    ax23.hist(init_vel[:, 2], bins=50, color='blue', alpha=0.7, density=True)
    ax23.set_xlabel('Vz', fontsize=12)
    ax23.set_ylabel('Density', fontsize=12)
    ax23.set_title('Initial Vz Distribution', fontsize=14)
    ax23.grid(True, alpha=0.3)
    ax23.text(0.05, 0.95, 
        f"Mean={jnp.mean(init_vel[:, 2]):.2f}\nStd={jnp.std(init_vel[:, 2]):.2f}",
        transform=ax23.transAxes, 
        bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment='top')

    # 4th row : velocity vector fields and magnitudes 

    ax30 = fig.add_subplot(gs[3, 0])
    ax30.quiver(init_pos[idx, 0], init_pos[idx, 1], init_vel[idx, 0], init_vel[idx, 1], 
                angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.7)
    ax30.set_title('Initial Velocity Field (XY)', fontsize=14)
    ax30.set_xlabel('X')
    ax30.set_ylabel('Y')
    ax30.set_xlim(0, length)
    ax30.set_ylim(0, length)
    ax30.grid(True, alpha=0.3)
    ax30.set_aspect('equal')

    ax31 = fig.add_subplot(gs[3, 1])
    ax31.quiver(init_pos[idx, 0], init_pos[idx, 2], init_vel[idx, 0], init_vel[idx, 2], 
                angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.7)
    ax31.set_title('Initial Velocity Field (XZ)', fontsize=14)
    ax31.set_xlabel('X')
    ax31.set_ylabel('Z')
    ax31.set_xlim(0, length)
    ax31.set_ylim(0, length)
    ax31.grid(True, alpha=0.3)
    ax31.set_aspect('equal')

    ax32 = fig.add_subplot(gs[3, 2])
    ax32.quiver(init_pos[idx, 1], init_pos[idx, 2], init_vel[idx, 1], init_vel[idx, 2], 
                angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.7)
    ax32.set_title('Initial Velocity Field (YZ)', fontsize=14)
    ax32.set_xlabel('Y')
    ax32.set_ylabel('Z')
    ax32.set_xlim(0, length)
    ax32.set_ylim(0, length)
    ax32.grid(True, alpha=0.3)
    ax32.set_aspect('equal')

    ax33 = fig.add_subplot(gs[3, 3])
    ax33.hist(init_vel_norm, bins=50, color='blue', alpha=0.7, density=True)
    ax33.set_xlabel('Velocity Magnitude', fontsize=12)
    ax33.set_ylabel('Density', fontsize=12)
    ax33.set_title('Initial Velocity Magnitudes', fontsize=14)
    ax33.text(0.05, 0.95,
        f"Mean={jnp.mean(init_vel_norm):.2f}\nStd={jnp.std(init_vel_norm):.2f}",
        transform=ax33.transAxes, 
        bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment='top')

    # 4 last rows : final velocity
    # 1st row : x
    ax40 = fig.add_subplot(gs[4, 0])
    im40 = ax40.imshow(jnp.sum(final_velocity_field_x, axis=0), cmap='inferno', origin='lower',
                     extent=[0, length, 0, length], aspect='equal')
    ax40.set_title(f'Final Vx Field (Projection X-Y)')
    ax40.set_xlabel('X')
    ax40.set_ylabel('Y')
    ax40.set_xlim(limits)
    ax40.set_ylim(limits)
    fig.colorbar(im40, ax=ax40, orientation='vertical')

    ax41 = fig.add_subplot(gs[4, 1])
    im41 = ax41.imshow(jnp.sum(final_velocity_field_x, axis=1), cmap='inferno', origin='lower',
                     extent=[0, length, 0, length], aspect='equal')
    ax41.set_title(f'Final Vx Field (Projection X-Z)')
    ax41.set_xlabel('X')
    ax41.set_ylabel('Z')
    ax41.set_xlim(limits)
    ax41.set_ylim(limits)
    fig.colorbar(im41, ax=ax41, orientation='vertical')

    ax42 = fig.add_subplot(gs[4, 2])
    im42 = ax42.imshow(jnp.sum(final_velocity_field_x, axis=2), cmap='inferno', origin='lower',
                     extent=[0, length, 0, length], aspect='equal')
    ax42.set_title(f'Final Vx Field (Projection Y-Z)')
    ax42.set_xlabel('Y')
    ax42.set_ylabel('Z')
    ax42.set_xlim(limits)
    ax42.set_ylim(limits)
    fig.colorbar(im42, ax=ax42, orientation='vertical')

    ax43 = fig.add_subplot(gs[4, 3])
    ax43.hist(final_vel[:, 0], bins=50, color='blue', alpha=0.7, density=True)
    ax43.set_xlabel('Vx', fontsize=12)
    ax43.set_ylabel('Density', fontsize=12)
    ax43.set_title('Final Vx Distribution', fontsize=14)
    ax43.grid(True, alpha=0.3)
    ax43.text(0.05, 0.95, 
        f"Mean={jnp.mean(final_vel[:, 0]):.2f}\nStd={jnp.std(final_vel[:, 0]):.2f}",
        transform=ax43.transAxes, 
        bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment='top')
    
    # 5th row : y

    ax50 = fig.add_subplot(gs[5, 0])
    im50 = ax50.imshow(jnp.sum(final_velocity_field_y, axis=0), cmap='inferno', origin='lower',
                     extent=[0, length, 0, length], aspect='equal')
    ax50.set_title(f'Final Vy Field (Projection X-Y)')
    ax50.set_xlabel('X')
    ax50.set_ylabel('Y')
    ax50.set_xlim(limits)
    ax50.set_ylim(limits)
    fig.colorbar(im50, ax=ax50, orientation='vertical')

    ax51 = fig.add_subplot(gs[5, 1])
    im51 = ax51.imshow(jnp.sum(final_velocity_field_y, axis=1), cmap='inferno', origin='lower',
                     extent=[0, length, 0, length], aspect='equal')
    ax51.set_title(f'Final Vy Field (Projection X-Z)')
    ax51.set_xlabel('X')
    ax51.set_ylabel('Z')
    ax51.set_xlim(limits)
    ax51.set_ylim(limits)
    fig.colorbar(im51, ax=ax51, orientation='vertical')

    ax52 = fig.add_subplot(gs[5, 2])
    im52 = ax52.imshow(jnp.sum(final_velocity_field_y, axis=2), cmap='inferno', origin='lower',
                     extent=[0, length, 0, length], aspect='equal')
    ax52.set_title(f'Final Vy Field (Projection Y-Z)')
    ax52.set_xlabel('Y')
    ax52.set_ylabel('Z')
    ax52.set_xlim(limits)
    ax52.set_ylim(limits)
    fig.colorbar(im52, ax=ax52, orientation='vertical')

    ax53 = fig.add_subplot(gs[5, 3])
    ax53.hist(final_vel[:, 1], bins=50, color='blue', alpha=0.7, density=True)
    ax53.set_xlabel('Vy', fontsize=12)
    ax53.set_ylabel('Density', fontsize=12)
    ax53.set_title('Final Vy Distribution', fontsize=14)
    ax53.grid(True, alpha=0.3)
    ax53.text(0.05, 0.95, 
        f"Mean={jnp.mean(final_vel[:, 1]):.2f}\nStd={jnp.std(final_vel[:, 1]):.2f}",
        transform=ax53.transAxes, 
        bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment='top')

    # 6th row : z
    
    ax60 = fig.add_subplot(gs[6, 0])
    im60 = ax60.imshow(jnp.sum(final_velocity_field_z, axis=0), cmap='inferno', origin='lower',
                     extent=[0, length, 0, length], aspect='equal')
    ax60.set_title(f'Final Vz Field (Projection X-Y)')
    ax60.set_xlabel('X')
    ax60.set_ylabel('Y')
    ax60.set_xlim(limits)
    ax60.set_ylim(limits)
    fig.colorbar(im60, ax=ax60, orientation='vertical')

    ax61 = fig.add_subplot(gs[6, 1])
    im61 = ax61.imshow(jnp.sum(final_velocity_field_z, axis=1), cmap='inferno', origin='lower',
                     extent=[0, length, 0, length], aspect='equal')
    ax61.set_title(f'Final Vz Field (Projection X-Z)')
    ax61.set_xlabel('X')
    ax61.set_ylabel('Z')
    ax61.set_xlim(limits)
    ax61.set_ylim(limits)
    fig.colorbar(im61, ax=ax61, orientation='vertical')

    ax62 = fig.add_subplot(gs[6, 2])
    im62 = ax62.imshow(jnp.sum(final_velocity_field_z, axis=2), cmap='inferno', origin='lower',
                     extent=[0, length, 0, length], aspect='equal')
    ax62.set_title(f'Final Vz Field (Projection Y-Z)')
    ax62.set_xlabel('Y')
    ax62.set_ylabel('Z')
    ax62.set_xlim(limits)
    ax62.set_ylim(limits)
    fig.colorbar(im62, ax=ax62, orientation='vertical')


    ax63 = fig.add_subplot(gs[6, 3])
    ax63.hist(final_vel[:, 2], bins=50, color='blue', alpha=0.7, density=True)
    ax63.set_xlabel('Vz', fontsize=12)
    ax63.set_ylabel('Density', fontsize=12)
    ax63.set_title('Final Vz Distribution', fontsize=14)
    ax63.grid(True, alpha=0.3)
    ax63.text(0.05, 0.95, 
        f"Mean={jnp.mean(final_vel[:, 2]):.2f}\nStd={jnp.std(final_vel[:, 2]):.2f}",
        transform=ax63.transAxes, 
        bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment='top')

    # 7th row : velocity vector fields and magnitudes 

    ax70 = fig.add_subplot(gs[7, 0])
    ax70.quiver(final_pos[idx, 0], final_pos[idx, 1], final_vel[idx, 0], final_vel[idx, 1], 
                angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.7)
    ax70.set_title('Final Velocity Field (XY)', fontsize=14)
    ax70.set_xlabel('X')
    ax70.set_ylabel('Y')
    ax70.set_xlim(0, length)
    ax70.set_ylim(0, length)
    ax70.grid(True, alpha=0.3)
    ax70.set_aspect('equal')

    ax71 = fig.add_subplot(gs[7, 1])
    ax71.quiver(final_pos[idx, 0], final_pos[idx, 2], final_vel[idx, 0], final_vel[idx, 2], 
                angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.7)
    ax71.set_title('Final Velocity Field (XZ)', fontsize=14)
    ax71.set_xlabel('X')
    ax71.set_ylabel('Z')
    ax71.set_xlim(0, length)
    ax71.set_ylim(0, length)
    ax71.grid(True, alpha=0.3)
    ax71.set_aspect('equal')

    ax72 = fig.add_subplot(gs[7, 2])
    ax72.quiver(final_pos[idx, 1], final_pos[idx, 2], final_vel[idx, 1], final_vel[idx, 2], 
                angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.7)
    ax72.set_title('Final Velocity Field (YZ)', fontsize=14)
    ax72.set_xlabel('Y')
    ax72.set_ylabel('Z')
    ax72.set_xlim(0, length)
    ax72.set_ylim(0, length)
    ax72.grid(True, alpha=0.3)
    ax72.set_aspect('equal')

    ax73 = fig.add_subplot(gs[7, 3])
    ax73.hist(final_vel_norm, bins=50, color='blue', alpha=0.7, density=True)
    ax73.set_xlabel('Velocity Magnitude', fontsize=12)
    ax73.set_ylabel('Density', fontsize=12)
    ax73.set_title('Final Velocity Magnitudes', fontsize=14)
    ax73.text(0.05, 0.95,
        f"Mean={jnp.mean(final_vel_norm):.2f}\nStd={jnp.std(final_vel_norm):.2f}",
        transform=ax73.transAxes, 
        bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment='top')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  
    
    return fig


def plot_lagrangian_radii_fractionwise(
    sol_ts,
    sol_ys,
    blobs_params,
    masses,
    length,
    kpc_per_pixel,
    gyr_per_time_unit,
    fractions=(0.1, 0.5, 0.9),
    dpi=150):
    """
    For each fraction in fractions create a plot of Lagrangian radius vs time
    including global system and each blob separately. Returns a figure with subplots for each fraction.
    """
    
    positions_over_time = sol_ys[:, 0]  # (T,N,3)
    # Build blob slices
    slices = []
    start = 0
    for bp in blobs_params:
        n = bp['n_part']
        slices.append(slice(start, start + n))
        start += n
        #assert start == masses.shape[0], "Mass array length mismatch."

    # Create figure and axes
    fig, axes = plt.subplots(len(fractions), 1, figsize=(8, 4 * len(fractions)), dpi=dpi)
    if len(fractions) == 1:
        axes = [axes]  # Ensure axes is iterable for a single fraction

    for ax, f in zip(axes, fractions):
        global_r = []
        blob_r_list = [ [] for _ in slices ]
        for t in range(positions_over_time.shape[0]):
            pos_t = positions_over_time[t]
            # Global
            global_r.append(lagrangian_radius(pos_t, masses, f, length))
            # Blobs
            for bi, sl in enumerate(slices):
                blob_r_list[bi].append(lagrangian_radius(pos_t[sl], masses[sl], f, length))
        global_r = jnp.array(global_r) * kpc_per_pixel
        blob_r_list = [jnp.array(br) * kpc_per_pixel for br in blob_r_list]

        # Plot
        ax.plot(sol_ts * gyr_per_time_unit, global_r, label="Global", linewidth=2.0, color="black")
        for bi, br in enumerate(blob_r_list):
            ax.plot(sol_ts * gyr_per_time_unit, br, label=f"Blob {bi+1}", alpha=0.85)
        ax.set_xlabel("Time [Gyr]")
        ax.set_ylabel(f"Radius [kpc] (f={f:.1f})")
        ax.set_title(f"Lagrangian Radius Evolution (f={f:.1f})")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

    plt.tight_layout()
    return fig


def plot_timesteps(sol_ts,
                   sol_ys,
                   boxL,
                   G,    
                   tf,
                   dt,
                   n_part,
                   softening,
                   masses,
                   solver,
                   num_timesteps,
                   enable_energy_tracking,
                   density_scaling,
                   energy_data,  # Pre-computed energy data
                   kpc_per_pixel,
                   msun_per_mass_unit,
                   gyr_per_time_unit,
                   jouleE50_per_unit
                   ):
    
    total_timesteps = len(sol_ts)
    if num_timesteps >= total_timesteps:
        skip = 1
    else:
        skip = max(1, total_timesteps // num_timesteps)
    steps = sol_ts[::skip]
    nrows = len(steps)

    # Use pre-computed energy data if available, otherwise compute if needed
    all_times = sol_ts
    all_ke = all_pe = all_te = None
    all_eta = None
    E0 = None  # NEW: initial total energy reference

    if enable_energy_tracking:
        if energy_data is not None:
            print("Using pre-computed energy data for timesteps plot...")
            all_times = energy_data['times']
            all_ke = energy_data['kinetic']
            all_pe = energy_data['potential']
            all_te = energy_data['total']
        else:
            print("Computing energies for all timesteps (no pre-computed data provided)...")
            all_ke, all_pe, all_te = [], [], []
            for i in range(len(sol_ts)):
                pos_t = sol_ys[i, 0]
                vel_t = sol_ys[i, 1]
                ke, pe, te = calculate_energy_variable_mass(pos_t, vel_t, masses, G, boxL, softening)
                all_ke.append(ke); all_pe.append(pe); all_te.append(te)
            all_ke = jnp.array(all_ke); all_pe = jnp.array(all_pe); all_te = jnp.array(all_te)

        # Convert energies to 10^50 J units
        all_ke = all_ke * jouleE50_per_unit 
        all_pe = all_pe * jouleE50_per_unit
        all_te = all_te * jouleE50_per_unit 

        # --- Virial ratio series ---
        denom = jnp.maximum(jnp.abs(all_pe), 1e-30)
        all_eta = 2.0 * all_ke / denom

        # --- NEW: energy tolerance band anchor ---
        if all_te is not None and len(all_te) > 0:
            E0 = float(all_te[0])  # use initial total energy as conserved target

    # Pre-build the canvas - number of columns depends on energy tracking
    ncols = 6 if enable_energy_tracking else 4  # +1 column for virial ratio when energies are on

    fig, axes = plt.subplots(nrows=nrows,
                             ncols=ncols,
                             figsize=(6*ncols, 4 * nrows))

    total_mass = jnp.sum(masses)
    title = 'Simulation with parameters:'
    param_info = f'G={float(G)}, tf={float(tf):.2f} Gyr, dt={float(dt):.2f} Gyr, L={float(boxL)*float(kpc_per_pixel):.1f} kpc, N={int(n_part)}, M_tot={float(total_mass)*float(msun_per_mass_unit):.2e} M$_\\odot$, density_scaling={density_scaling}, solver={solver}'
    if enable_energy_tracking:
        param_info += ', energy_tracking=True'
    param_info += f', Plotting {len(steps)} timesteps'
    title += f'\n{param_info}'
    fig.suptitle(title, fontsize=22)
    fig.subplots_adjust(top=0.96)

    for row, t in enumerate(steps): 
        current_step = row * skip
        pos_t = sol_ys[current_step, 0] 
        vel_t = sol_ys[current_step, 1]
        
        # --- projections ----------------------------------------------------
        axes[row, 0].scatter(pos_t[:, 0] * kpc_per_pixel, pos_t[:, 1] * kpc_per_pixel, s=1)
        axes[row, 1].scatter(pos_t[:, 0] * kpc_per_pixel, pos_t[:, 2] * kpc_per_pixel, s=1)
        axes[row, 2].scatter(pos_t[:, 1] * kpc_per_pixel, pos_t[:, 2] * kpc_per_pixel, s=1)

        for col, lbl in zip(range(3), ['XY', 'XZ', 'YZ']):
            ax = axes[row, col]
            ax.set_title(f't={t*gyr_per_time_unit:.2f} Gyr  ({lbl})')
            ax.set_xlabel(f'{lbl[0]} [kpc]'); ax.set_ylabel(f'{lbl[1]} [kpc]')
            ax.set_xlim(0, boxL * kpc_per_pixel);  ax.set_ylim(0, boxL * kpc_per_pixel)
            ax.set_aspect('equal')

        # --- density slice (weighted by mass) ------------------------------
        field_t = cic_paint(jnp.zeros((boxL, boxL, boxL)), pos_t, weight=masses)
        if density_scaling != "none":
            from utils import apply_density_scaling
            field_t = apply_density_scaling(field_t, density_scaling)
        
        # Convert density to physical units (Msun/kpc^3)
        field_t_physical = field_t * msun_per_mass_unit / (kpc_per_pixel**3)
        
        im = axes[row, 3].imshow(jnp.sum(field_t_physical, axis=2),
                                 cmap='inferno', origin='lower',
                                 extent=[0, boxL * kpc_per_pixel, 0, boxL * kpc_per_pixel])
        if density_scaling == "none":
            field_title = f'Density Field (t={t*gyr_per_time_unit:.2f} Gyr)'
            cbar_label = r'Density [M$_\odot$ kpc$^{-3}$]'
        else:
            field_title = f'{density_scaling.capitalize()}-scaled Density Field (t={t*gyr_per_time_unit:.2f} Gyr)'
            if density_scaling == "log":
                cbar_label = r'Log Density [M$_\odot$ kpc$^{-3}$]'
            elif density_scaling == "sqrt":
                cbar_label = r'$\sqrt{\text{Density}}$ [M$_\odot^{1/2}$ kpc$^{-3/2}$]'
            else:
                cbar_label = f'{density_scaling.capitalize()}-scaled Density'
        
        axes[row, 3].set_title(field_title)
        axes[row, 3].set_xlabel('X [kpc]')
        axes[row, 3].set_ylabel('Y [kpc]')
        cbar = fig.colorbar(im, ax=axes[row, 3], orientation='vertical')
        cbar.set_label(cbar_label)
        
        # --- energy evolution (if enabled) ---------------------------------
        if enable_energy_tracking:
            time_mask = all_times <= t

            # Energy panel (col 4)
            axE = axes[row, 4]
            # NEW: thin horizontal band around E0 ± 1e-3
            if E0 is not None and jnp.isfinite(E0):
                axE.axhspan(E0*(1.0 - 1e-2), E0*(1.0 + 1e-2),
                            alpha=0.15, zorder=0)  # light band behind curves

            axE.plot(all_times[time_mask] * gyr_per_time_unit, all_ke[time_mask], 'b-', label='Kinetic', linewidth=2)
            axE.plot(all_times[time_mask] * gyr_per_time_unit, all_pe[time_mask], 'r-', label='Potential', linewidth=2)
            axE.plot(all_times[time_mask] * gyr_per_time_unit, all_te[time_mask], 'k-', label='Total', linewidth=2)
            axE.axvline(x=t * gyr_per_time_unit, color='gray', linestyle='--', alpha=0.7)
            axE.set_xlabel('Time [Gyr]'); axE.set_ylabel(r'Energy [$10^{50}$ J]')
            axE.set_title(f'Energy Evolution (t={t*gyr_per_time_unit:.2f} Gyr)')
            axE.legend(); axE.grid(True, alpha=0.3)

            energy_min = min(jnp.min(all_ke), jnp.min(all_pe), jnp.min(all_te))
            energy_max = max(jnp.max(all_ke), jnp.max(all_pe), jnp.max(all_te))
            energy_range = energy_max - energy_min
            if energy_range > 0:
                axE.set_ylim(energy_min - 0.1*energy_range, energy_max + 0.1*energy_range)

            # Virial ratio panel (col 5)
            axV = axes[row, 5]
            # NEW: thin horizontal band around eta = 1 ± 0.05
            axV.axhspan(1.0 - 0.05, 1.0 + 0.05, alpha=0.15, zorder=0)
            axV.plot(all_times[time_mask] * gyr_per_time_unit, all_eta[time_mask], linewidth=2)
            axV.axhline(1.0, linestyle='--', alpha=0.6)
            axV.axvline(x=t * gyr_per_time_unit, color='gray', linestyle='--', alpha=0.7)
            axV.set_xlabel('Time [Gyr]'); axV.set_ylabel(r'$\eta=2K/|W|$')
            axV.set_title(f'Virial Ratio (t={t*gyr_per_time_unit:.2f} Gyr)')
            axV.grid(True, alpha=0.3)
            # optional y-limits
            try:
                eta_min = float(jnp.nanpercentile(all_eta[time_mask], 1))
                eta_max = float(jnp.nanpercentile(all_eta[time_mask], 99))
                if jnp.isfinite(eta_min) and jnp.isfinite(eta_max) and eta_max > eta_min:
                    pad = 0.1*(eta_max-eta_min)
                    axV.set_ylim(eta_min - pad, eta_max + pad)
            except Exception:
                pass

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig, axes


def plot_trajectories(sol_ys, G, tf, dt, length, n_part, solver, kpc_per_pixel, num_trajectories=10, zoom=True):    
    positions = sol_ys[:, 0]
    num_steps, n_particles, _ = positions.shape

    # Select random trajectories to plot
    if num_trajectories > n_particles:
        num_trajectories = n_particles
    trajectory_indices = np.random.choice(n_particles, num_trajectories, replace=False)

    fig = plt.figure(figsize=(20, 5))
    gs = plt.GridSpec(1, 4, figure=fig)
    ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax_xy = fig.add_subplot(gs[0, 1])
    ax_xz = fig.add_subplot(gs[0, 2])
    ax_yz = fig.add_subplot(gs[0, 3])

    # Create parameter info string
    param_info = f'G={G}, tf={tf:.2f} Gyr, dt={dt:.2f} Gyr, L={length*kpc_per_pixel:.1f} kpc, N={n_part}, solver={solver}'
    title = param_info

    ax_3d_title = '3D Trajectories'
    ax_xy.set_title('XY Projection')
    ax_xz.set_title('XZ Projection')
    ax_yz.set_title('YZ Projection')

    ax_3d.set_xlabel('X [kpc]')
    ax_3d.set_ylabel('Y [kpc]')
    ax_3d.set_zlabel('Z [kpc]')
    ax_xy.set_xlabel('X [kpc]')
    ax_xy.set_ylabel('Y [kpc]')
    ax_xz.set_xlabel('X [kpc]')
    ax_xz.set_ylabel('Z [kpc]')
    ax_yz.set_xlabel('Y [kpc]')
    ax_yz.set_ylabel('Z [kpc]')

    # Plot final positions of all particles in background with transparency
    final_pos_all = positions[-1, :] * kpc_per_pixel
    ax_3d.scatter(final_pos_all[:, 0], final_pos_all[:, 1], final_pos_all[:, 2], 
                  color='lightgray', alpha=0.2, s=5)
    ax_xy.scatter(final_pos_all[:, 0], final_pos_all[:, 1], color='lightgray', alpha=0.2, s=5)
    ax_xz.scatter(final_pos_all[:, 0], final_pos_all[:, 2], color='lightgray', alpha=0.2, s=5)
    ax_yz.scatter(final_pos_all[:, 1], final_pos_all[:, 2], color='lightgray', alpha=0.2, s=5)

    colors = plt.cm.jet(np.linspace(0, 1, len(trajectory_indices)))
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')
    
    smoothing = False
    if num_steps < 2 * length:
        from utils import smooth_trajectory
        smooth_window=5
        smoothing = True
        ax_3d_title += f' (smoothed, window={smooth_window})'
    ax_3d.set_title(ax_3d_title)
    
    for i, p_idx in enumerate(trajectory_indices):
        traj = positions[:, p_idx] * kpc_per_pixel  # Convert to kpc
        if smoothing:
            traj = smooth_trajectory(traj, window=smooth_window)
    
        x_min = min(x_min, traj[:, 0].min())
        x_max = max(x_max, traj[:, 0].max())
        y_min = min(y_min, traj[:, 1].min())
        y_max = max(y_max, traj[:, 1].max())
        z_min = min(z_min, traj[:, 2].min())
        z_max = max(z_max, traj[:, 2].max())
        ax_3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=colors[i], linewidth=1.0, alpha=0.7)
        ax_xy.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=1.0)
        ax_xz.plot(traj[:, 0], traj[:, 2], color=colors[i], linewidth=1.0)
        ax_yz.plot(traj[:, 1], traj[:, 2], color=colors[i], linewidth=1.0)
        ax_3d.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color=colors[i], marker='o', s=20)
        ax_3d.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color=colors[i], marker='s', s=20)

        # Add 3 arrows along the trajectory
        arrow_indices = np.linspace(0, len(traj) - 2, 3, dtype=int)
        for idx in arrow_indices:
            # 3D arrow
            ax_3d.quiver(
                traj[idx, 0], traj[idx, 1], traj[idx, 2],
                traj[idx + 1, 0] - traj[idx, 0],
                traj[idx + 1, 1] - traj[idx, 1],
                traj[idx + 1, 2] - traj[idx, 2],
                color=colors[i], arrow_length_ratio=0.2, linewidth=1.5, alpha=0.8
            )
            # 2D arrows
            ax_xy.annotate('', xy=(traj[idx + 1, 0], traj[idx + 1, 1]), 
                           xytext=(traj[idx, 0], traj[idx, 1]),
                           arrowprops=dict(facecolor=colors[i], edgecolor=colors[i], arrowstyle='-|>', lw=1.5, alpha=0.8))
            ax_xz.annotate('', xy=(traj[idx + 1, 0], traj[idx + 1, 2]), 
                           xytext=(traj[idx, 0], traj[idx, 2]),
                           arrowprops=dict(facecolor=colors[i], edgecolor=colors[i], arrowstyle='-|>', lw=1.5, alpha=0.8))
            ax_yz.annotate('', xy=(traj[idx + 1, 1], traj[idx + 1, 2]), 
                           xytext=(traj[idx, 1], traj[idx, 2]),
                           arrowprops=dict(facecolor=colors[i], edgecolor=colors[i], arrowstyle='-|>', lw=1.5, alpha=0.8))

    if zoom:
        padding=0.1
        x_range = max(x_max - x_min, 1e-10)
        y_range = max(y_max - y_min, 1e-10)
        z_range = max(z_max - z_min, 1e-10)
        x_padding = padding * x_range
        y_padding = padding * y_range
        z_padding = padding * z_range
        ax_3d.set_xlim([x_min - x_padding, x_max + x_padding])
        ax_3d.set_ylim([y_min - y_padding, y_max + y_padding])
        ax_3d.set_zlim([z_min - z_padding, z_max + z_padding])
        ax_xy.set_xlim([x_min - x_padding, x_max + x_padding])
        ax_xy.set_ylim([y_min - y_padding, y_max + y_padding])
        ax_xz.set_xlim([x_min - x_padding, x_max + x_padding])
        ax_xz.set_ylim([z_min - z_padding, z_max + z_padding])
        ax_yz.set_xlim([y_min - y_padding, y_max + y_padding])
        ax_yz.set_ylim([z_min - z_padding, z_max + z_padding])
    else:
        ax_3d.set_xlim([0, length * kpc_per_pixel])
        ax_3d.set_ylim([0, length * kpc_per_pixel])
        ax_3d.set_zlim([0, length * kpc_per_pixel])
        ax_xy.set_xlim([0, length * kpc_per_pixel])
        ax_xy.set_ylim([0, length * kpc_per_pixel])
        ax_xz.set_xlim([0, length * kpc_per_pixel])
        ax_xz.set_ylim([0, length * kpc_per_pixel])
        ax_yz.set_xlim([0, length * kpc_per_pixel])
        ax_yz.set_ylim([0, length * kpc_per_pixel])
        ax_xy.set_aspect('equal', adjustable='box')
        ax_xz.set_aspect('equal', adjustable='box')
        ax_yz.set_aspect('equal', adjustable='box')

    ax_xy.grid(True)
    ax_xz.grid(True)
    ax_yz.grid(True)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    start_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8, label='Start')
    end_marker = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='k', markersize=8, label='End')
    background_marker = plt.Line2D([0], [0], marker='o', color='lightgray', markersize=8, label='All Particles')
    ax_3d.legend(handles=[start_marker, end_marker, background_marker], loc='upper right')
    return fig


def plot_velocity_vs_radius_blobs(sol_ts, sol_ys, blobs_params, G, masses, softening, time_idx=0, kpc_per_pixel=1.5625, gyr_per_time_unit=1.0):
    """
    Plots particle velocity vs. radius from the center of each blob and compares with theory.
    Now handles variable masses per blob.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use final state by default
    positions = sol_ys[time_idx, 0]
    velocities = sol_ys[time_idx, 1]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(blobs_params)))
    
    particle_offset = 0
    for i, blob in enumerate(blobs_params):
        n_blob_part = blob['n_part']
        blob_mass = blob.get('m_part', 1.0)
        blob_positions = positions[particle_offset : particle_offset + n_blob_part]
        blob_velocities = velocities[particle_offset : particle_offset + n_blob_part]
        
        pos_params = blob['pos_params']
        vel_params = blob['vel_params']
        
        center = jnp.mean(blob_positions, axis=0)
        
        rel_pos = blob_positions - center
        r_particles = jnp.linalg.norm(rel_pos, axis=1) * kpc_per_pixel
        v_particles = jnp.linalg.norm(blob_velocities, axis=1) * kpc_per_pixel / gyr_per_time_unit
        
        ax.scatter(r_particles, v_particles, s=5, alpha=0.5, color=colors[i], 
                  label=f"Blob {i} ({blob['pos_type']}, m={blob_mass:.1f}) Particles")
        
        # --- Theoretical Curve Calculation ---
        max_radius = jnp.max(r_particles)
        r_theory = jnp.linspace(0, max_radius, 100)
        
        M_blob = n_blob_part * blob_mass
        
        M_enclosed = None
        if blob['pos_type'] == 'gaussian':
            M_enclosed = blob_enclosed_mass_gaussian(r_theory / kpc_per_pixel, M_blob, pos_params['sigma'])
        elif blob['pos_type'] == 'nfw':
            M_enclosed = blob_enclosed_mass_nfw(r_theory / kpc_per_pixel, M_blob, pos_params['rs'], pos_params['c'])
        elif blob['pos_type'] == 'plummer':
            M_enclosed = blob_enclosed_mass_plummer(r_theory / kpc_per_pixel, M_blob, pos_params['rs'])
        
        if M_enclosed is not None and 'circular' in blob['vel_type']:
            vel_factor = vel_params.get('vel_factor', 1.0)
            v_theory = jnp.sqrt(G * M_enclosed / jnp.sqrt((r_theory / kpc_per_pixel)**2 + softening**2)) * vel_factor * kpc_per_pixel / gyr_per_time_unit
            ax.plot(r_theory, v_theory, color=colors[i], linestyle='--', lw=2, 
                   label=f"Blob {i} Theory")
            
        particle_offset += n_blob_part

    ax.set_xlabel(r"Radius from Blob Center [kpc]")
    ax.set_ylabel(r"Velocity Magnitude [kpc Gyr$^{-1}$]")
    ax.set_title(f"Velocity vs. Radius at t={sol_ts[time_idx]*gyr_per_time_unit:.2f} Gyr")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    return fig

def plot_position_vs_radius_blobs(sol_ts, sol_ys, blobs_params, length, time_idx=0, kpc_per_pixel=1.5625, gyr_per_time_unit=1.0):
    """
    Plot particle positions in each projection (XY, XZ, YZ) vs distance to blob center for all blobs,
    plus a histogram of particle count vs radius, in a single row of 4 subplots.
    """
    positions = sol_ys[time_idx, 0]  # Positions at specified time
    time = sol_ts[time_idx]

    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    proj_labels = [('X', 'Y'), ('X', 'Z'), ('Y', 'Z')]
    proj_indices = [(0, 1), (0, 2), (1, 2)]

    particle_idx = 0
    colors = plt.cm.tab10(np.linspace(0, 1, len(blobs_params)))

    for blob_idx, blob in enumerate(blobs_params):
        n_blob_particles = blob['n_part']
        blob_center = jnp.array(blob['pos_params']['center'])
        blob_positions = positions[particle_idx:particle_idx + n_blob_particles]

        dx = blob_positions - blob_center
        dx = dx - length * jnp.round(dx / length)
        distances = jnp.sqrt(jnp.sum(dx**2, axis=1)) * kpc_per_pixel

        pos_type = blob['pos_type']

        # Label for legend
        if pos_type == 'gaussian':
            sigma = blob['pos_params']['sigma']
            blob_label = f"Blob {blob_idx}: Gaussian (σ={sigma:.1f})"
        elif pos_type == 'nfw':
            rs = blob['pos_params']['rs']
            c = blob['pos_params']['c']
            blob_label = f"Blob {blob_idx}: NFW (rs={rs:.1f}, c={c:.1f})"
        elif pos_type == 'plummer':
            rs = blob['pos_params']['rs']
            blob_label = f"Blob {blob_idx}: Plummer (rs={rs:.1f})"
        else:
            blob_label = f"Blob {blob_idx}: {pos_type}"

        for ax, (i, j), (lbl_i, lbl_j) in zip(axes[:3], proj_indices, proj_labels):
            ax.scatter(distances, blob_positions[:, i] * kpc_per_pixel, color=colors[blob_idx], alpha=0.6, s=20, label=blob_label if ax == axes[0] else None)
            ax.set_xlabel(r"Distance to blob center [kpc]")
            ax.set_ylabel(f"{lbl_i} coordinate [kpc]")
            ax.set_title(f"{lbl_i} vs Radius (t={time*gyr_per_time_unit:.2f} Gyr)")

        # Add histogram to the 4th subplot
        axes[3].hist(distances, bins=30, color=colors[blob_idx], alpha=0.5, label=f"Blob {blob_idx}")

        particle_idx += n_blob_particles

    # Add legends only to the first and last subplot
    axes[0].legend()
    axes[3].legend()
    for ax, (lbl_i, lbl_j) in zip(axes[:3], proj_labels):
        ax.grid(True)
    axes[3].set_xlabel(r"Distance to blob center [kpc]")
    axes[3].set_ylabel("Number of particles")
    axes[3].set_title("Particle count vs radius")
    axes[3].grid(True)
    plt.tight_layout()
    return fig

def create_video(sol_ts, sol_ys, length, G, tf, dt, n_part, density_scaling, solver, 
                 video_type="particles", softening=0.1, masses=None, 
                 enable_energy_tracking=False, save_path=None, fps=10, dpi=100, energy_data=None):
    """
    Create a simplified video of the N-body simulation with only one visualization type.
    
    Parameters:
    -----------
    sol_ts : array
        Simulation timesteps
    sol_ys : array
        Simulation state at each timestep (positions, velocities)
    length : float
        Box size
    G : float
        Gravitational constant
    tf : float
        Final simulation time
    dt : float
        Timestep
    n_part : int
        Number of particles
    density_scaling : str
        Type of density scaling applied
    solver : str
        ODE solver used
    video_type : str
        Type of visualization: "particles", "density", or "velocity"
    softening : float
        Softening length
    masses : array, optional
        Particle masses (defaults to unit masses)
    enable_energy_tracking : bool
        Whether to show energy plots
    save_path : str, optional
        Path to save video file
    fps : int
        Frames per second
    dpi : int
        Resolution DPI
    energy_data : dict, optional
        Pre-computed energy data with keys 'times', 'kinetic', 'potential', 'total'
    
    Returns:
    --------
    anim : matplotlib.animation.Animation
        Animation object
    """
    import matplotlib.animation as animation
    import shutil
    
    if not shutil.which('ffmpeg'):
        print("Warning: ffmpeg not found in PATH. Saving may fail or default to .gif format.")
    
    # Default to unit masses if not provided
    if masses is None:
        masses = jnp.ones(n_part)
    
    # Get positions and velocities
    positions = np.array(sol_ys[:, 0])  # Convert to numpy early for matplotlib
    velocities = np.array(sol_ys[:, 1]) if video_type == "velocity" else None
    timesteps = np.array(sol_ts)
    
    # Pre-compute energy data if needed and not provided
    if enable_energy_tracking and energy_data is None:
        from utils import calculate_energy_variable_mass
        print("Pre-computing energy data...")
        all_ke, all_pe, all_te = [], [], []
        for i in range(len(sol_ts)):
            ke, pe, te = calculate_energy_variable_mass(
                sol_ys[i, 0], sol_ys[i, 1], masses, G, length, softening)
            all_ke.append(ke)
            all_pe.append(pe)
            all_te.append(te)
        energy_data = {
            'times': timesteps,
            'kinetic': np.array(all_ke),
            'potential': np.array(all_pe),
            'total': np.array(all_te)
        }
    
    # Pre-compute density fields if needed
    density_fields = None
    if video_type == "density":
        print("Pre-computing density fields using JAX batching...")
        from utils import apply_density_scaling
        import jax
        # Create a vectorized version of the density field computation
        @jax.jit
        def compute_single_density(pos):
            density = cic_paint(jnp.zeros((length, length, length)), pos, weight=masses)
            if density_scaling != "none":
                density = apply_density_scaling(density, density_scaling)
            return density
        
        # Vectorize across the time dimension (first axis of sol_ys)
        batch_compute_density = jax.vmap(compute_single_density, in_axes=(0,))
        
        # Compute all density fields at once
        positions_jax = sol_ys[:, 0]  # All positions across timesteps
        density_fields_jax = batch_compute_density(positions_jax)
        
        # Convert to numpy for matplotlib compatibility
        density_fields = np.array(density_fields_jax)
        print(f"Computed {len(density_fields)} density fields with shape {density_fields[0].shape}")
    
    # Create figure based on video type
    if video_type == "particles":
        if enable_energy_tracking:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
            ax_energy = axes[1]
        else:
            fig = plt.figure(figsize=(10, 8))
            ax_3d = fig.add_subplot(111, projection='3d')
            ax_energy = None
        
        # Initialize 3D scatter plot
        scatter = ax_3d.scatter(
            positions[0, :, 0], positions[0, :, 1], positions[0, :, 2], 
            c='blue', alpha=0.6, s=2)
        ax_3d.set_xlim(0, length)
        ax_3d.set_ylim(0, length)
        ax_3d.set_zlim(0, length)
        ax_3d.set_title(f'Particle Positions (t={timesteps[0]:.2f})')
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        
        # Initialize energy plot if needed
        if enable_energy_tracking:
            ax_energy.set_xlabel('Time')
            ax_energy.set_ylabel('Energy')
            ax_energy.set_title('Energy Evolution')
            ax_energy.grid(True, alpha=0.3)
            
            # Create empty lines for energy plots
            ke_line, = ax_energy.plot([], [], 'r-', label='Kinetic')
            pe_line, = ax_energy.plot([], [], 'b-', label='Potential')
            te_line, = ax_energy.plot([], [], 'g-', label='Total')
            ax_energy.legend()
    
    elif video_type == "density":
        if enable_energy_tracking:
            fig = plt.figure(figsize=(16, 8))
            gs = plt.GridSpec(2, 2, figure=fig)
            ax_xy = fig.add_subplot(gs[0, 0])
            ax_xz = fig.add_subplot(gs[0, 1])
            ax_yz = fig.add_subplot(gs[1, 0])
            ax_energy = fig.add_subplot(gs[1, 1])
        else:
            fig = plt.figure(figsize=(15, 5))
            gs = plt.GridSpec(1, 3, figure=fig)
            ax_xy = fig.add_subplot(gs[0, 0])
            ax_xz = fig.add_subplot(gs[0, 1])
            ax_yz = fig.add_subplot(gs[0, 2])
            ax_energy = None
        
        # Initialize density field projections
        initial_xy = np.sum(density_fields[0], axis=0)
        initial_xz = np.sum(density_fields[0], axis=1)
        initial_yz = np.sum(density_fields[0], axis=2)
        
        vmin = min(np.min(initial_xy), np.min(initial_xz), np.min(initial_yz))
        vmax = max(np.max(initial_xy), np.max(initial_xz), np.max(initial_yz))
        
        im_xy = ax_xy.imshow(initial_xy, origin='lower', cmap='inferno', 
                           extent=[0, length, 0, length], vmin=vmin, vmax=vmax)
        im_xz = ax_xz.imshow(initial_xz, origin='lower', cmap='inferno', 
                           extent=[0, length, 0, length], vmin=vmin, vmax=vmax)
        im_yz = ax_yz.imshow(initial_yz, origin='lower', cmap='inferno', 
                           extent=[0, length, 0, length], vmin=vmin, vmax=vmax)
        
        plt.colorbar(im_xy, ax=ax_xy)
        plt.colorbar(im_xz, ax=ax_xz)
        plt.colorbar(im_yz, ax=ax_yz)
        
        ax_xy.set_title(f'Density XY (t={timesteps[0]:.2f})')
        ax_xz.set_title(f'Density XZ (t={timesteps[0]:.2f})')
        ax_yz.set_title(f'Density YZ (t={timesteps[0]:.2f})')
        
        ax_xy.set_xlabel('X'); ax_xy.set_ylabel('Y')
        ax_xz.set_xlabel('X'); ax_xz.set_ylabel('Z')
        ax_yz.set_xlabel('Y'); ax_yz.set_ylabel('Z')
        
        # Initialize energy plot if needed
        if enable_energy_tracking:
            ax_energy.set_xlabel('Time')
            ax_energy.set_ylabel('Energy')
            ax_energy.set_title('Energy Evolution')
            ax_energy.grid(True, alpha=0.3)
            
            ke_line, = ax_energy.plot([], [], 'r-', label='Kinetic')
            pe_line, = ax_energy.plot([], [], 'b-', label='Potential')
            te_line, = ax_energy.plot([], [], 'g-', label='Total')
            ax_energy.legend()
    
    elif video_type == "velocity":
        if enable_energy_tracking:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            ax_vel = axes[0]
            ax_energy = axes[1]
        else:
            fig = plt.figure(figsize=(10, 8))
            ax_vel = plt.gca()
            ax_energy = None
        
        # Calculate initial velocity magnitudes
        vel_magnitudes = np.sqrt(np.sum(velocities[0]**2, axis=1))
        
        # Initialize velocity histogram
        n, bins, patches = ax_vel.hist(vel_magnitudes, bins=50, alpha=0.7)
        ax_vel.set_xlabel('Velocity Magnitude')
        ax_vel.set_ylabel('Count')
        ax_vel.set_title(f'Velocity Distribution (t={timesteps[0]:.2f})')
        
        # Statistics text
        stats_text = ax_vel.text(0.05, 0.95, 
                                 f"Mean: {np.mean(vel_magnitudes):.3f}\nStd: {np.std(vel_magnitudes):.3f}",
                                 transform=ax_vel.transAxes, 
                                 verticalalignment='top',
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Initialize energy plot if needed
        if enable_energy_tracking:
            ax_energy.set_xlabel('Time')
            ax_energy.set_ylabel('Energy')
            ax_energy.set_title('Energy Evolution')
            ax_energy.grid(True, alpha=0.3)
            
            ke_line, = ax_energy.plot([], [], 'r-', label='Kinetic')
            pe_line, = ax_energy.plot([], [], 'b-', label='Potential')
            te_line, = ax_energy.plot([], [], 'g-', label='Total')
            ax_energy.legend()
    
    # Add title with simulation parameters
    param_info = f'G={G}, tf={tf}, dt={dt}, L={length}, N={n_part}, solver={solver}'
    if density_scaling != "none":
        param_info += f', scaling={density_scaling}'
    fig.suptitle(f'N-Body Simulation: {video_type.capitalize()} View\n{param_info}', fontsize=12)
    
    # Animation update function
    def animate(i):
        # Common for all types: update simulation time
        current_time = timesteps[i]
        
        if video_type == "particles":
            # Update particle positions
            scatter._offsets3d = (positions[i, :, 0], positions[i, :, 1], positions[i, :, 2])
            ax_3d.set_title(f'Particle Positions (t={current_time:.2f})')
            
            # Update energy plot if needed
            if enable_energy_tracking:
                shown_idx = min(i, len(energy_data['times'])-1)
                ke_line.set_data(energy_data['times'][:shown_idx+1], energy_data['kinetic'][:shown_idx+1])
                pe_line.set_data(energy_data['times'][:shown_idx+1], energy_data['potential'][:shown_idx+1])
                te_line.set_data(energy_data['times'][:shown_idx+1], energy_data['total'][:shown_idx+1])
                ax_energy.relim()
                ax_energy.autoscale_view()
                
            return [scatter]
            
        elif video_type == "density":
            # Update density projections
            xy_proj = np.sum(density_fields[i], axis=0)
            xz_proj = np.sum(density_fields[i], axis=1)
            yz_proj = np.sum(density_fields[i], axis=2)
            
            im_xy.set_data(xy_proj)
            im_xz.set_data(xz_proj)
            im_yz.set_data(yz_proj)
            
            ax_xy.set_title(f'Density XY (t={current_time:.2f})')
            ax_xz.set_title(f'Density XZ (t={current_time:.2f})')
            ax_yz.set_title(f'Density YZ (t={current_time:.2f})')
            
            # Update energy plot if needed
            if enable_energy_tracking:
                shown_idx = min(i, len(energy_data['times'])-1)
                ke_line.set_data(energy_data['times'][:shown_idx+1], energy_data['kinetic'][:shown_idx+1])
                pe_line.set_data(energy_data['times'][:shown_idx+1], energy_data['potential'][:shown_idx+1])
                te_line.set_data(energy_data['times'][:shown_idx+1], energy_data['total'][:shown_idx+1])
                ax_energy.relim()
                ax_energy.autoscale_view()
                
            return [im_xy, im_xz, im_yz]
            
        elif video_type == "velocity":
            # Update velocity histogram
            vel_magnitudes = np.sqrt(np.sum(velocities[i]**2, axis=1))
            
            # Clear and recreate histogram
            ax_vel.clear()
            ax_vel.hist(vel_magnitudes, bins=50, alpha=0.7)
            ax_vel.set_xlabel('Velocity Magnitude')
            ax_vel.set_ylabel('Count')
            ax_vel.set_title(f'Velocity Distribution (t={current_time:.2f})')
            
            # Update stats text
            stats_text = ax_vel.text(0.05, 0.95, 
                                    f"Mean: {np.mean(vel_magnitudes):.3f}\nStd: {np.std(vel_magnitudes):.3f}",
                                    transform=ax_vel.transAxes, 
                                    verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Update energy plot if needed
            if enable_energy_tracking:
                shown_idx = min(i, len(energy_data['times'])-1)
                ke_line.set_data(energy_data['times'][:shown_idx+1], energy_data['kinetic'][:shown_idx+1])
                pe_line.set_data(energy_data['times'][:shown_idx+1], energy_data['potential'][:shown_idx+1])
                te_line.set_data(energy_data['times'][:shown_idx+1], energy_data['total'][:shown_idx+1])
                ax_energy.relim()
                ax_energy.autoscale_view()
            
            return [stats_text]
    
    # Create animation
    print(f"Creating {video_type} animation with {len(timesteps)} frames...")
    plt.tight_layout()
    anim = animation.FuncAnimation(
        fig, animate, frames=len(timesteps), interval=1000/fps, blit=True)
    
    # Save animation if path provided
    if save_path:
        try:
            print(f"Saving video to {save_path}...")
            if save_path.endswith('.gif'):
                writer = animation.PillowWriter(fps=fps)
                anim.save(save_path, writer=writer)
            else:
                writer = animation.FFMpegWriter(fps=fps, bitrate=800, extra_args=['-vcodec', 'libx264'])
                anim.save(save_path, writer=writer, dpi=dpi)
            print(f"Video successfully saved to: {save_path}")
        except Exception as e:
            print(f"Error saving video: {e}")
            try:
                gif_path = save_path.replace('.mp4', '.gif') if save_path.endswith('.mp4') else save_path + '.gif'
                writer = animation.PillowWriter(fps=fps)
                anim.save(gif_path, writer=writer)
                print(f"Fallback: Video saved as GIF to: {gif_path}")
            except Exception as e2:
                print(f"Failed to save video in any format: {e2}")

    plt.close(fig)
    return anim

def plot_fft_fields(fft_initial, fft_final, length, G, tf, dt, n_part, solver, kpc_per_pixel):
    """
    Plot FFT magnitude projections for initial and final fields (density and velocity components).
    
    Parameters:
    -----------
    fft_initial : array
        Initial FFT fields (density, vx, vy, vz) with shape (..., 4)
    fft_final : array
        Final FFT fields (density, vx, vy, vz) with shape (..., 4)
    length : float
        Box size
    G, tf, dt, n_part, solver : simulation parameters
    kpc_per_pixel : float
        Physical scale conversion
    """
    
    # Extract FFT components
    fft_initial_density = fft_initial[..., 0]
    fft_initial_vx = fft_initial[..., 1]
    fft_final_density = fft_final[..., 0]
    fft_final_vx = fft_final[..., 1]
    
    # Create figure with grid layout
    fig = plt.figure(figsize=(36, 12))
    gs = plt.GridSpec(2, 6, figure=fig)
    
    title = 'FFT Magnitude Fields with parameters:'
    param_info = f'G={G}, tf={tf:.2f} Gyr, dt={dt:.2f} Gyr, L={length*kpc_per_pixel:.1f} kpc, N={n_part}, solver={solver}'
    title += f'\n{param_info}'
    
    fig.suptitle(title, fontsize=20)
    fig.subplots_adjust(top=0.90)
    
    # Calculate FFT magnitudes and apply log scaling for better visualization
    def get_fft_magnitude_log(fft_field):
        magnitude = jnp.abs(fft_field)
        return jnp.log10(magnitude + 1e-10)  # Add small value to avoid log(0)
    
    # Get k-space extent for proper axis labeling
    k_max = jnp.pi * length / kpc_per_pixel  # Nyquist frequency in physical units
    k_extent = [-k_max, k_max, -k_max, k_max]
    
    # First row: Initial fields
    # Initial density FFT projections
    ax00 = fig.add_subplot(gs[0, 0])
    fft_mag = get_fft_magnitude_log(fft_initial_density)
    im00 = ax00.imshow(jnp.sum(fft_mag, axis=0), cmap='viridis', origin='lower', 
                       extent=k_extent, aspect='equal')
    ax00.set_title('Initial Density FFT (X-Y projection)')
    ax00.set_xlabel('kx [kpc⁻¹]')
    ax00.set_ylabel('ky [kpc⁻¹]')
    cbar00 = fig.colorbar(im00, ax=ax00, orientation='vertical')
    cbar00.set_label('log₁₀|FFT|')
    
    ax01 = fig.add_subplot(gs[0, 1])
    im01 = ax01.imshow(jnp.sum(fft_mag, axis=1), cmap='viridis', origin='lower',
                       extent=k_extent, aspect='equal')
    ax01.set_title('Initial Density FFT (X-Z projection)')
    ax01.set_xlabel('kx [kpc⁻¹]')
    ax01.set_ylabel('kz [kpc⁻¹]')
    cbar01 = fig.colorbar(im01, ax=ax01, orientation='vertical')
    cbar01.set_label('log₁₀|FFT|')
    
    ax02 = fig.add_subplot(gs[0, 2])
    im02 = ax02.imshow(jnp.sum(fft_mag, axis=2), cmap='viridis', origin='lower',
                       extent=k_extent, aspect='equal')
    ax02.set_title('Initial Density FFT (Y-Z projection)')
    ax02.set_xlabel('ky [kpc⁻¹]')
    ax02.set_ylabel('kz [kpc⁻¹]')
    cbar02 = fig.colorbar(im02, ax=ax02, orientation='vertical')
    cbar02.set_label('log₁₀|FFT|')
    
    # Initial vx FFT projections
    ax03 = fig.add_subplot(gs[0, 3])
    fft_mag_vx = get_fft_magnitude_log(fft_initial_vx)
    im03 = ax03.imshow(jnp.sum(fft_mag_vx, axis=0), cmap='plasma', origin='lower',
                       extent=k_extent, aspect='equal')
    ax03.set_title('Initial Vx FFT (X-Y projection)')
    ax03.set_xlabel('kx [kpc⁻¹]')
    ax03.set_ylabel('ky [kpc⁻¹]')
    cbar03 = fig.colorbar(im03, ax=ax03, orientation='vertical')
    cbar03.set_label('log₁₀|FFT|')
    
    ax04 = fig.add_subplot(gs[0, 4])
    im04 = ax04.imshow(jnp.sum(fft_mag_vx, axis=1), cmap='plasma', origin='lower',
                       extent=k_extent, aspect='equal')
    ax04.set_title('Initial Vx FFT (X-Z projection)')
    ax04.set_xlabel('kx [kpc⁻¹]')
    ax04.set_ylabel('kz [kpc⁻¹]')
    cbar04 = fig.colorbar(im04, ax=ax04, orientation='vertical')
    cbar04.set_label('log₁₀|FFT|')
    
    ax05 = fig.add_subplot(gs[0, 5])
    im05 = ax05.imshow(jnp.sum(fft_mag_vx, axis=2), cmap='plasma', origin='lower',
                       extent=k_extent, aspect='equal')
    ax05.set_title('Initial Vx FFT (Y-Z projection)')
    ax05.set_xlabel('ky [kpc⁻¹]')
    ax05.set_ylabel('kz [kpc⁻¹]')
    cbar05 = fig.colorbar(im05, ax=ax05, orientation='vertical')
    cbar05.set_label('log₁₀|FFT|')
    
    # Second row: Final fields
    # Final density FFT projections
    ax10 = fig.add_subplot(gs[1, 0])
    fft_mag_final = get_fft_magnitude_log(fft_final_density)
    im10 = ax10.imshow(jnp.sum(fft_mag_final, axis=0), cmap='viridis', origin='lower',
                       extent=k_extent, aspect='equal')
    ax10.set_title('Final Density FFT (X-Y projection)')
    ax10.set_xlabel('kx [kpc⁻¹]')
    ax10.set_ylabel('ky [kpc⁻¹]')
    cbar10 = fig.colorbar(im10, ax=ax10, orientation='vertical')
    cbar10.set_label('log₁₀|FFT|')
    
    ax11 = fig.add_subplot(gs[1, 1])
    im11 = ax11.imshow(jnp.sum(fft_mag_final, axis=1), cmap='viridis', origin='lower',
                       extent=k_extent, aspect='equal')
    ax11.set_title('Final Density FFT (X-Z projection)')
    ax11.set_xlabel('kx [kpc⁻¹]')
    ax11.set_ylabel('kz [kpc⁻¹]')
    cbar11 = fig.colorbar(im11, ax=ax11, orientation='vertical')
    cbar11.set_label('log₁₀|FFT|')
    
    ax12 = fig.add_subplot(gs[1, 2])
    im12 = ax12.imshow(jnp.sum(fft_mag_final, axis=2), cmap='viridis', origin='lower',
                       extent=k_extent, aspect='equal')
    ax12.set_title('Final Density FFT (Y-Z projection)')
    ax12.set_xlabel('ky [kpc⁻¹]')
    ax12.set_ylabel('kz [kpc⁻¹]')
    cbar12 = fig.colorbar(im12, ax=ax12, orientation='vertical')
    cbar12.set_label('log₁₀|FFT|')
    
    # Final vx FFT projections
    ax13 = fig.add_subplot(gs[1, 3])
    fft_mag_vx_final = get_fft_magnitude_log(fft_final_vx)
    im13 = ax13.imshow(jnp.sum(fft_mag_vx_final, axis=0), cmap='plasma', origin='lower',
                       extent=k_extent, aspect='equal')
    ax13.set_title('Final Vx FFT (X-Y projection)')
    ax13.set_xlabel('kx [kpc⁻¹]')
    ax13.set_ylabel('ky [kpc⁻¹]')
    cbar13 = fig.colorbar(im13, ax=ax13, orientation='vertical')
    cbar13.set_label('log₁₀|FFT|')
    
    ax14 = fig.add_subplot(gs[1, 4])
    im14 = ax14.imshow(jnp.sum(fft_mag_vx_final, axis=1), cmap='plasma', origin='lower',
                       extent=k_extent, aspect='equal')
    ax14.set_title('Final Vx FFT (X-Z projection)')
    ax14.set_xlabel('kx [kpc⁻¹]')
    ax14.set_ylabel('kz [kpc⁻¹]')
    cbar14 = fig.colorbar(im14, ax=ax14, orientation='vertical')
    cbar14.set_label('log₁₀|FFT|')
    
    ax15 = fig.add_subplot(gs[1, 5])
    im15 = ax15.imshow(jnp.sum(fft_mag_vx_final, axis=2), cmap='plasma', origin='lower',
                       extent=k_extent, aspect='equal')
    ax15.set_title('Final Vx FFT (Y-Z projection)')
    ax15.set_xlabel('ky [kpc⁻¹]')
    ax15.set_ylabel('kz [kpc⁻¹]')
    cbar15 = fig.colorbar(im15, ax=ax15, orientation='vertical')
    cbar15.set_label('log₁₀|FFT|')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_power_spectra(k_center_initial, power_spectrum_initial, k_center_final, power_spectrum_final, 
                      length, G, tf, dt, n_part, solver, kpc_per_pixel):
    """
    Plot 1D power spectra for all field components (density, vx, vy, vz).
    
    Parameters:
    -----------
    k_center_initial : array
        K-values for initial power spectrum
    power_spectrum_initial : array
        Initial power spectrum values for all fields
    k_center_final : array
        K-values for final power spectrum  
    power_spectrum_final : array
        Final power spectrum values for all fields
    length, G, tf, dt, n_part, solver : simulation parameters
    kpc_per_pixel : float
        Physical scale conversion
    """
    
    field_names = ['Density', 'Vx', 'Vy', 'Vz']
    colors_initial = ['blue', 'red', 'green', 'orange']
    colors_final = ['navy', 'darkred', 'darkgreen', 'darkorange']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    title = 'Power Spectra Evolution with parameters:'
    param_info = f'G={G}, tf={tf:.2f} Gyr, dt={dt:.2f} Gyr, L={length*kpc_per_pixel:.1f} kpc, N={n_part}, solver={solver}'
    title += f'\n{param_info}'
    fig.suptitle(title, fontsize=16)
    
    for i, (field_name, ax) in enumerate(zip(field_names, axes)):
        # Extract power spectra for this field
        k_init = k_center_initial[..., i]
        power_init = power_spectrum_initial[..., i]
        k_final = k_center_final[..., i]
        power_final = power_spectrum_final[..., i]
        
        # Filter out zero or negative values for log plotting
        mask_init = (k_init > 0) & (power_init > 0)
        mask_final = (k_final > 0) & (power_final > 0)
        
        # Plot initial and final power spectra
        ax.loglog(k_init[mask_init], power_init[mask_init], 
                 color=colors_initial[i], linewidth=2, label=f'Initial {field_name}', marker='o', markersize=4)
        ax.loglog(k_final[mask_final], power_final[mask_final], 
                 color=colors_final[i], linewidth=2, label=f'Final {field_name}', marker='s', markersize=4)
        
        ax.set_xlabel('k [kpc⁻¹]')
        ax.set_ylabel('P(k)')
        ax.set_title(f'{field_name} Power Spectrum')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add some reference lines for common power law slopes
        if i == 0:  # Only for density field
            k_ref = k_init[mask_init]
            if len(k_ref) > 0:
                k_min, k_max = k_ref.min(), k_ref.max()
                k_range = jnp.logspace(jnp.log10(k_min), jnp.log10(k_max), 10)
                
                # Add k^-3 reference line (typical for dark matter)
                power_ref = power_init[mask_init].max() * (k_range / k_range[0])**(-3)
                ax.loglog(k_range, power_ref, '--', color='gray', alpha=0.5, label='k⁻³ reference')
                
                ax.legend()
    
    # Additional subplot for comparison of all fields at final time
    fig2, ax_comp = plt.subplots(1, 1, figsize=(12, 8))
    
    for i, field_name in enumerate(field_names):
        k_final = k_center_final[..., i]
        power_final = power_spectrum_final[..., i]
        mask_final = (k_final > 0) & (power_final > 0)
        
        ax_comp.loglog(k_final[mask_final], power_final[mask_final], 
                      color=colors_final[i], linewidth=2, label=f'{field_name}', 
                      marker='o', markersize=4)
    
    ax_comp.set_xlabel('k [kpc⁻¹]', fontsize=14)
    ax_comp.set_ylabel('P(k)', fontsize=14)
    ax_comp.set_title(f'Final Power Spectra Comparison\n{param_info}', fontsize=14)
    ax_comp.grid(True, alpha=0.3)
    ax_comp.legend(fontsize=12)
    
    plt.tight_layout()
    return fig, fig2

