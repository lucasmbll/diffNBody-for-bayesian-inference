# src/plotting.py

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless servers
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jaxpm.painting import cic_paint
import corner
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2


### Plotting functions for DiffNBody simulations

def calculate_energy(pos, vel, G, length, softening, m_part):
    """Calculate kinetic, potential, and total energy of the system"""
    n_particles = pos.shape[0]
    
    # Kinetic energy
    ke = 0.5 * m_part * jnp.sum(vel**2)
    
    # Potential energy with periodic boundaries
    dx = pos[:, None, :] - pos[None, :, :]
    dx = dx - length * jnp.round(dx / length)  # periodic boundaries
    r2 = jnp.sum(dx**2, axis=-1) + softening**2  # softening squared
    r = jnp.sqrt(r2)
    
    # Upper triangular part to avoid double counting, exclude diagonal
    mask = jnp.triu(jnp.ones((n_particles, n_particles)), k=1)
    pe = -G * m_part * m_part * jnp.sum(mask / r)
    
    total_energy = ke + pe
    
    return ke, pe, total_energy

def plot_density_fields_and_positions(G, tf, dt, length, n_part, input_field, init_pos, final_pos, output_field, random_vel, density_scaling, save_path=None):
    """
    Plot density fields and particle positions in various projections.

    Parameters:
    -----------
    input_field : array
        The input density field (3D array) - already scaled
    init_pos : array
        Initial particle positions (N x 3 array)
    final_pos : array
        Final particle positions (N x 3 array)
    output_field : array
        The output density field (3D array) - already scaled
    random_vel : bool
        Whether random velocities were used in the simulation
    density_scaling : str
        Type of density scaling applied
    """
    # Create figure with a grid layout
    fig = plt.figure(figsize=(25, 20))
    gs = plt.GridSpec(4, 4, figure=fig)

    title = 'Simulation with parameters:'
    param_info = f'G={G}, tf={tf}, dt={dt}, L={length}, N={n_part}, random_vel={random_vel}, density_scaling={density_scaling}'

    title += f'\n{param_info}'
    fig.suptitle(title, y=1.0, fontsize=22)

    # Determine colorbar label based on scaling
    if density_scaling == "log":
        cbar_label = "Log Density"
    elif density_scaling == "sqrt":
        cbar_label = "√Density"
    elif density_scaling == "normalize":
        cbar_label = "Normalized Density"
    elif density_scaling == "standardize":
        cbar_label = "Standardized Density"
    elif density_scaling == "power":
        cbar_label = "Power-scaled Density"
    else:
        cbar_label = "Density"

    # First row: Input density field (3 plots)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(jnp.sum(input_field, axis=0), cmap='inferno')
    ax1.set_title(f'Input {cbar_label} Field (Projection X-Y)')
    fig.colorbar(im1, ax=ax1, orientation='vertical')

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(jnp.sum(input_field, axis=1), cmap='inferno')
    ax2.set_title(f'Input {cbar_label} Field (Projection X-Z)')
    fig.colorbar(im2, ax=ax2, orientation='vertical')

    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(jnp.sum(input_field, axis=2), cmap='inferno')
    ax3.set_title(f'Input {cbar_label} Field (Projection Y-Z)')
    fig.colorbar(im3, ax=ax3, orientation='vertical')

    # Empty plot in top-right corner
    fig.add_subplot(gs[0, 3]).set_visible(False)

    # Second row: Initial positions (4 plots)
    ax4 = fig.add_subplot(gs[1, 0], projection='3d')
    ax4.scatter(init_pos[:, 0], init_pos[:, 1], init_pos[:, 2], c='r', marker='o', alpha=0.5, s=1)
    ax4.set_title('Initial Particle Positions (3D)')
    ax4.set_xlabel('X'); ax4.set_ylabel('Y'); ax4.set_zlabel('Z')

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(init_pos[:, 0], init_pos[:, 1], c='r', marker='o', alpha=0.5, s=1)
    ax5.set_title('Initial Positions (X-Y)')
    ax5.set_xlabel('X'); ax5.set_ylabel('Y')

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(init_pos[:, 0], init_pos[:, 2], c='r', marker='o', alpha=0.5, s=1)
    ax6.set_title('Initial Positions (X-Z)')
    ax6.set_xlabel('X'); ax6.set_ylabel('Z')

    ax7 = fig.add_subplot(gs[1, 3])
    ax7.scatter(init_pos[:, 1], init_pos[:, 2], c='r', marker='o', alpha=0.5, s=1)
    ax7.set_title('Initial Positions (Y-Z)')
    ax7.set_xlabel('Y'); ax7.set_ylabel('Z')

    # Third row: Final positions (4 plots)
    ax8 = fig.add_subplot(gs[2, 0], projection='3d')
    ax8.scatter(final_pos[:, 0], final_pos[:, 1], final_pos[:, 2], c='b', marker='o', alpha=0.5, s=1)
    ax8.set_title('Final Particle Positions (3D)')
    ax8.set_xlabel('X'); ax8.set_ylabel('Y'); ax8.set_zlabel('Z')

    ax9 = fig.add_subplot(gs[2, 1])
    ax9.scatter(final_pos[:, 0], final_pos[:, 1], c='b', marker='o', alpha=0.5, s=1)
    ax9.set_title('Final Positions (X-Y)')
    ax9.set_xlabel('X'); ax9.set_ylabel('Y')

    ax10 = fig.add_subplot(gs[2, 2])
    ax10.scatter(final_pos[:, 0], final_pos[:, 2], c='b', marker='o', alpha=0.5, s=1)
    ax10.set_title('Final Positions (X-Z)')
    ax10.set_xlabel('X'); ax10.set_ylabel('Z')

    ax11 = fig.add_subplot(gs[2, 3])
    ax11.scatter(final_pos[:, 1], final_pos[:, 2], c='b', marker='o', alpha=0.5, s=1)
    ax11.set_title('Final Positions (Y-Z)')
    ax11.set_xlabel('Y'); ax11.set_ylabel('Z')

    # Fourth row: Output density field (3 plots)
    ax12 = fig.add_subplot(gs[3, 0])
    im12 = ax12.imshow(jnp.sum(output_field, axis=0), cmap='inferno')
    ax12.set_title(f'Output {cbar_label} Field (Projection X-Y)')
    fig.colorbar(im12, ax=ax12, orientation='vertical')

    ax13 = fig.add_subplot(gs[3, 1])
    im13 = ax13.imshow(jnp.sum(output_field, axis=1), cmap='inferno')
    ax13.set_title(f'Output {cbar_label} Field (Projection X-Z)')
    fig.colorbar(im13, ax=ax13, orientation='vertical')

    ax14 = fig.add_subplot(gs[3, 2])
    im14 = ax14.imshow(jnp.sum(output_field, axis=2), cmap='inferno')
    ax14.set_title(f'Output {cbar_label} Field (Projection Y-Z)')
    fig.colorbar(im14, ax=ax14, orientation='vertical')

    # Empty plot in bottom-right corner
    fig.add_subplot(gs[3, 3]).set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.show()
    return fig

def plot_timesteps(sol,
                   boxL,
                   G,    
                   tf,
                   dt,
                   n_part,
                   random_vel,
                   softening=0.1,
                   m_part=1.0,
                   num_timesteps=10,      # Default to 10 timesteps
                   s=1,                   # marker size for scatters
                   cmap='inferno',
                   enable_energy_tracking=True,  # New parameter to enable/disable energy tracking
                   density_scaling="none",  # Add density scaling parameter
                   save_path=None):
    """
    Parameters
    ----------
    sol   : diffrax solution object  (ys shape = (T, 2, N, 3))
    boxL  : box length used in cic_paint
    num_timesteps : int, number of timesteps to plot
    s     : matplotlib scatter size
    random_vel : bool
        Whether random velocities were used in the simulation
    enable_energy_tracking : bool
        Whether to calculate and plot energy evolution (can be slow for large simulations)
    density_scaling : str
        Type of density scaling applied
    """
    total_timesteps = len(sol.ts)
    
    # Determine skip based on num_timesteps
    if num_timesteps >= total_timesteps:
        skip = 1
    else:
        skip = max(1, total_timesteps // num_timesteps)
    
    # How many rows will we draw?
    steps = sol.ts[::skip]       # time steps to plot
    nrows = len(steps)

    # Pre-calculate energies only if energy tracking is enabled
    all_times = sol.ts
    all_ke = all_pe = all_te = None
    
    if enable_energy_tracking:
        print("Calculating energies for all timesteps...")
        all_ke = []
        all_pe = []
        all_te = []
        
        for i in range(len(sol.ts)):
            pos_t = sol.ys[i, 0]
            vel_t = sol.ys[i, 1]
            ke, pe, te = calculate_energy(pos_t, vel_t, G, boxL, softening, m_part)
            all_ke.append(ke)
            all_pe.append(pe)
            all_te.append(te)
        
        all_ke = jnp.array(all_ke)
        all_pe = jnp.array(all_pe)
        all_te = jnp.array(all_te)

    # Pre-build the canvas - number of columns depends on energy tracking
    ncols = 5 if enable_energy_tracking else 4
    fig, axes = plt.subplots(nrows=nrows,
                             ncols=ncols,
                             figsize=(6*ncols, 4 * nrows),
                             squeeze=False)
    
    # Add global title with simulation parameters
    title = 'Simulation with parameters:'
    param_info = f'G={G}, tf={tf}, dt={dt}, L={boxL}, N={n_part}, random_vel={random_vel}, density_scaling={density_scaling}'
    if enable_energy_tracking:
        param_info += ', energy_tracking=True'
    
    param_info += f', Plotting {len(steps)} timesteps'
    title += f'\n{param_info}'
    fig.suptitle(title, y=1.0, fontsize=22)

    for row, t in enumerate(steps): 
        current_step = row * skip
        
        # Check if sol.ys is a tuple (central mass case) or array (first simulation)
        if isinstance(sol.ys, tuple):
            pos_t = sol.ys[0][current_step]  # Account for skip when indexing
            vel_t = sol.ys[1][current_step]
        else:
            pos_t = sol.ys[current_step][0]
            vel_t = sol.ys[current_step][1]
        
        # --- projections ----------------------------------------------------
        axes[row, 0].scatter(pos_t[:, 0], pos_t[:, 1], s=s)
        axes[row, 1].scatter(pos_t[:, 0], pos_t[:, 2], s=s)
        axes[row, 2].scatter(pos_t[:, 1], pos_t[:, 2], s=s)

        for col, lbl in zip(range(3),
                            ['XY', 'XZ', 'YZ']):
            ax = axes[row, col]
            ax.set_title(f't={t:.2f}  ({lbl})')
            ax.set_xlabel(lbl[0]); ax.set_ylabel(lbl[1])
            ax.set_xlim(0, boxL);  ax.set_ylim(0, boxL)

        # --- density slice --------------------------------------------------
        field_t = cic_paint(jnp.zeros((boxL, boxL, boxL)), pos_t)
        # Apply the same scaling as used in the model
        if density_scaling != "none":
            from model import apply_density_scaling
            field_t = apply_density_scaling(field_t, density_scaling)
        
        im = axes[row, 3].imshow(jnp.sum(field_t, axis=2),
                                 cmap=cmap, origin='lower')
        
        # Determine title based on scaling
        if density_scaling == "none":
            field_title = f'Density Field (t={t:.2f})'
        else:
            field_title = f'{density_scaling.capitalize()}-scaled Density Field (t={t:.2f})'
        
        axes[row, 3].set_title(field_title)
        fig.colorbar(im, ax=axes[row, 3], orientation='vertical')
        
        # --- energy evolution plot (only if enabled) --------------------
        if enable_energy_tracking:
            # Plot energy evolution up to current time
            time_mask = all_times <= t
            
            axes[row, 4].plot(all_times[time_mask], all_ke[time_mask], 'b-', label='Kinetic', linewidth=2)
            axes[row, 4].plot(all_times[time_mask], all_pe[time_mask], 'r-', label='Potential', linewidth=2)
            axes[row, 4].plot(all_times[time_mask], all_te[time_mask], 'k-', label='Total', linewidth=2)
            
            # Mark current time with vertical line
            axes[row, 4].axvline(x=t, color='gray', linestyle='--', alpha=0.7)
            
            axes[row, 4].set_xlabel('Time')
            axes[row, 4].set_ylabel('Energy')
            axes[row, 4].set_title(f'Energy Evolution (t={t:.2f})')
            axes[row, 4].legend()
            axes[row, 4].grid(True, alpha=0.3)
            
            # Set consistent y-limits for energy plot
            energy_min = min(jnp.min(all_ke), jnp.min(all_pe), jnp.min(all_te))
            energy_max = max(jnp.max(all_ke), jnp.max(all_pe), jnp.max(all_te))
            energy_range = energy_max - energy_min
            if energy_range > 0:
                axes[row, 4].set_ylim(energy_min - 0.1*energy_range, energy_max + 0.1*energy_range)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.show()
    return fig, axes



def plot_trajectories(solution, G, tf, dt, length, n_part, random_vel, num_trajectories=10, figsize=(20, 5), zoom=True, padding=0.1, smooth_window=5):    
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

    positions = solution.ys[:, 0]
    num_steps, n_particles, _ = positions.shape

    # Select random trajectories to plot
    if num_trajectories > n_particles:
        num_trajectories = n_particles
    trajectory_indices = np.random.choice(n_particles, num_trajectories, replace=False)

    fig = plt.figure(figsize=figsize)
    ax_3d = fig.add_subplot(1, 4, 1, projection='3d')
    ax_xy = fig.add_subplot(1, 4, 2)
    ax_xz = fig.add_subplot(1, 4, 3)
    ax_yz = fig.add_subplot(1, 4, 4)

    # Create parameter info string with v_circ
    param_info = f'G={G}, tf={tf}, dt={dt}, L={length}, N={n_part}, random_vel={random_vel}'
    
    title = param_info

    ax_3d_title = '3D Trajectories'
    ax_xy.set_title('XY Projection')
    ax_xz.set_title('XZ Projection')
    ax_yz.set_title('YZ Projection')

    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_xy.set_xlabel('X')
    ax_xy.set_ylabel('Y')
    ax_xz.set_xlabel('X')
    ax_xz.set_ylabel('Z')
    ax_yz.set_xlabel('Y')
    ax_yz.set_ylabel('Z')

    # Plot final positions of all particles in background with transparency
    final_pos_all = positions[-1, :]
    ax_3d.scatter(final_pos_all[:, 0], final_pos_all[:, 1], final_pos_all[:, 2], 
                  color='lightgray', alpha=0.2, s=5)
    ax_xy.scatter(final_pos_all[:, 0], final_pos_all[:, 1], color='lightgray', alpha=0.2, s=5)
    ax_xz.scatter(final_pos_all[:, 0], final_pos_all[:, 2], color='lightgray', alpha=0.2, s=5)
    ax_yz.scatter(final_pos_all[:, 1], final_pos_all[:, 2], color='lightgray', alpha=0.2, s=5)

    colors = plt.cm.jet(np.linspace(0, 1, len(trajectory_indices)))
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')
    

    for i, p_idx in enumerate(trajectory_indices):
        traj = positions[:, p_idx]
        if num_steps < 4 * length:
            traj = smooth_trajectory(traj, window=smooth_window)
            ax_3d_title += f' (smoothed, window={smooth_window})'
        ax_3d.set_title(ax_3d_title)

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
        ax_3d.set_xlim([0, length])
        ax_3d.set_ylim([0, length])
        ax_3d.set_zlim([0, length])
        ax_xy.set_xlim([0, length])
        ax_xy.set_ylim([0, length])
        ax_xz.set_xlim([0, length])
        ax_xz.set_ylim([0, length])
        ax_yz.set_xlim([0, length])
        ax_yz.set_ylim([0, length])

    ax_xy.grid(True)
    ax_xz.grid(True)
    ax_yz.grid(True)

    fig.suptitle(title, fontsize=16)
    start_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8, label='Start')
    end_marker = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='k', markersize=8, label='End')
    background_marker = plt.Line2D([0], [0], marker='o', color='lightgray', markersize=8, label='All Particles')
    ax_3d.legend(handles=[start_marker, end_marker, background_marker], loc='upper right')
    plt.tight_layout()
    return fig


def plot_velocity_distributions(sol, G, tf, dt, length, n_part, random_vel, save_path=None):
    # Calculate velocity norms for initial and final velocities
    init_vel = sol.ys[0, 1]  # Initial velocities
    init_vel_norm = jnp.sqrt(jnp.sum(init_vel**2, axis=1))
    final_vel = sol.ys[-1, 1]  # Final velocities
    final_vel_norm = jnp.sqrt(jnp.sum(final_vel**2, axis=1))

    # Create a figure with 2 rows and 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    title = 'Velocity Distribution Comparison with parameters:'
    param_info = f'G={G}, tf={tf}, dt={dt}, L={length}, N={n_part}, random_vel={random_vel}'
    
    title += f'\n{param_info}'
    fig.suptitle(title, y=1.05, fontsize=22)

    # First row - Initial velocities
    # Histogram of velocity magnitudes
    axes[0, 0].hist(init_vel_norm, bins=50, color='blue', alpha=0.7, density=True)
    axes[0, 0].set_xlabel('Velocity Magnitude', fontsize=12)
    axes[0, 0].set_ylabel('Density', fontsize=12)
    axes[0, 0].set_title('Initial Velocity Magnitudes', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(0.05, 0.95, 
        f"Mean={jnp.mean(init_vel_norm):.2f}\nStd={jnp.std(init_vel_norm):.2f}",
        transform=axes[0, 0].transAxes, 
        bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment='top')

    # Histograms for vx, vy, vz components of initial velocities
    axes[0, 1].hist(init_vel[:, 0], bins=50, color='blue', alpha=0.7, density=True)
    axes[0, 1].set_xlabel('Vx', fontsize=12)
    axes[0, 1].set_ylabel('Density', fontsize=12)
    axes[0, 1].set_title('Initial Vx Distribution', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].text(0.05, 0.95, 
        f"Mean={jnp.mean(init_vel[:, 0]):.2f}\nStd={jnp.std(init_vel[:, 0]):.2f}",
        transform=axes[0, 1].transAxes, 
        bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment='top')

    axes[0, 2].hist(init_vel[:, 1], bins=50, color='blue', alpha=0.7, density=True)
    axes[0, 2].set_xlabel('Vy', fontsize=12)
    axes[0, 2].set_ylabel('Density', fontsize=12)
    axes[0, 2].set_title('Initial Vy Distribution', fontsize=14)
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].text(0.05, 0.95, 
        f"Mean={jnp.mean(init_vel[:, 1]):.2f}\nStd={jnp.std(init_vel[:, 1]):.2f}",
        transform=axes[0, 2].transAxes, 
        bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment='top')

    axes[0, 3].hist(init_vel[:, 2], bins=50, color='blue', alpha=0.7, density=True)
    axes[0, 3].set_xlabel('Vz', fontsize=12)
    axes[0, 3].set_ylabel('Density', fontsize=12)
    axes[0, 3].set_title('Initial Vz Distribution', fontsize=14)
    axes[0, 3].grid(True, alpha=0.3)
    axes[0, 3].text(0.05, 0.95, 
        f"Mean={jnp.mean(init_vel[:, 2]):.2f}\nStd={jnp.std(init_vel[:, 2]):.2f}",
        transform=axes[0, 3].transAxes, 
        bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment='top')

    # Second row - Final velocities
    # Histogram of velocity magnitudes
    axes[1, 0].hist(final_vel_norm, bins=50, color='red', alpha=0.7, density=True)
    axes[1, 0].set_xlabel('Velocity Magnitude', fontsize=12)
    axes[1, 0].set_ylabel('Density', fontsize=12)
    axes[1, 0].set_title('Final Velocity Magnitudes', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].text(0.05, 0.95, 
        f"Mean={jnp.mean(final_vel_norm):.2f}\nStd={jnp.std(final_vel_norm):.2f}",
        transform=axes[1, 0].transAxes, 
        bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment='top')

    # Histograms for vx, vy, vz components of final velocities
    axes[1, 1].hist(final_vel[:, 0], bins=50, color='red', alpha=0.7, density=True)
    axes[1, 1].set_xlabel('Vx', fontsize=12)
    axes[1, 1].set_ylabel('Density', fontsize=12)
    axes[1, 1].set_title('Final Vx Distribution', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].text(0.05, 0.95, 
        f"Mean={jnp.mean(final_vel[:, 0]):.2f}\nStd={jnp.std(final_vel[:, 0]):.2f}",
        transform=axes[1, 1].transAxes, 
        bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment='top')

    axes[1, 2].hist(final_vel[:, 1], bins=50, color='red', alpha=0.7, density=True)
    axes[1, 2].set_xlabel('Vy', fontsize=12)
    axes[1, 2].set_ylabel('Density', fontsize=12)
    axes[1, 2].set_title('Final Vy Distribution', fontsize=14)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].text(0.05, 0.95, 
        f"Mean={jnp.mean(final_vel[:, 1]):.2f}\nStd={jnp.std(final_vel[:, 1]):.2f}",
        transform=axes[1, 2].transAxes, 
        bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment='top')

    axes[1, 3].hist(final_vel[:, 2], bins=50, color='red', alpha=0.7, density=True)
    axes[1, 3].set_xlabel('Vz', fontsize=12)
    axes[1, 3].set_ylabel('Density', fontsize=12)
    axes[1, 3].set_title('Final Vz Distribution', fontsize=14)
    axes[1, 3].grid(True, alpha=0.3)
    axes[1, 3].text(0.05, 0.95, 
        f"Mean={jnp.mean(final_vel[:, 2]):.2f}\nStd={jnp.std(final_vel[:, 2]):.2f}",
        transform=axes[1, 3].transAxes, 
        bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment='top')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Adjust for the suptitle
    if save_path:
        fig.savefig(save_path)
    plt.show()
    
    return fig, axes

def create_video(sol, length, G, t_f, dt, n_part, random_vel, softening=0.1, m_part=1.0, 
                enable_energy_tracking=True, save_path=None, fps=10, dpi=100):
    """
    Create a video showing particle evolution over time.
    
    Parameters:
    -----------
    sol : diffrax solution object
        Solution containing particle trajectories
    length : float
        Box size
    G : float
        Gravitational constant
    t_f : float
        Final time
    dt : float
        Time step
    n_part : int
        Number of particles
    random_vel : bool
        Whether random velocities were used in the simulation
    softening : float
        Softening parameter
    m_part : float
        Particle mass
    enable_energy_tracking : bool
        Whether to calculate and plot energy evolution (can be slow for large simulations)
    save_path : str
        Path to save the video file
    fps : int
        Frames per second for the video
    dpi : int
        Resolution of the video frames
    """
    import matplotlib.animation as animation
    import shutil
    
    # Check if ffmpeg is available
    if not shutil.which('ffmpeg'):
        print("Warning: ffmpeg not found. Trying alternative video writers...")
        if save_path and save_path.endswith('.mp4'):
            save_path = save_path.replace('.mp4', '.gif')
            print(f"Saving as GIF instead: {save_path}")
    
    print(f"Creating video with {len(sol.ts)} frames...")
    
    # Pre-calculate all energies only if energy tracking is enabled
    all_times = []
    all_ke = all_pe = all_te = None
    
    if enable_energy_tracking:
        print("Pre-calculating energies for all frames...")
        all_ke = []
        all_pe = []
        all_te = []
        
        for i in range(len(sol.ts)):
            pos = sol.ys[i, 0]
            vel = sol.ys[i, 1]
            ke, pe, te = calculate_energy(pos, vel, G, length, softening, m_part)
            all_times.append(sol.ts[i])
            all_ke.append(ke)
            all_pe.append(pe)
            all_te.append(te)
        
        all_ke = jnp.array(all_ke)
        all_pe = jnp.array(all_pe)
        all_te = jnp.array(all_te)
    else:
        all_times = sol.ts
    
    print("Energy calculation complete. Creating animation...")
    
    # Create figure and subplots - layout depends on energy tracking
    if enable_energy_tracking:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Create title
    title = f'N-Body Simulation (G={G}, tf={t_f}, dt={dt}, L={length}, N={n_part}, random_vel={random_vel}'
    if enable_energy_tracking:
        title += ', energy_tracking=True'
    title += ')'
    fig.suptitle(title, fontsize=14)
    
    def animate(frame):
        # Clear all axes
        if enable_energy_tracking:
            for i in range(2):
                for j in range(3):
                    axes[i, j].clear()
        else:
            for i in range(2):
                for j in range(2):
                    axes[i, j].clear()
        
        # Get positions and velocities for current frame
        pos = sol.ys[frame, 0]
        vel = sol.ys[frame, 1]
        current_time = sol.ts[frame]
        
        # 3D scatter plot
        ax_3d = axes[0, 0]
        ax_3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='blue', s=1, alpha=0.6)
        ax_3d.set_xlim(0, length)
        ax_3d.set_ylim(0, length)
        ax_3d.set_zlim(0, length)
        ax_3d.set_title(f'3D Positions (t={current_time:.3f})')
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        
        # XY scatter plot
        ax_xy = axes[0, 1]
        ax_xy.scatter(pos[:, 0], pos[:, 1], c='blue', s=1, alpha=0.6)
        ax_xy.set_xlim(0, length)
        ax_xy.set_ylim(0, length)
        ax_xy.set_title(f'XY Projection (t={current_time:.3f})')
        ax_xy.set_xlabel('X')
        ax_xy.set_ylabel('Y')
        ax_xy.grid(True, alpha=0.3)
        
        if enable_energy_tracking:
            # XZ scatter plot
            ax_xz = axes[0, 2]
            ax_xz.scatter(pos[:, 0], pos[:, 2], c='blue', s=1, alpha=0.6)
            ax_xz.set_xlim(0, length)
            ax_xz.set_ylim(0, length)
            ax_xz.set_title(f'XZ Projection (t={current_time:.3f})')
            ax_xz.set_xlabel('X')
            ax_xz.set_ylabel('Z')
            ax_xz.grid(True, alpha=0.3)
            
            # Density field
            ax_density = axes[1, 0]
            density_field = cic_paint(jnp.zeros((length, length, length)), pos)
            density_2d = jnp.sum(density_field, axis=2)
            
            im = ax_density.imshow(density_2d, cmap='inferno', origin='lower')
            ax_density.set_title(f'Density Field (t={current_time:.3f})')
            
            # Velocity magnitude histogram
            ax_vel = axes[1, 1]
            vel_magnitudes = jnp.sqrt(jnp.sum(vel**2, axis=1))
            ax_vel.hist(vel_magnitudes, bins=30, color='green', alpha=0.7, density=True)
            ax_vel.set_xlabel('Velocity Magnitude')
            ax_vel.set_ylabel('Density')
            ax_vel.set_title(f'Velocity Distribution (t={current_time:.3f})')
            ax_vel.grid(True, alpha=0.3)
            
            # Energy evolution plot (up to current frame)
            ax_energy = axes[1, 2]
            current_frame_times = all_times[:frame+1]
            current_frame_ke = all_ke[:frame+1]
            current_frame_pe = all_pe[:frame+1]
            current_frame_te = all_te[:frame+1]
            
            ax_energy.plot(current_frame_times, current_frame_ke, 'b-', label='Kinetic', linewidth=2)
            ax_energy.plot(current_frame_times, current_frame_pe, 'r-', label='Potential', linewidth=2)
            ax_energy.plot(current_frame_times, current_frame_te, 'k-', label='Total', linewidth=2)
            
            # Mark current time
            ax_energy.axvline(x=current_time, color='gray', linestyle='--', alpha=0.7)
            
            ax_energy.set_xlabel('Time')
            ax_energy.set_ylabel('Energy')
            ax_energy.set_title('Energy Evolution')
            ax_energy.legend()
            ax_energy.grid(True, alpha=0.3)
            
            # Set consistent y-limits for energy plot
            if len(all_ke) > 0:
                energy_min = min(jnp.min(all_ke), jnp.min(all_pe), jnp.min(all_te))
                energy_max = max(jnp.max(all_ke), jnp.max(all_pe), jnp.max(all_te))
                energy_range = energy_max - energy_min
                if energy_range > 0:
                    ax_energy.set_ylim(energy_min - 0.1*energy_range, energy_max + 0.1*energy_range)
        else:
            # Simpler layout without energy tracking
            # Density field
            ax_density = axes[1, 0]
            density_field = cic_paint(jnp.zeros((length, length, length)), pos)
            density_2d = jnp.sum(density_field, axis=2)
            
            im = ax_density.imshow(density_2d, cmap='inferno', origin='lower')
            ax_density.set_title(f'Density Field (t={current_time:.3f})')
            
            # Velocity magnitude histogram
            ax_vel = axes[1, 1]
            vel_magnitudes = jnp.sqrt(jnp.sum(vel**2, axis=1))
            ax_vel.hist(vel_magnitudes, bins=30, color='green', alpha=0.7, density=True)
            ax_vel.set_xlabel('Velocity Magnitude')
            ax_vel.set_ylabel('Density')
            ax_vel.set_title(f'Velocity Distribution (t={current_time:.3f})')
            ax_vel.grid(True, alpha=0.3)
        
        return []
    
    anim = animation.FuncAnimation(fig, animate, frames=len(sol.ts), interval=100, blit=False, repeat=False)
    
    if save_path:
        try:
            if save_path.endswith('.gif'):
                writer = animation.PillowWriter(fps=fps)
                anim.save(save_path, writer=writer)
            else:
                writer = animation.FFMpegWriter(fps=fps, bitrate=800, extra_args=['-vcodec', 'libx264'])
                anim.save(save_path, writer=writer, dpi=dpi)
        except Exception as e:
            print(f"Error saving video: {e}")
    
    plt.close(fig)
    return anim

### Plotting functions for sampling experiments
def plot_trace_subplots(mcmc_samples, theta, G, t_f, dt, softening, length, n_part, random_vel, figsize=(18, 5), method="HMC", param_order=("sigma", "mean", "vel_sigma"), infer_vel_sigma=True, save_path=None):
    """
    Plot trace plots for parameters with true values as horizontal lines.
    """
    title = 'Sampling of the model parameters distribution with ' + method
    param_info = f'G={G}, tf={t_f}, dt={dt}, L={length}, N={n_part}, softening={softening}, random_vel={random_vel}'

    # Adjust figure size and subplot count based on number of parameters
    n_params = len(param_order)
    fig, axes = plt.subplots(1, n_params, figsize=(6*n_params, 5), sharex=True)
    
    # Handle case where we only have 1 or 2 parameters
    if n_params == 1:
        axes = [axes]
    elif n_params == 2:
        axes = list(axes)
    
    if title or param_info:
        plt.suptitle(f"{title}\n{param_info}", fontsize=14)
    
    colors = ["blue", "green", "orange"]
    true_labels_map = {"sigma": "pos_std", "mean": "pos_mean", "vel_sigma": "vel_std"}
    
    for i, param in enumerate(param_order):
        if param in mcmc_samples and param in true_labels_map:
            axes[i].plot(mcmc_samples[param], label=param, color=colors[i % len(colors)])
            
            true_label = true_labels_map[param]
            if true_label in theta:
                axes[i].axhline(
                    y=theta[true_label],
                    color='r', linestyle='--', alpha=0.5,
                    label=f'True {param}={theta[true_label]}'
                )
            
            axes[i].set_title(f'Sampling of {param.capitalize()}')
            axes[i].set_xlabel('Iteration')
            axes[i].set_ylabel('Parameter value')
            axes[i].legend()
            axes[i].grid(True)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.show()
    return fig, axes

def plot_corner_after_burnin(mcmc_samples, theta, burnin=1000, param_order=("sigma", "mean", "vel_sigma"), title="Posterior distribution of model's parameters", infer_vel_sigma=True, save_path=None):
    """
    Plot a corner plot of posterior samples after burn-in.
    """
    # Build samples array only for parameters we're actually inferring
    samples_list = []
    labels_list = []
    truths_list = []
    
    true_labels_map = {"sigma": "pos_std", "mean": "pos_mean", "vel_sigma": "vel_std"}
    
    for param in param_order:
        if param in mcmc_samples:
            samples_list.append(np.array(mcmc_samples[param])[burnin:])
            labels_list.append(param)
            
            true_label = true_labels_map[param]
            if true_label in theta:
                truths_list.append(theta[true_label])
            else:
                truths_list.append(None)
    
    if not samples_list:
        print("No valid samples found for corner plot")
        return None
    
    samples = np.column_stack(samples_list)
    
    fig = corner.corner(
        samples,
        labels=labels_list,
        truths=truths_list,
        show_titles=True,
        title_kwargs={"fontsize": 12},
        quantiles=[0.16, 0.5, 0.84],
        levels=(0.68, 0.95),
        plot_contours=True,
        fill_contours=True,
        bins=30
    )
    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.90)
    if save_path:
        fig.savefig(save_path)
    plt.show()
    return fig
