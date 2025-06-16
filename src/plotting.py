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
from utils import calculate_energy


### Plotting functions for DiffNBody simulations

def plot_density_fields_and_positions(G, tf, dt, length, n_part, input_field, init_pos, final_pos, output_field, density_scaling, save_path=None):
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
    density_scaling : str
        Type of density scaling applied
    """
    # Create figure with a grid layout
    fig = plt.figure(figsize=(25, 20))
    gs = plt.GridSpec(4, 4, figure=fig)

    title = 'Simulation with parameters:'
    param_info = f'G={G}, tf={tf}, dt={dt}, L={length}, N={n_part}, density_scaling={density_scaling}'

    title += f'\n{param_info}'
    # Place suptitle at the very top
    fig.suptitle(title, fontsize=22)
    fig.subplots_adjust(top=0.92)  # Adjust for the suptitle


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

    # Helper for axis limits
    limits = [0, length]

    # First row: Input density field (3 plots)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(jnp.sum(input_field, axis=0), cmap='inferno', origin='lower',
                     extent=[0, length, 0, length], aspect='auto')
    ax1.set_title(f'Input {cbar_label} Field (Projection X-Y)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_xlim(limits)
    ax1.set_ylim(limits)
    fig.colorbar(im1, ax=ax1, orientation='vertical')

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(jnp.sum(input_field, axis=1), cmap='inferno', origin='lower',
                     extent=[0, length, 0, length], aspect='auto')
    ax2.set_title(f'Input {cbar_label} Field (Projection X-Z)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_xlim(limits)
    ax2.set_ylim(limits)
    fig.colorbar(im2, ax=ax2, orientation='vertical')

    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(jnp.sum(input_field, axis=2), cmap='inferno', origin='lower',
                     extent=[0, length, 0, length], aspect='auto')
    ax3.set_title(f'Input {cbar_label} Field (Projection Y-Z)')
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')
    ax3.set_xlim(limits)
    ax3.set_ylim(limits)
    fig.colorbar(im3, ax=ax3, orientation='vertical')
    fig.add_subplot(gs[0, 3]).set_visible(False)

    # Second row: Initial positions (4 plots)
    ax4 = fig.add_subplot(gs[1, 0], projection='3d')
    ax4.scatter(init_pos[:, 0], init_pos[:, 1], init_pos[:, 2], c='r', marker='o', alpha=0.5, s=1)
    ax4.set_title('Initial Particle Positions (3D)')
    ax4.set_xlabel('X'); ax4.set_ylabel('Y'); ax4.set_zlabel('Z')
    ax4.set_xlim(limits)
    ax4.set_ylim(limits)
    ax4.set_zlim(limits)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(init_pos[:, 0], init_pos[:, 1], c='r', marker='o', alpha=0.5, s=1)
    ax5.set_title('Initial Positions (X-Y)')
    ax5.set_xlabel('X'); ax5.set_ylabel('Y')
    ax5.set_xlim(limits)
    ax5.set_ylim(limits)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(init_pos[:, 0], init_pos[:, 2], c='r', marker='o', alpha=0.5, s=1)
    ax6.set_title('Initial Positions (X-Z)')
    ax6.set_xlabel('X'); ax6.set_ylabel('Z')
    ax6.set_xlim(limits)
    ax6.set_ylim(limits)

    ax7 = fig.add_subplot(gs[1, 3])
    ax7.scatter(init_pos[:, 1], init_pos[:, 2], c='r', marker='o', alpha=0.5, s=1)
    ax7.set_title('Initial Positions (Y-Z)')
    ax7.set_xlabel('Y'); ax7.set_ylabel('Z')
    ax7.set_xlim(limits)
    ax7.set_ylim(limits)

    # Third row: Final positions (4 plots)
    ax8 = fig.add_subplot(gs[2, 0], projection='3d')
    ax8.scatter(final_pos[:, 0], final_pos[:, 1], final_pos[:, 2], c='b', marker='o', alpha=0.5, s=1)
    ax8.set_title('Final Particle Positions (3D)')
    ax8.set_xlabel('X'); ax8.set_ylabel('Y'); ax8.set_zlabel('Z')
    ax8.set_xlim(limits)
    ax8.set_ylim(limits)
    ax8.set_zlim(limits)

    ax9 = fig.add_subplot(gs[2, 1])
    ax9.scatter(final_pos[:, 0], final_pos[:, 1], c='b', marker='o', alpha=0.5, s=1)
    ax9.set_title('Final Positions (X-Y)')
    ax9.set_xlabel('X'); ax9.set_ylabel('Y')
    ax9.set_xlim(limits)
    ax9.set_ylim(limits)

    ax10 = fig.add_subplot(gs[2, 2])
    ax10.scatter(final_pos[:, 0], final_pos[:, 2], c='b', marker='o', alpha=0.5, s=1)
    ax10.set_title('Final Positions (X-Z)')
    ax10.set_xlabel('X'); ax10.set_ylabel('Z')
    ax10.set_xlim(limits)
    ax10.set_ylim(limits)

    ax11 = fig.add_subplot(gs[2, 3])
    ax11.scatter(final_pos[:, 1], final_pos[:, 2], c='b', marker='o', alpha=0.5, s=1)
    ax11.set_title('Final Positions (Y-Z)')
    ax11.set_xlabel('Y'); ax11.set_ylabel('Z')
    ax11.set_xlim(limits)
    ax11.set_ylim(limits)

    # Fourth row: Output density field (3 plots)
    ax12 = fig.add_subplot(gs[3, 0])
    im12 = ax12.imshow(jnp.sum(output_field, axis=0), cmap='inferno', origin='lower',
                       extent=[0, length, 0, length], aspect='auto')
    ax12.set_title(f'Output {cbar_label} Field (Projection X-Y)')
    ax12.set_xlabel('X')
    ax12.set_ylabel('Y')
    ax12.set_xlim(limits)
    ax12.set_ylim(limits)
    fig.colorbar(im12, ax=ax12, orientation='vertical')

    ax13 = fig.add_subplot(gs[3, 1])
    im13 = ax13.imshow(jnp.sum(output_field, axis=1), cmap='inferno', origin='lower',
                       extent=[0, length, 0, length], aspect='auto')
    ax13.set_title(f'Output {cbar_label} Field (Projection X-Z)')
    ax13.set_xlabel('X')
    ax13.set_ylabel('Z')
    ax13.set_xlim(limits)
    ax13.set_ylim(limits)
    fig.colorbar(im13, ax=ax13, orientation='vertical')

    ax14 = fig.add_subplot(gs[3, 2])
    im14 = ax14.imshow(jnp.sum(output_field, axis=2), cmap='inferno', origin='lower',
                       extent=[0, length, 0, length], aspect='auto')
    ax14.set_title(f'Output {cbar_label} Field (Projection Y-Z)')
    ax14.set_xlabel('Y')
    ax14.set_ylabel('Z')
    ax14.set_xlim(limits)
    ax14.set_ylim(limits)
    fig.colorbar(im14, ax=ax14, orientation='vertical')
    fig.add_subplot(gs[3, 3]).set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
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
                   softening,
                   m_part,
                   num_timesteps,
                   s=1,
                   cmap='inferno',
                   enable_energy_tracking=True,
                   density_scaling="none",
                   energy_data=None,  # Pre-computed energy data
                   save_path=None):
    """
    Parameters
    ----------
    sol   : diffrax solution object  (ys shape = (T, 2, N, 3))
    boxL  : box length used in cic_paint
    num_timesteps : int, number of timesteps to plot
    s     : matplotlib scatter size
    enable_energy_tracking : bool
        Whether to calculate and plot energy evolution (can be slow for large simulations)
    density_scaling : str
        Type of density scaling applied
    energy_data : dict, optional
        Pre-computed energy data with keys 'times', 'kinetic', 'potential', 'total'
    """
    total_timesteps = len(sol.ts)
    
    # Determine skip based on num_timesteps
    if num_timesteps >= total_timesteps:
        skip = 1
    else:
        skip = max(1, total_timesteps // num_timesteps)
    
    # How many rows will we draw?
    steps = sol.ts[::skip]
    nrows = len(steps)

    # Use pre-computed energy data if available, otherwise compute if needed
    all_times = sol.ts
    all_ke = all_pe = all_te = None
    
    if enable_energy_tracking:
        if energy_data is not None:
            print("Using pre-computed energy data for timesteps plot...")
            all_times = energy_data['times']
            all_ke = energy_data['kinetic']
            all_pe = energy_data['potential']
            all_te = energy_data['total']
        else:
            print("Computing energies for all timesteps (no pre-computed data provided)...")
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
    param_info = f'G={G}, tf={tf}, dt={dt}, L={boxL}, N={n_part}, density_scaling={density_scaling}'
    if enable_energy_tracking:
        param_info += ', energy_tracking=True'
    
    param_info += f', Plotting {len(steps)} timesteps'
    title += f'\n{param_info}'
    # Place suptitle at the very top
    fig.suptitle(title, fontsize=22)
    fig.subplots_adjust(top=0.96)

    for row, t in enumerate(steps): 
        current_step = row * skip
        pos_t = sol.ys[current_step, 0]
        vel_t = sol.ys[current_step, 1]
        
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

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if save_path:
        fig.savefig(save_path)
    plt.show()
    return fig, axes

def plot_trajectories(solution, G, tf, dt, length, n_part, num_trajectories=10, figsize=(20, 5), zoom=True, padding=0.1, smooth_window=5):    
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

    # Create parameter info string
    param_info = f'G={G}, tf={tf}, dt={dt}, L={length}, N={n_part}'
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

    # Place suptitle at the very top
    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(top=0.98)
    start_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8, label='Start')
    end_marker = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='k', markersize=8, label='End')
    background_marker = plt.Line2D([0], [0], marker='o', color='lightgray', markersize=8, label='All Particles')
    ax_3d.legend(handles=[start_marker, end_marker, background_marker], loc='upper right')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def plot_velocity_distributions(sol, G, tf, dt, length, n_part, save_path=None, quiver_stride=5):
    # Calculate velocity norms for initial and final velocities
    init_vel = sol.ys[0, 1]  # Initial velocities
    init_pos = sol.ys[0, 0]  # Initial positions
    init_vel_norm = jnp.sqrt(jnp.sum(init_vel**2, axis=1))
    final_vel = sol.ys[-1, 1]  # Final velocities
    final_pos = sol.ys[-1, 0]  # Final positions
    final_vel_norm = jnp.sqrt(jnp.sum(final_vel**2, axis=1))

    # Create a figure with 4 rows and 4 columns
    fig, axes = plt.subplots(4, 4, figsize=(24, 18))
    title = 'Velocity Distribution Comparison with parameters:'
    param_info = f'G={G}, tf={tf}, dt={dt}, L={length}, N={n_part}'
    
    title += f'\n{param_info}'
    # Place suptitle at the very top
    fig.suptitle(title, fontsize=22)
    fig.subplots_adjust(top=0.92)

    # First row - Initial velocities (histograms)
    axes[0, 0].hist(init_vel_norm, bins=50, color='blue', alpha=0.7, density=True)
    axes[0, 0].set_xlabel('Velocity Magnitude', fontsize=12)
    axes[0, 0].set_ylabel('Density', fontsize=12)
    axes[0, 0].set_title('Initial Velocity Magnitudes', fontsize=14)
    axes[0, 0].text(0.05, 0.95,
        f"Mean={jnp.mean(init_vel_norm):.2f}\nStd={jnp.std(init_vel_norm):.2f}",
        transform=axes[0, 0].transAxes, 
        bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment='top')

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

    # Second row - Initial velocity vector fields (quiver plots)
    stride = quiver_stride
    idx = np.arange(0, n_part, stride)
    # XY plane
    axes[1, 0].quiver(init_pos[idx, 0], init_pos[idx, 1], init_vel[idx, 0], init_vel[idx, 1], 
                      angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.7)
    axes[1, 0].set_title('Initial Velocity Field (XY)', fontsize=14)
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    axes[1, 0].set_xlim(0, length)
    axes[1, 0].set_ylim(0, length)
    axes[1, 0].grid(True, alpha=0.3)

    # XZ plane
    axes[1, 1].quiver(init_pos[idx, 0], init_pos[idx, 2], init_vel[idx, 0], init_vel[idx, 2], 
                      angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.7)
    axes[1, 1].set_title('Initial Velocity Field (XZ)', fontsize=14)
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Z')
    axes[1, 1].set_xlim(0, length)
    axes[1, 1].set_ylim(0, length)
    axes[1, 1].grid(True, alpha=0.3)

    # YZ plane
    axes[1, 2].quiver(init_pos[idx, 1], init_pos[idx, 2], init_vel[idx, 1], init_vel[idx, 2], 
                      angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.7)
    axes[1, 2].set_title('Initial Velocity Field (YZ)', fontsize=14)
    axes[1, 2].set_xlabel('Y')
    axes[1, 2].set_ylabel('Z')
    axes[1, 2].set_xlim(0, length)
    axes[1, 2].set_ylim(0, length)
    axes[1, 2].grid(True, alpha=0.3)

    # Leave axes[1, 3] empty
    axes[1, 3].axis('off')

    # Third row - Final velocities (histograms)
    axes[2, 0].hist(final_vel_norm, bins=50, color='red', alpha=0.7, density=True)
    axes[2, 0].set_xlabel('Velocity Magnitude', fontsize=12)
    axes[2, 0].set_ylabel('Density', fontsize=12)
    axes[2, 0].set_title('Final Velocity Magnitudes', fontsize=14)
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].text(0.05, 0.95, 
        f"Mean={jnp.mean(final_vel_norm):.2f}\nStd={jnp.std(final_vel_norm):.2f}",
        transform=axes[2, 0].transAxes, 
        bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment='top')

    axes[2, 1].hist(final_vel[:, 0], bins=50, color='red', alpha=0.7, density=True)
    axes[2, 1].set_xlabel('Vx', fontsize=12)
    axes[2, 1].set_ylabel('Density', fontsize=12)
    axes[2, 1].set_title('Final Vx Distribution', fontsize=14)
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].text(0.05, 0.95, 
        f"Mean={jnp.mean(final_vel[:, 0]):.2f}\nStd={jnp.std(final_vel[:, 0]):.2f}",
        transform=axes[2, 1].transAxes, 
        bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment='top')

    axes[2, 2].hist(final_vel[:, 1], bins=50, color='red', alpha=0.7, density=True)
    axes[2, 2].set_xlabel('Vy', fontsize=12)
    axes[2, 2].set_ylabel('Density', fontsize=12)
    axes[2, 2].set_title('Final Vy Distribution', fontsize=14)
    axes[2, 2].grid(True, alpha=0.3)
    axes[2, 2].text(0.05, 0.95, 
        f"Mean={jnp.mean(final_vel[:, 1]):.2f}\nStd={jnp.std(final_vel[:, 1]):.2f}",
        transform=axes[2, 2].transAxes, 
        bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment='top')

    axes[2, 3].hist(final_vel[:, 2], bins=50, color='red', alpha=0.7, density=True)
    axes[2, 3].set_xlabel('Vz', fontsize=12)
    axes[2, 3].set_ylabel('Density', fontsize=12)
    axes[2, 3].set_title('Final Vz Distribution', fontsize=14)
    axes[2, 3].grid(True, alpha=0.3)
    axes[2, 3].text(0.05, 0.95, 
        f"Mean={jnp.mean(final_vel[:, 2]):.2f}\nStd={jnp.std(final_vel[:, 2]):.2f}",
        transform=axes[2, 3].transAxes, 
        bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment='top')

    # Fourth row - Final velocity vector fields (quiver plots)
    # XY plane
    axes[3, 0].quiver(final_pos[idx, 0], final_pos[idx, 1], final_vel[idx, 0], final_vel[idx, 1], 
                      angles='xy', scale_units='xy', scale=1, color='red', alpha=0.7)
    axes[3, 0].set_title('Final Velocity Field (XY)', fontsize=14)
    axes[3, 0].set_xlabel('X')
    axes[3, 0].set_ylabel('Y')
    axes[3, 0].set_xlim(0, length)
    axes[3, 0].set_ylim(0, length)
    axes[3, 0].grid(True, alpha=0.3)

    # XZ plane
    axes[3, 1].quiver(final_pos[idx, 0], final_pos[idx, 2], final_vel[idx, 0], final_vel[idx, 2], 
                      angles='xy', scale_units='xy', scale=1, color='red', alpha=0.7)
    axes[3, 1].set_title('Final Velocity Field (XZ)', fontsize=14)
    axes[3, 1].set_xlabel('X')
    axes[3, 1].set_ylabel('Z')
    axes[3, 1].set_xlim(0, length)
    axes[3, 1].set_ylim(0, length)
    axes[3, 1].grid(True, alpha=0.3)

    # YZ plane
    axes[3, 2].quiver(final_pos[idx, 1], final_pos[idx, 2], final_vel[idx, 1], final_vel[idx, 2], 
                      angles='xy', scale_units='xy', scale=1, color='red', alpha=0.7)
    axes[3, 2].set_title('Final Velocity Field (YZ)', fontsize=14)
    axes[3, 2].set_xlabel('Y')
    axes[3, 2].set_ylabel('Z')
    axes[3, 2].set_xlim(0, length)
    axes[3, 2].set_ylim(0, length)
    axes[3, 2].grid(True, alpha=0.3)

    # Leave axes[3, 3] empty
    axes[3, 3].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust for the suptitle
    if save_path:
        fig.savefig(save_path)
    plt.show()
    
    return fig, axes

def create_video(
    sol, length, G, t_f, dt, n_part, density_scaling, softening=0.1, m_part=1.0, 
    enable_energy_tracking=True, save_path=None, fps=10, dpi=100, energy_data=None
):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import shutil
    import numpy as np
    import jax.numpy as jnp

    # Check ffmpeg
    if not shutil.which('ffmpeg'):
        print("Warning: ffmpeg not found. Trying alternative video writers...")
        if save_path and save_path.endswith('.mp4'):
            save_path = save_path.replace('.mp4', '.gif')
            print(f"Saving as GIF instead: {save_path}")

    # Energy calculation
    all_times = []
    all_ke = all_pe = all_te = None
    if enable_energy_tracking:
        if energy_data is not None:
            print("Using pre-computed energy data for video creation...")
            all_times = energy_data['times']
            all_ke = energy_data['kinetic']
            all_pe = energy_data['potential']
            all_te = energy_data['total']
        else:
            print("Computing energies for all frames (no pre-computed data provided)...")
            all_ke, all_pe, all_te, all_times = [], [], [], []
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

    fig = plt.figure(figsize=(24, 12))
    axes = np.empty((2, 4), dtype=object)
    axes[0, 0] = fig.add_subplot(2, 4, 1, projection='3d')
    axes[0, 1] = fig.add_subplot(2, 4, 2)
    axes[0, 2] = fig.add_subplot(2, 4, 3)
    axes[0, 3] = fig.add_subplot(2, 4, 4)
    axes[1, 0] = fig.add_subplot(2, 4, 5)
    axes[1, 1] = fig.add_subplot(2, 4, 6)
    axes[1, 2] = fig.add_subplot(2, 4, 7)
    axes[1, 3] = fig.add_subplot(2, 4, 8)

    fig.subplots_adjust(wspace=0.4, hspace=0.5)
    title = f'N-Body Simulation (G={G}, tf={t_f}, dt={dt}, L={length}, N={n_part})'
    fig.suptitle(title, y=0.98, fontsize=14)

    # --- INITIALIZE PLOTS (frame 0 data) ---
    pos = sol.ys[0, 0]
    vel = sol.ys[0, 1]
    current_time = sol.ts[0]

    # 3D scatter (frame 0)
    ax_3d = axes[0, 0]
    scatter3d = ax_3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='blue', alpha=0.6, s=1)
    ax_3d.set_xlim(0, length)
    ax_3d.set_ylim(0, length)
    ax_3d.set_zlim(0, length)
    ax_3d.set_title(f'3D Positions (t={current_time:.3f})')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')

    # 2D projections (scatter)
    ax_xy = axes[0, 1]
    scatter_xy = ax_xy.scatter(pos[:, 0], pos[:, 1], c='blue', alpha=0.6, s=1)
    ax_xy.set_xlim(0, length)
    ax_xy.set_ylim(0, length)
    ax_xy.set_title(f'XY Projection (t={current_time:.3f})')
    ax_xy.set_xlabel('X')
    ax_xy.set_ylabel('Y')
    ax_xy.grid(True, alpha=0.3)

    ax_xz = axes[0, 2]
    scatter_xz = ax_xz.scatter(pos[:, 0], pos[:, 2], c='blue', alpha=0.6, s=1)
    ax_xz.set_xlim(0, length)
    ax_xz.set_ylim(0, length)
    ax_xz.set_title(f'XZ Projection (t={current_time:.3f})')
    ax_xz.set_xlabel('X')
    ax_xz.set_ylabel('Z')
    ax_xz.grid(True, alpha=0.3)

    ax_yz = axes[0, 3]
    scatter_yz = ax_yz.scatter(pos[:, 1], pos[:, 2], c='blue', alpha=0.6, s=1)
    ax_yz.set_xlim(0, length)
    ax_yz.set_ylim(0, length)
    ax_yz.set_title(f'YZ Projection (t={current_time:.3f})')
    ax_yz.set_xlabel('Y')
    ax_yz.set_ylabel('Z')
    ax_yz.grid(True, alpha=0.3)

    # --- Density Field projections (imshow) ---
    density_field = cic_paint(jnp.zeros((length, length, length)), pos)
    if density_scaling != "none":
        from model import apply_density_scaling
        density_field = apply_density_scaling(density_field, density_scaling)

    ax_density_xy = axes[1, 0]
    im_xy = ax_density_xy.imshow(jnp.sum(density_field, axis=2), cmap='inferno', origin='lower',
                                 extent=[0, length, 0, length], aspect='auto')
    ax_density_xy.set_title(f'Density Field XY (t={current_time:.3f})')
    ax_density_xy.set_xlabel('X')
    ax_density_xy.set_ylabel('Y')
    ax_density_xy.set_xlim(0, length)
    ax_density_xy.set_ylim(0, length)
    cb_xy = fig.colorbar(im_xy, ax=ax_density_xy, orientation='vertical', fraction=0.046, pad=0.04)

    ax_density_xz = axes[1, 1]
    im_xz = ax_density_xz.imshow(jnp.sum(density_field, axis=1), cmap='inferno', origin='lower',
                                 extent=[0, length, 0, length], aspect='auto')
    ax_density_xz.set_title(f'Density Field XZ (t={current_time:.3f})')
    ax_density_xz.set_xlabel('X')
    ax_density_xz.set_ylabel('Z')
    ax_density_xz.set_xlim(0, length)
    ax_density_xz.set_ylim(0, length)
    cb_xz = fig.colorbar(im_xz, ax=ax_density_xz, orientation='vertical', fraction=0.046, pad=0.04)

    ax_density_yz = axes[1, 2]
    im_yz = ax_density_yz.imshow(jnp.sum(density_field, axis=0), cmap='inferno', origin='lower',
                                 extent=[0, length, 0, length], aspect='auto')
    ax_density_yz.set_title(f'Density Field YZ (t={current_time:.3f})')
    ax_density_yz.set_xlabel('Y')
    ax_density_yz.set_ylabel('Z')
    ax_density_yz.set_xlim(0, length)
    ax_density_yz.set_ylim(0, length)
    cb_yz = fig.colorbar(im_yz, ax=ax_density_yz, orientation='vertical', fraction=0.046, pad=0.04)

    # --- Energy plot (initialize three lines) ---
    ax_energy = axes[1, 3]
    if enable_energy_tracking:
        (line_ke,) = ax_energy.plot([], [], 'b-', label='Kinetic', linewidth=2)
        (line_pe,) = ax_energy.plot([], [], 'r-', label='Potential', linewidth=2)
        (line_te,) = ax_energy.plot([], [], 'k-', label='Total', linewidth=2)
        ax_energy.set_xlabel('Time')
        ax_energy.set_ylabel('Energy')
        ax_energy.set_title('Energy Evolution')
        ax_energy.legend()
        ax_energy.grid(True, alpha=0.3)
        # Set y-limits
        if len(all_ke) > 0:
            energy_min = min(jnp.min(all_ke), jnp.min(all_pe), jnp.min(all_te))
            energy_max = max(jnp.max(all_ke), jnp.max(all_pe), jnp.max(all_te))
            energy_range = energy_max - energy_min
            if energy_range > 0:
                ax_energy.set_ylim(energy_min - 0.1*energy_range, energy_max + 0.1*energy_range)
    else:
        ax_energy.axis('off')

    def animate(frame):
        pos = sol.ys[frame, 0]
        vel = sol.ys[frame, 1]
        current_time = sol.ts[frame]

        # Update scatter plots
        scatter3d._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        ax_3d.set_title(f'3D Positions (t={current_time:.3f})')

        scatter_xy.set_offsets(np.c_[pos[:, 0], pos[:, 1]])
        ax_xy.set_title(f'XY Projection (t={current_time:.3f})')

        scatter_xz.set_offsets(np.c_[pos[:, 0], pos[:, 2]])
        ax_xz.set_title(f'XZ Projection (t={current_time:.3f})')

        scatter_yz.set_offsets(np.c_[pos[:, 1], pos[:, 2]])
        ax_yz.set_title(f'YZ Projection (t={current_time:.3f})')

        # Update density fields
        density_field = cic_paint(jnp.zeros((length, length, length)), pos)
        if density_scaling != "none":
            from model import apply_density_scaling
            density_field = apply_density_scaling(density_field, density_scaling)
        im_xy.set_data(jnp.sum(density_field, axis=2))
        ax_density_xy.set_title(f'Density Field XY (t={current_time:.3f})')
        im_xz.set_data(jnp.sum(density_field, axis=1))
        ax_density_xz.set_title(f'Density Field XZ (t={current_time:.3f})')
        im_yz.set_data(jnp.sum(density_field, axis=0))
        ax_density_yz.set_title(f'Density Field YZ (t={current_time:.3f})')

        # Update energy plot
        if enable_energy_tracking:
            current_frame_times = all_times[:frame+1]
            current_frame_ke = all_ke[:frame+1]
            current_frame_pe = all_pe[:frame+1]
            current_frame_te = all_te[:frame+1]
            line_ke.set_data(current_frame_times, current_frame_ke)
            line_pe.set_data(current_frame_times, current_frame_pe)
            line_te.set_data(current_frame_times, current_frame_te)
            ax_energy.set_xlim(all_times[0], all_times[-1])
            # Mark current time (vertical line)
            for vline in ax_energy.lines[3:]:  # Remove previous vertical lines
                vline.remove()
            ax_energy.axvline(x=current_time, color='gray', linestyle='--', alpha=0.7)

        return (
            scatter3d, scatter_xy, scatter_xz, scatter_yz,
            im_xy, im_xz, im_yz,
        )
    print("Creating animation...")
    anim = animation.FuncAnimation(
        fig, animate, frames=len(sol.ts), interval=100, blit=False, repeat=False
    )

    if save_path:
        try:
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


### Plotting functions for sampling experiments

def plot_trace_subplots(mcmc_samples, theta, G, t_f, dt, softening, length, n_part, method, param_order, save_path=None):
    """
    Plot trace plots for MCMC samples with blob parameters.
    Vector parameters are plotted with separate subplots for each component.
    Scalar parameters each get their own subplot.

    Parameters:
    -----------
    mcmc_samples : dict
        Dictionary with parameter names as keys and sample arrays as values
    theta : dict
        Dictionary with true parameter values
    param_order : tuple
        Tuple of parameter names to plot
    """
    title = 'Sampling of the model parameters with ' + method
    param_info = f'G={G}, tf={t_f}, dt={dt}, L={length}, N={n_part}, softening={softening}'

    # Determine total number of subplots needed (scalar + vector components)
    subplot_info = []
    for param_name in param_order:
        if param_name in mcmc_samples:
            samples = mcmc_samples[param_name]
            if samples.ndim == 2:
                n_components = samples.shape[1]
                for j in range(n_components):
                    subplot_info.append((param_name, j))
            else:
                subplot_info.append((param_name, None))

    n_subplots = len(subplot_info)
    if n_subplots == 0:
        print("No parameters to plot")
        return None, None

    fig, axes = plt.subplots(n_subplots, 1, figsize=(12, 3 * n_subplots), squeeze=False)
    axes = axes.flatten()
    plt.suptitle(f"{title}\n{param_info}", fontsize=14)
    plt.subplots_adjust(top=0.92)

    for idx, (param_name, comp_idx) in enumerate(subplot_info):
        samples = mcmc_samples[param_name]
        ax = axes[idx]
        if comp_idx is not None:
            # Vector component
            ax.plot(samples[:, comp_idx], color=plt.cm.tab10(comp_idx), label=f'{param_name}[{comp_idx}]', alpha=0.8)
            # Add true value line if available
            if param_name in theta:
                true_val = theta[param_name]
                if isinstance(true_val, (list, tuple, np.ndarray)):
                    true_val = np.array(true_val)
                    if comp_idx < len(true_val):
                        ax.axhline(true_val[comp_idx], color='red', linestyle='--',
                                   label=f'True {param_name}[{comp_idx}] = {true_val[comp_idx]:.2f}')
            ax.set_ylabel(f'{param_name}[{comp_idx}]')
            ax.legend()
        else:
            # Scalar parameter
            ax.plot(samples, label=param_name)
            if param_name in theta:
                true_val = theta[param_name]
                if not isinstance(true_val, (list, tuple, np.ndarray)):
                    ax.axhline(true_val, color='red', linestyle='--', label=f'True {param_name} = {true_val:.2f}')
            ax.set_ylabel(param_name)
            ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel('Sample')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig, axes

def plot_corner_after_burnin(mcmc_samples, theta, G, t_f, dt, softening, length, n_part, method, burnin, param_order, save_path=None):
    """
    Plot corner plot for MCMC samples with blob parameters.
    """    
    # Prepare data for corner plot
    samples_list = []
    labels = []
    truths = []
    
    title = "Posterior distribution of initial distribution's parameters with " + method.upper()
    param_info = f'G={G}, tf={t_f}, dt={dt}, L={length}, N={n_part}, softening={softening}'
    title = f"{title}\n{param_info}"
    
    for param_name in param_order:
        if param_name in mcmc_samples:
            param_samples = mcmc_samples[param_name][burnin:]
            
            # Handle vector parameters
            if param_samples.ndim > 1 and param_samples.shape[1] > 1:
                # Treat each coordinate as a separate parameter
                for coord_idx in range(param_samples.shape[1]):
                    samples_list.append(param_samples[:, coord_idx])
                    labels.append(f"{param_name}[{coord_idx}]")
                    
                    if param_name in theta:
                        true_val = np.array(theta[param_name])
                        truths.append(true_val[coord_idx] if true_val is not None else None)
                    else:
                        truths.append(None)
            else:
                samples_list.append(param_samples.flatten())
                labels.append(param_name)
                
                if param_name in theta:
                    truths.append(theta[param_name])
                else:
                    truths.append(None)
    
    if not samples_list:
        print("No valid samples for corner plot")
        return None
    
    samples_array = np.column_stack(samples_list)
    
    fig = corner.corner(
        samples_array,
        labels=labels,
        truths=truths,
        show_titles=True,
        title_kwargs={"fontsize": 10},
        label_kwargs={"fontsize": 10},
        quantiles=[0.16, 0.5, 0.84],
        levels=(0.68, 0.95),
        plot_contours=True,
        fill_contours=True,
        bins=30
    )
    
    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(top=0.90)  # Adjust for the suptitle

    if save_path:
        fig.savefig(save_path)
    
    return fig