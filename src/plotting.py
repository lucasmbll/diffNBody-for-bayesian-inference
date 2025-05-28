# src/plotting.py

import matplotlib.pyplot as plt
import jax.numpy as jnp
from jaxpm.painting import cic_paint
import corner
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

### Plotting functions for DiffNBody simulations

def plot_density_fields_and_positions(G, tf, dt, length, n_part, input_field, init_pos, final_pos, output_field, save_path=None):
    """
    Plot density fields and particle positions in various projections.
    
    Parameters:
    -----------
    input_field : array
        The input density field (3D array)
    init_pos : array
        Initial particle positions (N x 3 array)
    final_pos : array
        Final particle positions (N x 3 array)
    output_field : array
        The output density field (3D array)
    """
    # Create figure with a grid layout
    fig = plt.figure(figsize=(25, 20))
    gs = plt.GridSpec(4, 4, figure=fig)

    title = 'Simulation with parameters:'
    param_info = f'G={G}, tf={tf}, dt={dt}, L={length}, N={n_part}'
    title += f'\n{param_info}'
    fig.suptitle(title, y=1.0, fontsize=22)
    
    # First row: Input density field (3 plots)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(jnp.sum(input_field, axis=0), cmap='inferno')
    ax1.set_title('Input Density Field (Projection X-Y)')
    fig.colorbar(im1, ax=ax1, orientation='vertical')

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(jnp.sum(input_field, axis=1), cmap='inferno')
    ax2.set_title('Input Density Field (Projection X-Z)')
    fig.colorbar(im2, ax=ax2, orientation='vertical')

    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(jnp.sum(input_field, axis=2), cmap='inferno')
    ax3.set_title('Input Density Field (Projection Y-Z)')
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
    ax12.set_title('Output Density Field (Projection X-Y)')
    fig.colorbar(im12, ax=ax12, orientation='vertical')

    ax13 = fig.add_subplot(gs[3, 1])
    im13 = ax13.imshow(jnp.sum(output_field, axis=1), cmap='inferno')
    ax13.set_title('Output Density Field (Projection X-Z)')
    fig.colorbar(im13, ax=ax13, orientation='vertical')

    ax14 = fig.add_subplot(gs[3, 2])
    im14 = ax14.imshow(jnp.sum(output_field, axis=2), cmap='inferno')
    ax14.set_title('Output Density Field (Projection Y-Z)')
    fig.colorbar(im14, ax=ax14, orientation='vertical')
    
    # Empty plot in bottom-right corner
    fig.add_subplot(gs[3, 3]).set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.show()
    return fig

def plot_all_timesteps(sol,
                       boxL,
                       G,    
                       tf,
                       dt,
                       n_part,
                       skip=1,            # save memory: plot every 'skip'-th step
                       s=1,               # marker size for scatters
                       cmap='inferno',
                       save_path=None):
    """
    Parameters
    ----------
    sol   : diffrax solution object  (ys shape = (T, 2, N, 3))
    boxL  : box length used in cic_paint
    skip  : int, plot every `skip`-th stored step
    s     : matplotlib scatter size
    """
    # How many rows will we draw?
    steps = sol.ts[::skip]       # time steps to plot
    nrows = len(steps)

    # Pre-build the canvas
    fig, axes = plt.subplots(nrows=nrows,
                             ncols=4,
                             figsize=(24, 4 * nrows),
                             squeeze=False)
    
    # Add global title with simulation parameters
    title = 'Simulation with parameters:'
    param_info = f'G={G}, tf={tf}, dt={dt}, L={boxL}, N={n_part}'
    title += f'\n{param_info}'
    fig.suptitle(title, y=1.0, fontsize=22)

    for row, t in enumerate(steps): 
        # Check if sol.ys is a tuple (central mass case) or array (first simulation)
        if isinstance(sol.ys, tuple):
            pos_t = sol.ys[0][row]  # For tuple format: (positions, velocities)
        else:
            pos_t = sol.ys[row][0]
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
        im = axes[row, 3].imshow(jnp.sum(field_t, axis=2),
                                 cmap=cmap, origin='lower')
        axes[row, 3].set_title('Σ ρ(x,y)')
        fig.colorbar(im, ax=axes[row, 3], orientation='vertical')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.show()
    return fig, axes

def plot_trajectories(solution, G, tf, dt, length, n_part, particle_indices=None, box=None, title=None, figsize=(20, 5), zoom=True, padding=0.1, smooth_window=5, arrow_frac=0.7, save_path=None):    
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

    if isinstance(solution.ys, tuple):
        positions = solution.ys[0]
    else:
        positions = solution.ys[:, 0]
    n_timesteps, n_particles, n_dims = positions.shape

    if particle_indices is None:
        if n_particles > 50:
            particle_indices = np.random.choice(n_particles, 50, replace=False)
        else:
            particle_indices = np.arange(n_particles)

    fig = plt.figure(figsize=figsize)
    ax_3d = fig.add_subplot(1, 4, 1, projection='3d')
    ax_xy = fig.add_subplot(1, 4, 2)
    ax_xz = fig.add_subplot(1, 4, 3)
    ax_yz = fig.add_subplot(1, 4, 4)

    # Create parameter info string
    param_info = f'G={G}, tf={tf}, dt={dt}, L={length}, N={n_part}'
    if title:
        title = f'{title}\n{param_info}'
    else:
        title = param_info

    ax_3d.set_title('3D Trajectories (Smoothed)')
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
                  color='lightgray', alpha=0.4, s=5)
    ax_xy.scatter(final_pos_all[:, 0], final_pos_all[:, 1], color='lightgray', alpha=0.2, s=5)
    ax_xz.scatter(final_pos_all[:, 0], final_pos_all[:, 2], color='lightgray', alpha=0.2, s=5)
    ax_yz.scatter(final_pos_all[:, 1], final_pos_all[:, 2], color='lightgray', alpha=0.2, s=5)

    colors = plt.cm.jet(np.linspace(0, 1, len(particle_indices)))
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    for i, p_idx in enumerate(particle_indices):
        traj = positions[:, p_idx]
        traj_smooth = smooth_trajectory(traj, window=smooth_window)
        x_min = min(x_min, traj_smooth[:, 0].min())
        x_max = max(x_max, traj_smooth[:, 0].max())
        y_min = min(y_min, traj_smooth[:, 1].min())
        y_max = max(y_max, traj_smooth[:, 1].max())
        z_min = min(z_min, traj_smooth[:, 2].min())
        z_max = max(z_max, traj_smooth[:, 2].max())
        ax_3d.plot(traj_smooth[:, 0], traj_smooth[:, 1], traj_smooth[:, 2], color=colors[i], linewidth=1.0, alpha=0.7)
        ax_xy.plot(traj_smooth[:, 0], traj_smooth[:, 1], color=colors[i], linewidth=1.0)
        ax_xz.plot(traj_smooth[:, 0], traj_smooth[:, 2], color=colors[i], linewidth=1.0)
        ax_yz.plot(traj_smooth[:, 1], traj_smooth[:, 2], color=colors[i], linewidth=1.0)
        ax_3d.scatter(traj_smooth[0, 0], traj_smooth[0, 1], traj_smooth[0, 2], color=colors[i], marker='o', s=20)
        ax_3d.scatter(traj_smooth[-1, 0], traj_smooth[-1, 1], traj_smooth[-1, 2], color=colors[i], marker='s', s=20)

        # Add arrows to indicate direction (at arrow_frac along the trajectory)
        idx = int(len(traj_smooth) * arrow_frac)
        if idx < len(traj_smooth) - 1:
            # 3D arrow
            ax_3d.quiver(
                traj_smooth[idx, 0], traj_smooth[idx, 1], traj_smooth[idx, 2],
                traj_smooth[idx+1, 0] - traj_smooth[idx, 0],
                traj_smooth[idx+1, 1] - traj_smooth[idx, 1],
                traj_smooth[idx+1, 2] - traj_smooth[idx, 2],
                color=colors[i], arrow_length_ratio=0.3, linewidth=1.5, alpha=0.8
            )
            # 2D arrows
            ax_xy.annotate('', xy=(traj_smooth[idx+1, 0], traj_smooth[idx+1, 1]), 
                           xytext=(traj_smooth[idx, 0], traj_smooth[idx, 1]),
                           arrowprops=dict(facecolor=colors[i], edgecolor=colors[i], arrowstyle='->', lw=1.5, alpha=0.8))
            ax_xz.annotate('', xy=(traj_smooth[idx+1, 0], traj_smooth[idx+1, 2]), 
                           xytext=(traj_smooth[idx, 0], traj_smooth[idx, 2]),
                           arrowprops=dict(facecolor=colors[i], edgecolor=colors[i], arrowstyle='->', lw=1.5, alpha=0.8))
            ax_yz.annotate('', xy=(traj_smooth[idx+1, 1], traj_smooth[idx+1, 2]), 
                           xytext=(traj_smooth[idx, 1], traj_smooth[idx, 2]),
                           arrowprops=dict(facecolor=colors[i], edgecolor=colors[i], arrowstyle='->', lw=1.5, alpha=0.8))

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
    elif box is not None:
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
    if title:
        fig.suptitle(title, fontsize=16)
    start_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8, label='Start')
    end_marker = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='k', markersize=8, label='End')
    background_marker = plt.Line2D([0], [0], marker='o', color='lightgray', markersize=8, label='All Particles')
    ax_3d.legend(handles=[start_marker, end_marker, background_marker], loc='upper right')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_velocity_distributions(sol, G, tf, dt, length, n_part, save_path=None):
    # Calculate velocity norms for initial and final velocities
    init_vel = sol.ys[0, 1]  # Initial velocities
    init_vel_norm = jnp.sqrt(jnp.sum(init_vel**2, axis=1))
    final_vel = sol.ys[-1, 1]  # Final velocities
    final_vel_norm = jnp.sqrt(jnp.sum(final_vel**2, axis=1))

    # Create a figure with 2 rows and 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    title = 'Velocity Distribution Comparison with parameters:'
    param_info = f'G={G}, tf={tf}, dt={dt}, L={length}, N={n_part}'
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


### Plotting functions for sampling experiments
def plot_trace_subplots(mcmc_samples, theta, G, t_f, dt, softening, length, n_part, figsize=(18, 5), method="HMC", param_order=("sigma", "mean", "vel_sigma"), save_path=None):
    """
    Plot trace plots for (sigma, mean, vel_sigma) with true values as horizontal lines.
    """
    title = 'Sampling of the model parameters distribution with ' + method
    param_info = f'G={G}, tf={t_f}, dt={dt}, L={length}, N={n_part}, softening={softening}'
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True)
    if title or param_info:
        plt.suptitle(f"{title}\n{param_info}", fontsize=14)
    colors = ["blue", "green", "orange"]
    true_labels = ["pos_std", "pos_mean", "vel_std"]
    for i, param in enumerate(param_order):
        axes[i].plot(mcmc_samples[param], label=param, color=colors[i])
        axes[i].axhline(
            y=theta[true_labels[i]],
            color='r', linestyle='--', alpha=0.5,
            label=f'True {param}={theta[true_labels[i]]}'
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

def plot_corner_after_burnin(mcmc_samples, theta, burnin=1000, param_order=("sigma", "mean", "vel_sigma"), title="Posterior distribution of model's parameters", save_path=None):
    """
    Plot a corner plot of posterior samples after burn-in.
    """
    samples = np.column_stack([
        np.array(mcmc_samples[param])[burnin:] for param in param_order
    ])
    truths = [theta["pos_std"], theta["pos_mean"], theta["vel_std"]]
    fig = corner.corner(
        samples,
        labels=list(param_order),
        truths=truths,
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
