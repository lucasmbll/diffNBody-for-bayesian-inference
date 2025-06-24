# N-Body Inverse Sampling

A JAX-based framework for N-body simulations and Bayesian parameter inference using gradient-based MCMC sampling. This package allows you to run gravitational N-body simulations with flexible blob-based initialization and perform parameter inference to recover initial conditions from final density fields.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Modes](#usage-modes)
- [Configuration Files](#configuration-files)
  - [Simulation Mode](#simulation-mode)
  - [Sampling Mode](#sampling-mode)
- [Parameter Reference](#parameter-reference)
- [Examples](#examples)
- [Output Structure](#output-structure)
- [Troubleshooting](#troubleshooting)

## Features

- **Flexible N-body simulations** with blob-based initialization
- **Multiple initialization types**: Gaussian and NFW profiles
- **Various velocity distributions**: Cold, virial, and circular
- **Bayesian parameter inference** using HMC, NUTS, MALA or Random Walk samplers
- **Automatic differentiation** for efficient gradient computation
- **GPU acceleration** with JAX
- **Comprehensive visualization** with automatic plot generation
- **Video generation** of simulation evolution
- **Energy tracking** and conservation monitoring

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended)
- FFmpeg (for video generation, optional)

Librairies :
- jax (>=0.5.3)
- jaxlib (>=0.5.3)
- diffrax (>=0.7.0)
- blackjax (>=1.2.5)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/nbody-inverse-sampling.git
cd nbody-inverse-sampling
```

### Step 2: Create a Conda Environment (Recommended)

```bash
conda create -n diffnbody python=3.10
conda activate diffnbody
```

### Step 3: Install Dependencies

```bash
pip install jax jaxlib diffrax blackjax matplotlib numpy
# For GPU: see [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html)
```

### Step 4: Install Optional Tools

```bash
pip install tqdm
```

## Quick Start

This section provides step-by-step instructions to get you started with N-body simulations and parameter inference. The framework supports two main modes: **simulation** and **sampling**. Let's start with a basic simulation.

#### 1. Simulation Mode

Simulation mode runs an N-body simulation with predefined initial conditions and generates comprehensive visualizations.

**Basic Example:**

```bash
python src/run_experiments.py --config configs/default_sim.yaml
```

This will:
- Initialize 3000 particles in 3 different blobs (Gaussian + NFW profiles)
- Run the simulation for 0.75 time units, with a time resolution of 0.05
- Generate density field plots, trajectories, and velocity distributions
- Save all results to `results/nbody_sim_results/`

**Understanding the Configuration:**

The `default_sim.yaml` demonstrates a multi-blob setup:

```yaml
mode: sim  # Simulation mode

model_params:
  G: 5.0           # Strong gravity for dramatic evolution
  length: 64       # 64³ simulation box
  t_f: 0.75        # Short simulation time
  dt: 0.05         # Small timestep for accuracy
  
  # Three different blob types:
  blobs_params:
    # Blob 1: Gaussian distribution with circular motion
    - n_part: 1000
      pos_type: gaussian
      pos_params:
        sigma: 8.0
        center: [12.0, 52.0, 12.0]
      vel_type: circular
      vel_params:
        vel_factor: 1.0
        
    # Blob 2: NFW halo with virial equilibrium  
    - n_part: 1000
      pos_type: nfw
      pos_params:
        rs: 5.0
        c: 2.0
        center: [12.0, 12.0, 52.0]
      vel_type: virial
      vel_params:
        virial_ratio: 0.5
        
    # Blob 3: Cold Gaussian blob
    - n_part: 1000
      pos_type: gaussian
      pos_params:
        sigma: 6.0
        center: [52.0, 12.0, 12.0]
      vel_type: cold
      vel_params:
        vel_dispersion: 0.1
```

**Customizing Your Simulation:**

1. **Change the physics:**
   ```yaml
   G: 1.0          # Weaker gravity
   softening: 0.5  # More softening
   t_f: 2.0        # Longer simulation
   ```

2. **Modify blob properties:**
   ```yaml
   blobs_params:
     - n_part: 2000           # More particles
       pos_type: gaussian
       pos_params:
         sigma: 15.0          # Larger blob
         center: [32.0, 32.0, 32.0]  # Centered
       vel_type: circular
       vel_params:
         vel_factor: 0.5      # Slower rotation
     - n_part: 1000           # Add as blobs as needed
       ...
   ```

3. **Enable advanced features:**
   ```yaml
   plot_settings:
     enable_energy_tracking: true  # Monitor energy conservation
     generate_video: 
       do: true               # Create MP4 animation
       video_fps: 30
       video_dpi: 200
   ```

**Expected Output:**

After running the simulation, you'll find in `results/nbody_sim_results/`:

```
default_sim_YYYYMMDD_HHMMSS/
├── config.yaml                          # Copy of your configuration
├── density_fields_and_positions.png     # Multi-panel density plots
├── timesteps.png                        # Time evolution snapshots
├── trajectories.png                     # Particle trajectory plots
├── velocity_distributions.png           # Velocity analysis
└── simulation_video.mp4                 # Animation (if enabled)
```

**Performance Tips:**

- **GPU Usage:** Set `cuda_visible_devices` to your GPU index
- **Video Generation:** Requires FFmpeg installation, may be long to generate
- **Memory:** Reduce `n_part` if you encounter memory issues

**Common Blob Configurations:**

```yaml
# Colliding galaxies
blobs_params:
  - n_part: 1500
    pos_type: nfw
    pos_params: {rs: 8.0, c: 3.0, center: [20.0, 32.0, 32.0]}
    vel_type: circular
    vel_params: {vel_factor: 0.8}
  - n_part: 1500  
    pos_type: nfw
    pos_params: {rs: 6.0, c: 4.0, center: [44.0, 32.0, 32.0]}
    vel_type: circular
    vel_params: {vel_factor: 0.8}

# Cluster formation
blobs_params:
  - n_part: 800
    pos_type: gaussian
    pos_params: {sigma: 5.0, center: [32.0, 32.0, 32.0]}
    vel_type: virial
    vel_params: {virial_ratio: 0.3}
  # Add 4-6 smaller blobs around the main one...
```

#### 2. Sampling Mode

Sampling mode allows you to infer the initial conditions of an N-body system from observed final density fields using Bayesian parameter inference. This mode leverages MCMC samplers such as NUTS, HMC, MALA, and Random Walk Metropolis.

**Basic Example:**

```bash
python src/run_experiments.py --config configs/default_sampling_nuts.yaml
```
This will:

- Use the NUTS sampler to infer parameters of a single Gaussian blob (e.g., sigma and center).
- Run 2000 sampling steps with 500 warmup steps for adaptation.
- Save results to sampling_results.
- Understanding the Configuration:

The `default_sampling.yaml` demonstrates a simple setup for parameter inference:

```yaml
mode: sampling  # Sampling mode

model_params:
  G: 5.0           # Gravitational constant
  length: 64       # 64³ simulation box
  t_f: 1.0         # Short simulation time
  dt: 0.05         # Small timestep for accuracy
  
  blobs_params:
    - n_part: 2000
      pos_type: gaussian
      pos_params:
        sigma: 10.0  
        center: [32.0, 32.0, 32.0]  
      vel_type: circular
      vel_params:
        vel_factor: 0.5  

prior_params:
  blob0_sigma: {mu: 10.0, sigma: 5.0}
  blob0_center: {mu: [32.0, 32.0, 32.0], sigma: 10.0}

sampler: nuts
num_samples: 2000
num_warmup: 500
```
**Customizing Your Sampling:**

1. **Change the sampler:**
```yaml
sampler: hmc  # Use Hamiltonian Monte Carlo
step_size: 0.01
num_integration_steps: 50
```
2. **Add more blobs:**
```yaml
blobs_params:
  - n_part: 2000
    pos_type: gaussian
    pos_params:
      sigma: 15.0
      center: [32.0, 32.0, 32.0]
    vel_type: circular
    vel_params:
      vel_factor: 0.5
  - n_part: 1000
    pos_type: nfw
    pos_params:
      rs: 5.0
      c: 2.0
      center: [12.0, 12.0, 52.0]
    vel_type: virial
    vel_params:
      virial_ratio: 0.5
```
3. **Modify priors:**
```yaml
prior_params:
  blob0_sigma: {mu: 12.0, sigma: 3.0}
  blob0_center: {mu: [30.0, 30.0, 30.0], sigma: 5.0}
  #Only parameters with provided prior will be sampled
```
**Expected Outputs:**
```
default_sampling_nuts_YYYYMMDD_HHMMSS/
├── config.yaml                          # Copy of your configuration
├── trace_sampling.png                   # Trace plots for sampled parameters
├── corner_sampling.png                  # Corner plots for posterior distributions
├── samples.npz                          # Saved samples in NumPy format
```

**Performance Tips:**
- Warmup Steps: Increase num_warmup for better adaptation in high-dimensional problems.
- Sampling Steps: Use more num_samples for higher accuracy in posterior estimation.
- GPU Usage: Ensure cuda_visible_devices is set correctly for faster sampling.


## References

* [JAX documentation](https://jax.readthedocs.io/)
* [Diffrax documentation](https://docs.kidger.site/diffrax/)
* [BlackJAX documentation](https://blackjax-devs.github.io/blackjax/)

## License

MIT License. See `LICENSE`.

---

**Contact:**
Maintainer: [Lucas Mebille](mailto:lucas.mebille.pro@gmail.com)

