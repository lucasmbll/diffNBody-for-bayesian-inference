# N-Body Inverse Sampling

A JAX-based framework for N-body simulations and Bayesian parameter inference using MCMC sampling. This package allows you to run gravitational N-body simulations with flexible blob-based initialization and perform parameter inference to recover initial conditions from final density fields.

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
- **Bayesian parameter inference** using HMC and NUTS samplers
- **Automatic differentiation** for efficient gradient computation
- **GPU acceleration** with JAX
- **Comprehensive visualization** with automatic plot generation
- **Video generation** of simulation evolution
- **Energy tracking** and conservation monitoring

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- FFmpeg (for video generation, optional)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/nbody-inverse-sampling.git
cd nbody-inverse-sampling
```

### Step 2: Create a Conda Environment (Recommended)

```bash
conda create -n nbody python=3.10
conda activate nbody
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

### Run an Experiment

Edit your configuration in `configs/defaults.yaml` (see below for an example).

Then run:

```bash
python src/run_experiments.py --config configs/defaults.yaml
```

This will:

- Generate mock data using your forward model.
- Run MCMC sampling to infer parameters from the data.
- Save outputs (samples, traces, plots) to the results directory.

### Run multiple sampling sequantially

1. Run specific config

```
cd /home/lucasm/Files/DiffNBody-main/nbody-inverse-sampling/src
python run_batch_experiments.py --configs ../configs/sampling/config1.yaml ../configs/sampling/config2.yaml ../configs/sampling/config3.yaml
```

2. Run all config files in a directory:
```
python run_batch_experiments.py --config-dir ../configs/sampling/
```

3. Run configs listed in a text file:
```
# Create a file with config paths
echo "../configs/sampling/test_sampling.yaml" > config_list.txt
echo "../configs/sampling/another_config.yaml" >> config_list.txt

python run_batch_experiments.py --config-list config_list.txt
```

## Configuration Files

### Simulation Mode

Example `configs/defaults.yaml`:

```yaml
output_path: ./results/samples.npz

likelihood_type: ll2          # "ll1", "ll2", ou "ll3"
likelihood_kwargs: 
  noise: 0.05

prior_type: gaussian   # "gaussian" or "uniform"

prior_params:
  # For gaussian prior, use:
  sigma: {mu: 10.0, sigma: 0.5}        
  mean: {mu: 30.0, sigma: 0.5}
  vel_sigma: {mu: 1.0, sigma: 0.1}
  # For uniform prior, use:
  # sigma: {low: 8.0, high: 12.0}
  # mean: {low: 28.0, high: 32.0}
  # vel_sigma: {low: 0.8, high: 1.2}

sampler: hmc         # "nuts" ou "hmc"
num_samples: 5000
num_warmup: 1000
inv_mass_matrix: [0.1, 0.5, 0.1]
step_size: 0.001
num_integration_steps: 50

cuda_visible_devices: 1 # Set to the GPU number you want to use in your cluster

progress_bar: true

initial_position:
  sigma: 10.0
  mean: 30.0
  vel_sigma: 1.0
random_seed: 123456

data_params:
  sigma: 10.0
  mean: 30.0
  vel_sigma: 1.0
  n_part: 1000
  G: 5.0
  length: 64
  softening: 0.1
  t_f: 1.0
  dt: 0.5
  data_seed: 0
  # Add other model parameters as needed
```

### Sampling Mode

Example `configs/sampling.yaml`:

```yaml
output_path: ./results/sampling_results.npz

likelihood_type: ll2          # "ll1", "ll2", ou "ll3"
likelihood_kwargs: 
  noise: 0.05

prior_type: gaussian   # "gaussian" or "uniform"

prior_params:
  sigma: {mu: 10.0, sigma: 0.5}        
  mean: {mu: 30.0, sigma: 0.5}
  vel_sigma: {mu: 1.0, sigma: 0.1}

sampler: nuts         # "nuts" ou "hmc"
num_samples: 10000
num_warmup: 2000
inv_mass_matrix: [0.1, 0.5, 0.1]
step_size: 0.001
num_integration_steps: 50

cuda_visible_devices: 0 # Set to the GPU number you want to use in your cluster

progress_bar: true

initial_position:
  sigma: 10.0
  mean: 30.0
  vel_sigma: 1.0
random_seed: 654321

data_params:
  sigma: 10.0
  mean: 30.0
  vel_sigma: 1.0
  n_part: 1000
  G: 5.0
  length: 64
  softening: 0.1
  t_f: 1.0
  dt: 0.5
  data_seed: 1
```

## Parameter Reference

* `model.py` – Defines forward simulation (`gaussian_model`), N-body dynamics, and utility functions.
* `likelihood.py` – Implements likelihoods (Gaussian, Monte Carlo, etc.) and priors.
* `sampling.py` – Contains MCMC wrappers for BlackJAX (NUTS/HMC), warmup, sample extraction.
* `run_experiments.py` – Command-line experiment runner, handles data generation, sampling, and saving results.
* `plotting.py` – Plot data density field, particles positions and trajectories, MCMC chains, posteriors. 

## Examples

## Output Structure

## Troubleshooting

## References

* [JAX documentation](https://jax.readthedocs.io/)
* [Diffrax documentation](https://docs.kidger.site/diffrax/)
* [BlackJAX documentation](https://blackjax-devs.github.io/blackjax/)

## License

MIT License. See `LICENSE`.

---

**Contact:**
Maintainer: [Lucas Mebille](mailto:lucas.mebille.pro@gmail.com)

```

