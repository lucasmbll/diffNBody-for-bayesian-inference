# N-Body Inverse Sampling

A differentiable pipeline for N-body gravitational simulations with JAX and BlackJAX, supporting both forward simulation and Bayesian inference of initial conditions via MCMC.

## Features

- **Forward modeling** of particle dynamics using [JAX](https://github.com/jax-ml/jax) and [Diffrax](https://github.com/patrick-kidger/diffrax)
- **Multiple model types**: Gaussian blobs and dual Gaussian blobs initial conditions
- **Density field scaling**: Log, sqrt, normalization, standardization, and power scaling options
- **Velocity configurations**: Random + circular or pure circular velocity initialization
- **Mock data generation** with customizable initial conditions and comprehensive visualization
- **Inverse inference**: Recover initial conditions from simulated observations using gradient-based MCMC (NUTS/HMC) with [BlackJAX](https://github.com/blackjax-devs/blackjax)
- **Flexible likelihood functions**: Standard Gaussian (ll1) and Monte Carlo averaged (ll2)
- **Multiple prior distributions**: Gaussian and uniform priors with extensible framework
- **Comprehensive plotting**: Density fields, particle trajectories, energy evolution, velocity distributions, MCMC traces, and corner plots
- **Video generation**: Create animated visualizations of particle evolution
- **GPU acceleration** out of the box

## Project Structure

```

nbody-inverse-sampling/
│
├── configs/            # YAML config files
│   └── defaults.yaml
│
├── src/
│   ├── model.py        # Forward models (e.g., gaussian\_model)
│   ├── likelihood.py   # Likelihood and prior definitions
│   ├── sampling.py     # MCMC sampling utilities (BlackJAX)
│   ├── plotting.py     # Plotting utilities (trace plots, etc.)
│   └── run\_experiments.py  # Main experiment script (CLI)
│
└── README.md

````

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/nbody-inverse-sampling.git
   cd nbody-inverse-sampling
````

2. **(Recommended) Create a Conda environment**

   ```bash
   conda create -n nbody python=3.10
   conda activate nbody
   ```

3. **Install requirements**

   ```bash
   pip install jax jaxlib diffrax blackjax matplotlib numpy
   # For GPU: see [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html)
   ```

4. **(Optional) Install extra tools**

   ```bash
   pip install tqdm
   ```

## Usage

### 1. Run an experiment

Edit your configuration in `configs/defaults.yaml` (see below for an example).

Then run:

```bash
python src/run_experiments.py --config configs/defaults.yaml
```

* Generates mock data using your forward model.
* Runs MCMC sampling to infer parameters from the data.
* Saves outputs (samples, traces, plots) to the results directory.

### 2. Running Multiple Experiments

Then, run:

```bash
python src/run_multiple_experiments.py ./configs/exp1.yaml ./configs/exp2.yaml
```

You can specify up to 4 config files. Each experiment will use its own config and run in parallel.

### 3. Configuration

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
## Extending the Framework

### Adding New Priors

To add a new prior distribution, follow these steps:

#### 1. Define the Prior Function

Add your prior function to `src/likelihood.py`:

```python
def exponential_prior(params_dict, prior_params):
    """
    Example: Exponential prior for all parameters
    """
    sigma = params_dict["sigma"]
    mean = params_dict["mean"]
    vel_sigma = params_dict["vel_sigma"]
    
    # Each param should have a 'rate' parameter in prior_params
    p_sigma = stats.expon.logpdf(sigma, scale=1/prior_params["sigma"]["rate"])
    p_mean = stats.expon.logpdf(mean, scale=1/prior_params["mean"]["rate"])
    p_vel = stats.expon.logpdf(vel_sigma, scale=1/prior_params["vel_sigma"]["rate"])
    
    return p_sigma + p_mean + p_vel

def beta_prior(params_dict, prior_params):
    """
    Example: Beta prior (parameters must be scaled to [0,1])
    """
    # Assume parameters are already scaled to [0,1] range
    sigma = params_dict["sigma"]
    mean = params_dict["mean"] 
    vel_sigma = params_dict["vel_sigma"]
    
    p_sigma = stats.beta.logpdf(sigma, prior_params["sigma"]["alpha"], prior_params["sigma"]["beta"])
    p_mean = stats.beta.logpdf(mean, prior_params["mean"]["alpha"], prior_params["mean"]["beta"])
    p_vel = stats.beta.logpdf(vel_sigma, prior_params["vel_sigma"]["alpha"], prior_params["vel_sigma"]["beta"])
    
    return p_sigma + p_mean + p_vel
```

#### 2. Register the Prior

Add your new prior to the `PRIOR_REGISTRY` in `src/likelihood.py`:

```python
PRIOR_REGISTRY = {
    "gaussian": gaussian_prior,
    "uniform": uniform_prior,
    "exponential": exponential_prior,  # Add your new prior here
    "beta": beta_prior,                # Add another example
    # Add more priors here as needed
}
```

#### 3. Update Configuration

Use your new prior in your YAML config file:

```yaml
prior_type: exponential   # Use your new prior

prior_params:
  # For exponential prior:
  sigma: {rate: 0.1}        # rate parameter for exponential distribution
  mean: {rate: 0.033}       # 1/30 ≈ 0.033 for mean around 30
  vel_sigma: {rate: 1.0}
  
  # Or for beta prior:
  # sigma: {alpha: 2.0, beta: 5.0}
  # mean: {alpha: 3.0, beta: 2.0}
  # vel_sigma: {alpha: 2.0, beta: 2.0}
```


## Code Structure Overview

* `model.py` – Defines forward simulation (`gaussian_model`), N-body dynamics, and utility functions.
* `likelihood.py` – Implements likelihoods (Gaussian, Monte Carlo, etc.) and priors.
* `sampling.py` – Contains MCMC wrappers for BlackJAX (NUTS/HMC), warmup, sample extraction.
* `run_experiments.py` – Command-line experiment runner, handles data generation, sampling, and saving results.
* `plotting.py` – Plot data density field, particles positions and trajectories, MCMC chains, posteriors. 

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

