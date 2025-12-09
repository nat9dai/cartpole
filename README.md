# Lifted NMPC with the Inverted Pendulum on a Cart Example

This repository contains the simulation code accompanying the paper **"Enhanced Sampled-Data Model Predictive Control via Nonlinear Lifting."**

## Citation

If you use this code in your research, please cite:

```bibtex
@article{gerdpratoom2025enhanced,
  title={Enhanced Sampled-Data Model Predictive Control via Nonlinear Lifting},
  author={Gerdpratoom, Nuthasith and Matsuzaki, Fumiya and Yamamoto, Yutaka and Yamamoto, Kaoru},
  journal={International Journal of Robust and Nonlinear Control},
  year={2025},
  publisher={Wiley Online Library}
}
```

Available: [Wiley Online Library](https://onlinelibrary.wiley.com/doi/10.1002/rnc.70083)

## Requirements

Before running the simulation, make sure the following are installed:

- **Python 3.x** with the following packages:
  - `numpy`
  - `matplotlib`
  - `casadi`
  - `opengen`
- **Rust** (required by OpEn for solver compilation)
- **[OpEn (Optimization Engine)](https://alphaville.github.io/optimization-engine/)**

Please follow the installation instructions in the [official OpEn documentation](https://alphaville.github.io/optimization-engine/docs/installation).

## Usage

### Step 1: Generate Optimizers

Before running any simulations, you must first generate the optimizer solvers using OpEn. This compiles Rust-based optimization solvers for each controller type.

#### For Single-Rate Simulations:

```bash
cd OpEn/singlerate

# Generate standard NMPC optimizer
python gen.py

# Generate lifted NMPC optimizer
python gen_lifted.py
```

#### For Multi-Rate Simulations:

You need to generate optimizers for three different sampling periods (0.1s, 0.25s, 0.5s). Edit the `sampling_time` parameter in each file before running:

```bash
cd OpEn/multirate

# Generate for sampling_time = 0.1
# Edit gen.py, gen_lifting.py, and gen_lifting_multirate.py to set sampling_time = 0.1
python gen.py
python gen_lifting.py
python gen_lifting_multirate.py

# Generate for sampling_time = 0.25
# Edit the files to set sampling_time = 0.25
python gen.py
python gen_lifting.py
python gen_lifting_multirate.py

# Generate for sampling_time = 0.5
# Edit the files to set sampling_time = 0.5
python gen.py
python gen_lifting.py
python gen_lifting_multirate.py
```

**Note:** Each optimizer generation may take several minutes as OpEn compiles the Rust solver code.

### Step 2: Run Simulations

After generating all optimizers, you can run the main simulation scripts:

#### Single-Rate Simulation:

```bash
python main_singlerate.py
```

This runs two controllers (NMPC and Lifted NMPC) and displays:
- Cart position and velocity trajectories
- Control input sequences
- State norms
- Solver computation times
- Performance metrics (RMS values)

#### Multi-Rate Simulation:

```bash
python main_multirate.py
```

This runs nine simulations (3 sampling periods and 3 controller types) and compares:
- Standard NMPC
- Lifted NMPC
- Lifted NMPC with multi-rate control

across sampling periods of 0.1s, 0.25s, and 0.5s.

## Customization

### Modifying Simulation Parameters

Edit the parameters in `main_singlerate.py` or `main_multirate.py`:

```python
# Simulation time step
sampling_time_sim = 0.001

# Sampling period
sampler_period = 0.05  # for single-rate

# Initial condition
x_state_0 = [0, np.pi, 0, 0]  # [position, angle, velocity, angular_velocity]

# Cost weights
Q = [2.5, 10, 0.01, 0.01]   # State weights
Qt = [3.0, 30, 0.1, 0.02]   # Terminal weights
R = [0.1]                    # Control weight
```

### Modifying System Parameters

Edit the physical parameters when creating the model:

```python
model = CartPoleModel(
    gravity_acceleration=9.8,
    length=1.0,
    mass_cart=1.0,
    mass_pole=0.2
)
```