# Neural Wind Tunnel

**A Physics-Informed Neural Network (PINN) surrogate for real-time 3D incompressible fluid dynamics, with GPU-accelerated C++/OpenGL visualization.**

Neural Wind Tunnel replaces expensive CFD solvers with a trained neural network that satisfies the Navier-Stokes equations *by construction*. The ML pipeline (PyTorch) trains a PINN on a virtual wind tunnel domain with a spherical obstacle, exports the learned weights to ONNX, and a C++20/OpenGL 4.3 engine performs real-time inference and rendering via ONNX Runtime — enabling interactive exploration of flow fields at framerates no traditional solver can match.

---

## Table of Contents

- [Motivation](#motivation)
- [Architecture Overview](#architecture-overview)
- [The Physics: 3D Incompressible Navier-Stokes](#the-physics-3d-incompressible-navier-stokes)
- [Network Architecture](#network-architecture)
  - [Fourier Feature Embedding](#fourier-feature-embedding)
  - [Modified Residual MLP](#modified-residual-mlp)
- [Training Pipeline](#training-pipeline)
  - [Domain Sampling](#domain-sampling)
  - [Loss Formulation](#loss-formulation)
  - [Boundary Conditions](#boundary-conditions)
- [C++ Visualization Engine](#c-visualization-engine)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [ML Environment Setup](#ml-environment-setup)
  - [Engine Build (CMake + vcpkg)](#engine-build-cmake--vcpkg)
- [Roadmap](#roadmap)
- [License](#license)

---

## Motivation

Computational Fluid Dynamics (CFD) simulations — finite volume, finite element, lattice Boltzmann — produce highly accurate results but are computationally prohibitive for interactive use. A single steady-state RANS simulation of flow around a bluff body can take minutes to hours. Unsteady DNS or LES runs at moderate Reynolds numbers can take days on HPC clusters.

**Physics-Informed Neural Networks** offer a fundamentally different trade-off: invest heavy computation *once* during training, then enjoy millisecond-level inference forever after. The neural network doesn't just learn a data-driven mapping — it is *constrained* by the governing PDEs during training, meaning its predictions are physically plausible even in regions of the domain it has never explicitly seen.

Neural Wind Tunnel exploits this paradigm to build a real-time, interactive 3D fluid visualizer where users can adjust flow parameters (inflow velocity, viscosity) and immediately see physically-consistent velocity and pressure fields rendered in OpenGL.

---

## Architecture Overview

The system is split into two major subsystems connected by an ONNX model artifact:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        NEURAL WIND TUNNEL                              │
│                                                                        │
│  ┌──────────────────────────┐         ┌──────────────────────────────┐ │
│  │     ML Pipeline          │  ONNX   │     C++ Engine               │ │
│  │     (PyTorch)            │ ──────► │     (OpenGL 4.3)             │ │
│  │                          │  .onnx  │                              │ │
│  │  ┌────────────────────┐  │         │  ┌────────────────────────┐  │ │
│  │  │ WindTunnelDomain   │  │         │  │ ONNX Runtime           │  │ │
│  │  │ (Collocation Pts)  │  │         │  │ (CPU/GPU Inference)    │  │ │
│  │  └────────┬───────────┘  │         │  └────────┬───────────────┘  │ │
│  │           │              │         │           │                  │ │
│  │  ┌────────▼───────────┐  │         │  ┌────────▼───────────────┐  │ │
│  │  │ FluidPINN          │  │         │  │ OpenGL Renderer        │  │ │
│  │  │ (Fourier + ResNet) │  │         │  │ (Volume / Streamlines) │  │ │
│  │  └────────┬───────────┘  │         │  └────────┬───────────────┘  │ │
│  │           │              │         │           │                  │ │
│  │  ┌────────▼───────────┐  │         │  ┌────────▼───────────────┐  │ │
│  │  │ N-S PDE Loss       │  │         │  │ Dear ImGui (Docking)   │  │ │
│  │  │ + Boundary Loss    │  │         │  │ Controls + Viewport    │  │ │
│  │  └────────────────────┘  │         │  └────────────────────────┘  │ │
│  └──────────────────────────┘         └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

**Data flow:**
1. The Python pipeline defines a 3D wind tunnel domain with a spherical obstacle.
2. Collocation points are randomly sampled throughout the domain at each training step.
3. The PINN is trained by minimizing PDE residuals + boundary condition violations (no labeled simulation data required).
4. The trained model is exported to ONNX format.
5. The C++ engine loads the ONNX model, evaluates it on a dense grid, and renders the resulting flow field in real time.

---

## The Physics: 3D Incompressible Navier-Stokes

The governing equations for an incompressible Newtonian fluid in three spatial dimensions are:

### Continuity Equation (Mass Conservation)

$$\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0$$

This enforces **incompressibility** — the velocity field must be divergence-free everywhere. Physically, fluid neither compresses nor expands; what flows into any infinitesimal control volume must flow out.

### Momentum Equations

For each velocity component $q \in \{u, v, w\}$ with corresponding pressure gradient direction:

$$\frac{\partial q}{\partial t} + (\mathbf{u} \cdot \nabla) q = -\frac{1}{\rho}\frac{\partial p}{\partial x_q} + \nu \nabla^2 q$$

Written out explicitly for all three components (with $\rho = 1$ for non-dimensionalized form):

$$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} + w\frac{\partial u}{\partial z} = -\frac{\partial p}{\partial x} + \nu\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2}\right)$$

$$\frac{\partial v}{\partial t} + u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} + w\frac{\partial v}{\partial z} = -\frac{\partial p}{\partial y} + \nu\left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} + \frac{\partial^2 v}{\partial z^2}\right)$$

$$\frac{\partial w}{\partial t} + u\frac{\partial w}{\partial x} + v\frac{\partial w}{\partial y} + w\frac{\partial w}{\partial z} = -\frac{\partial p}{\partial z} + \nu\left(\frac{\partial^2 w}{\partial x^2} + \frac{\partial^2 w}{\partial y^2} + \frac{\partial^2 w}{\partial z^2}\right)$$

**Term-by-term breakdown:**

| Term | Physical Meaning |
|------|-----------------|
| $\partial q / \partial t$ | **Unsteady acceleration** — how the velocity field changes in time |
| $(\mathbf{u} \cdot \nabla) q$ | **Convective acceleration** — nonlinear advection of momentum by the flow itself |
| $-\nabla p$ | **Pressure gradient** — fluid accelerates from high to low pressure |
| $\nu \nabla^2 q$ | **Viscous diffusion** — momentum diffuses through the fluid due to internal friction |

The **Reynolds number** $Re = UL/\nu$ characterizes the ratio of inertial to viscous forces. At the default settings ($U = 1.0$ m/s, $L = 1.0$ m, $\nu = 0.01$), $Re = 100$, which sits in the laminar-to-transitional regime where vortex shedding begins to appear behind the obstacle.

---

## Network Architecture

### Fourier Feature Embedding

Standard MLPs suffer from **spectral bias** — they preferentially learn low-frequency functions and struggle to capture sharp gradients, thin boundary layers, and turbulent eddies. Neural Wind Tunnel addresses this with a random Fourier feature embedding:

$$\gamma(\mathbf{v}) = \Big[\sin(2\pi \mathbf{B} \mathbf{v}),\; \cos(2\pi \mathbf{B} \mathbf{v})\Big]^T$$

where $\mathbf{v} = (x, y, z, t) \in \mathbb{R}^4$ and $\mathbf{B} \in \mathbb{R}^{4 \times 128}$ is a fixed random Gaussian matrix sampled at initialization with scale $\sigma = 10.0$.

**Why this works:** The random projection maps low-dimensional inputs into a 256-dimensional space where each basis function oscillates at a different frequency. This gives the subsequent MLP direct access to a rich frequency spectrum from the first layer, eliminating the need to *learn* high-frequency representations through many layers of composition.

**The scale parameter $\sigma$** controls the bandwidth of the frequency distribution. Too low ($\sigma < 1$) and the network can't resolve fine structures; too high ($\sigma > 100$) and training becomes unstable. The value $\sigma = 10$ provides good coverage for the characteristic length scales present in the wind tunnel domain.

```
Input: (x, y, z, t) ∈ ℝ⁴
           │
           ▼
    ┌──────────────┐
    │  x_proj =    │
    │  2π · x @ B  │     B ∈ ℝ⁴ˣ¹²⁸ (frozen Gaussian)
    └──────┬───────┘
           │
     ┌─────┴─────┐
     ▼           ▼
  sin(x_proj) cos(x_proj)    each ∈ ℝ¹²⁸
     │           │
     └─────┬─────┘
           ▼
    concat → ℝ²⁵⁶           Fourier-embedded output
```

### Modified Residual MLP

After embedding, the signal passes through a deep MLP with **modified residual (skip) connections**:

$$h_{i+1} = \frac{h_i + \sigma(W_i h_i + b_i)}{\sqrt{2}}$$

where $\sigma$ is the **SiLU (Swish)** activation: $\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$.

**Why SiLU over ReLU?** The Navier-Stokes loss requires computing second-order spatial derivatives ($\nabla^2 u$, $\nabla^2 v$, $\nabla^2 w$) via automatic differentiation through the network. ReLU has a zero second derivative almost everywhere, which kills the viscous diffusion term during backpropagation. SiLU is $C^\infty$-smooth, ensuring all derivative orders are well-defined and non-trivial.

**Why the $1/\sqrt{2}$ normalization?** Without it, the variance of $h$ grows by a factor of 2 at each residual addition. The $1/\sqrt{2}$ factor preserves the signal magnitude across depth, preventing gradient explosion in a 6-layer network.

```
Fourier Output: ℝ²⁵⁶
        │
  ┌─────▼─────┐
  │  fc_in     │  Linear(256 → 256) + SiLU
  └─────┬──────┘
        │
        ▼
  ┌───────────────────┐
  │  Residual Block 1 │─── h_new = SiLU(W₁h + b₁)
  │  h = (h + h_new)  │    h = (h + h_new) / √2
  │      / √2         │
  └────────┬──────────┘
        │  (× 6 blocks, hidden_dim = 256)
        ▼
  ┌─────────────┐
  │   fc_out    │  Linear(256 → 4)
  └──────┬──────┘
         │
         ▼
   (u, v, w, p)       Velocity field + pressure
```

**Full network dimensions:**

| Layer | Input Dim | Output Dim | Parameters |
|-------|-----------|------------|------------|
| Fourier Embedding | 4 | 256 | 512 (frozen) |
| fc_in | 256 | 256 | 65,792 |
| hidden ×6 | 256 | 256 | 394,752 |
| fc_out | 256 | 4 | 1,028 |
| **Total** | | | **~462K trainable** |

---

## Training Pipeline

### Domain Sampling

The wind tunnel domain is defined as:

$$\Omega = [-2, 4] \times [-1, 1] \times [-1, 1] \times [0, 5]$$

representing $(x, y, z, t)$ — a rectangular channel 6 meters long, 2 meters wide, 2 meters tall, over a 5-second time window. A **spherical obstacle** of radius $r = 0.5$ is centered at the origin:

$$\mathcal{O} = \left\{(x,y,z) \;\Big|\; x^2 + y^2 + z^2 \leq 0.25 \right\}$$

At each training step, collocation points are sampled uniformly at random from $\Omega \setminus \mathcal{O}$ (the domain minus the obstacle interior). This **meshless** approach is a key advantage of PINNs — no structured grid or mesh generation required.

```
        y
        ▲
   1.0  │  ┌──────────────────────────────────|
        │  │           FLUID DOMAIN           │
        │  │                                  │
        │  │  INLET        ┌───┐              │  OUTLET
  U=1 ──│──│──────►   ○    │   │  ──────────► │──────►
        │  │         ( )   └───┘              │
        │  │          ▲                       │
        │  │     Obstacle                     │
  -1.0  │  └─────────r=0.5───────────────────-┘
        └──────────────────────────────────────► x
      -2.0              0.0                  4.0

   Domain: 6m × 2m × 2m (3D, shown as 2D cross-section)
   Obstacle: Sphere at origin, radius 0.5m
```

The interior sampler uses **rejection sampling**: it generates 1.2× the requested number of points, filters out those inside the obstacle sphere, and keeps exactly $N$ valid points. Boundary points are sampled on the inlet face and obstacle surface using spherical coordinates:

$$x_s = r\sin\phi\cos\theta, \quad y_s = r\sin\phi\sin\theta, \quad z_s = r\cos\phi$$

where $\theta \sim \mathcal{U}(0, 2\pi)$ and $\phi = \arccos(u)$ with $u \sim \mathcal{U}(-1, 1)$ for uniform distribution on the sphere.

### Loss Formulation

The total loss is composed entirely of **physics-based terms** — no labeled simulation data is used:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{PDE}} + \mathcal{L}_{\text{BC}}$$

#### PDE Loss (Interior Points)

The PDE residuals are computed by differentiating the network outputs with respect to the inputs via `torch.autograd.grad` (with `create_graph=True` to allow second-order differentiation). The loss is:

$$\mathcal{L}_{\text{PDE}} = \frac{1}{N}\sum_{i=1}^{N}\left[ |\nabla \cdot \mathbf{u}_i|^2 + |\mathcal{R}_{u,i}|^2 + |\mathcal{R}_{v,i}|^2 + |\mathcal{R}_{w,i}|^2 \right]$$

where the residuals $\mathcal{R}_q$ for $q \in \{u, v, w\}$ are defined as:

$$\mathcal{R}_q = \frac{\partial q}{\partial t} + \left(u\frac{\partial q}{\partial x} + v\frac{\partial q}{\partial y} + w\frac{\partial q}{\partial z}\right) + \frac{\partial p}{\partial x_q} - \nu\left(\frac{\partial^2 q}{\partial x^2} + \frac{\partial^2 q}{\partial y^2} + \frac{\partial^2 q}{\partial z^2}\right)$$

A perfectly trained PINN drives all residuals to zero, meaning the network outputs *exactly* satisfy the Navier-Stokes equations at every collocation point.

**Gradient computation chain:** Computing $\mathcal{L}_{\text{PDE}}$ requires:
1. A forward pass through the network: $(x,y,z,t) \to (u,v,w,p)$
2. First-order autodiff: 4 outputs × 4 inputs = 16 first derivatives
3. Second-order autodiff: 3 velocity components × 3 spatial dims = 9 second derivatives
4. Assembling the 4 residual equations from these 25 derivative quantities

This makes PINN training significantly more expensive per step than standard supervised learning, but eliminates the need for any training data.

#### Boundary Condition Loss

$$\mathcal{L}_{\text{BC}} = \mathcal{L}_{\text{inlet}} + \mathcal{L}_{\text{obstacle}}$$

**Inlet** ($x = -2$): Uniform inflow at 1 m/s in the $x$-direction:

$$\mathcal{L}_{\text{inlet}} = \frac{1}{M}\sum_{i=1}^{M}\left[(u_i - 1)^2 + v_i^2 + w_i^2\right]$$

**Obstacle surface** ($\|\mathbf{x}\| = r$): No-slip condition — fluid velocity is zero at a solid wall:

$$\mathcal{L}_{\text{obstacle}} = \frac{1}{M}\sum_{i=1}^{M}\left[u_i^2 + v_i^2 + w_i^2\right]$$

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | Adam | Standard for initial PINN convergence; L-BFGS refinement planned |
| Learning Rate | $10^{-3}$ | Aggressive initial rate; scheduler planned |
| Interior Batch | 4,000 | Dense sampling for 4D domain |
| Boundary Batch | 1,000 per surface | Sufficient for inlet + obstacle |
| Epochs | 1,000 | Baseline; production runs need 10-50K |
| $\nu$ | 0.01 | $Re = 100$, laminar-transitional regime |

---

## C++ Visualization Engine

The real-time engine is built on a modern C++20 / OpenGL 4.3 stack:

| Component | Library | Role |
|-----------|---------|------|
| Windowing & Input | SDL2 | Cross-platform window, GL context, events |
| OpenGL Loading | GLAD | Function pointer loading for GL 4.3 Core |
| Math | GLM | GLSL-compatible vector/matrix math |
| GUI | Dear ImGui (Docking) | Editor-style dockable panels |
| Logging | spdlog | Fast, header-only structured logging |
| ML Inference | ONNX Runtime | Cross-platform neural network inference |

The engine creates a dockable editor layout with:

- **PINN Controls panel** — sliders for inflow velocity and kinematic viscosity, live FPS counter
- **3D Viewport** — OpenGL render target for flow field visualization (volume rendering / streamlines planned)

The inference pipeline will evaluate the ONNX model on a dense 3D grid each frame, map the output velocity magnitude and pressure to color, and render the result as either a volumetric texture or particle-based streamlines.

```
┌──────────────────────────────────────────────────────────────┐
│  Neural Wind Tunnel - PINN Surrogate                    ─ □ ×│
├──────────────────────┬───────────────────────────────────────┤
│  PINN Controls       │                                       │
│  ──────────────────  │         3D Viewport                   │
│                      │                                       │
│  Fluid Dynamics      │    ┌─────────────────────────────┐    │
│  Parameters          │    │                             │    │
│                      │    │      ○ ──────►              │    │
│  Inflow Velocity     │    │     ( )  flow streamlines   │    │
│  [━━━━━━━●━━] 1.0    │    │      ○ ──────►              │    │
│                      │    │                             │    │
│  Kinematic Viscosity │    └─────────────────────────────┘    │
│  [━●━━━━━━━━] 0.01   │                                       │
│                      │                                       │
│  ──────────────────  │                                       │
│  16.7 ms (60.0 FPS)  │                                       │
├──────────────────────┴───────────────────────────────────────┤
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
neural-wind-tunnel/
│
├── CMakeLists.txt            # Build system — vcpkg deps, ImGui via FetchContent
├── CMakePresets.json          # MSVC and Ninja presets with vcpkg toolchain
├── vcpkg.json                 # Dependency manifest (SDL2, GLAD, GLM, spdlog, ONNX Runtime)
│
├── engine/
│   └── src/
│       └── main.cpp           # SDL2/OpenGL/ImGui application scaffold
│
├── ml/
│   ├── environment.yml        # Conda env: PyTorch + CUDA 11.8 + visualization tools
│   ├── dataset.py             # WindTunnelDomain: 3D+t domain, interior/boundary sampling
│   ├── model.py               # FluidPINN: Fourier embedding + residual MLP
│   ├── train.py               # Training loop: N-S PDE loss + boundary loss
│   └── validate.py            # Validation and visualization (PyVista)
│
└── assets/                    # Runtime assets (shaders, ONNX models — auto-copied to build)
```

---

## Getting Started

### ML Environment Setup

```bash
# Create the conda environment
conda env create -f ml/environment.yml
conda activate pinn-env

# Train the PINN (uses CUDA if available, falls back to CPU)
cd ml
python train.py

# Output: pinn_fluid.pth (PyTorch weights)
# TODO: ONNX export will produce assets/pinn_fluid.onnx
```

### Engine Build (CMake + vcpkg)

**Prerequisites:** Visual Studio 2022+, CMake 3.20+, vcpkg installed at `~/vcpkg`.

```bash
# Install dependencies via vcpkg (automatic with manifest mode)
# The vcpkg.json file handles: sdl2, glad, glm, spdlog, gtest, onnxruntime

# Configure (Visual Studio generator)
cmake --preset msvc

# Build
cmake --build build/msvc --config Release

# Run
cmake --build build/msvc --target run
```

For Neovim / terminal workflows:

```bash
cmake --preset nvim
cmake --build build/ninja
./build/ninja/WindTunnelApp
```

---

## Roadmap

- [x] PINN model architecture (Fourier embedding + residual MLP)
- [x] 3D Navier-Stokes PDE loss with full autodiff gradient chain
- [x] Domain sampling with obstacle rejection
- [x] Inlet + no-slip boundary conditions
- [x] C++ engine scaffold (SDL2 + OpenGL 4.3 + ImGui docking)
- [ ] ONNX export from PyTorch
- [ ] ONNX Runtime inference integration in C++ engine
- [ ] Volume rendering of velocity/pressure fields
- [ ] Streamline / particle advection visualization
- [ ] Compute shader acceleration for dense grid evaluation
- [ ] L-BFGS optimizer for fine-grained convergence
- [ ] Learning rate scheduling and loss weighting strategies
- [ ] Outlet and wall boundary conditions
- [ ] Parameterized inference (varying $Re$ at runtime)
- [ ] PyVista-based offline validation against analytical solutions

---

## License

*License TBD*