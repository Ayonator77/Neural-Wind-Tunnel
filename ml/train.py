import torch
import torch.optim as optim
from model import FluidPINN
from dataset import WindTunnelDomain

#Hyperparamters
NU = 0.01 # Kinematic viscosity
EPOCHS = 1000
BATCH_SIZE_INTERIOR = 4000
BATCH_SIZE_BOUNDARY = 1000

def compute_gradients(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Helper function to compute gradients of y with respect to x using autograd.
    """
    grad = torch.autograd.grad(
        outputs=y, inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]
    return grad

def navier_stokes_loss(model: FluidPINN, points: torch.Tensor) -> torch.Tensor:
    """
    Computes the physics loss using the 3D Incompressible Navier-Stokes equations.
    points: tensor of shape $(N, 4)$ representing $(x, y, z, t)$

    The system being solved is:

    Continuity (incompressibility):

    $$\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0$$

    Momentum (x, y, z):
    
    $$\frac{\partial u}{\partial t} + \left(u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} + w\frac{\partial u}{\partial z}\right) + \frac{\partial p}{\partial x} - \nu\nabla^2 u = 0$$

    $$\frac{\partial v}{\partial t} + \left(u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} + w\frac{\partial v}{\partial z}\right) + \frac{\partial p}{\partial y} - \nu\nabla^2 v = 0$$

    $$\frac{\partial w}{\partial t} + \left(u\frac{\partial w}{\partial x} + v\frac{\partial w}{\partial y} + w\frac{\partial w}{\partial z}\right) + \frac{\partial p}{\partial z} - \nu\nabla^2 w = 0$$

    where $\nu$ is kinematic viscosity and $\nabla^2 q = \frac{\partial^2 q}{\partial x^2} + \frac{\partial^2 q}{\partial y^2} + \frac{\partial^2 q}{\partial z^2}$.

    PDE Loss:
    $$\mathcal{L}_{PDE} = \|\nabla\cdot\mathbf{u}\|^2 + \|\mathcal{R}_u\|^2 + \|\mathcal{R}_v\|^2 + \|\mathcal{R}_w\|^2$$
    """

    # Forward pass to get u, v, w, p
    predictions = model(points)
    u, v, w, p = predictions[:, 0:1], predictions[:, 1:2], predictions[:, 2:3], predictions[:, 3:4]

    #Extract coordinates for differentiation
    x, y, z, t = points[:, 0:1], points[:, 1:2], points[:, 2:3], points[:, 3:4]

    # First-Order derivatives (velocities and pressure)
    u_g = compute_gradients(u, points)
    u_x, u_y, u_z, u_t = u_g[:, 0:1], u_g[:, 1:2], u_g[:, 2:3], u_g[:, 3:4]

    v_g = compute_gradients(v, points)
    v_x, v_y, v_z, v_t = v_g[:, 0:1], v_g[:, 1:2], v_g[:, 2:3], v_g[:, 3:4]

    w_g = compute_gradients(w, points)
    w_x, w_y, w_z, w_t = w_g[:, 0:1], w_g[:, 1:2], w_g[:, 2:3], w_g[:, 3:4]
    
    p_g = compute_gradients(p, points)
    p_x, p_y, p_z = p_g[:, 0:1], p_g[:, 1:2], p_g[:, 2:3]

    # Second-order spatial derivatives (for viscosity)
    u_xx = compute_gradients(u_x, points)[:, 0:1]
    u_yy = compute_gradients(u_y, points)[:, 1:2]
    u_zz = compute_gradients(u_z, points)[:, 2:3]
    
    v_xx = compute_gradients(v_x, points)[:, 0:1]
    v_yy = compute_gradients(v_y, points)[:, 1:2]
    v_zz = compute_gradients(v_z, points)[:, 2:3]

    w_xx = compute_gradients(w_x, points)[:, 0:1]
    w_yy = compute_gradients(w_y, points)[:, 1:2]
    w_zz = compute_gradients(w_z, points)[:, 2:3]

    # Construct the Navier-Stokes Residuals
    # Continuity equation (Mass conservation div(u) = 0
    f_c = u_x + v_y + w_z

    """Momentum residuals: $\mathcal{R}_q = q_t + (\mathbf{u}\cdot\nabla)q + p_q - \nu\nabla^2 q = 0$
    $\mathcal{R}_u = u_t + (u\,u_x + v\,u_y + w\,u_z) + p_x - \nu(u_{xx}+u_{yy}+u_{zz})$
    $\mathcal{R}_v = v_t + (u\,v_x + v\,v_y + w\,v_z) + p_y - \nu(v_{xx}+v_{yy}+v_{zz})$
    $\mathcal{R}_w = w_t + (u\,w_x + v\,w_y + w\,w_z) + p_z - \nu(w_{xx}+w_{yy}+w_{zz})$
    """

    # Momentum Equations
    f_u = u_t + (u * u_x + v * u_y + w * u_z) + p_x - NU * (u_xx + u_yy + u_zz)
    f_v = v_t + (u * v_x + v * v_y + w * v_z) + p_y - NU * (v_xx + v_yy + v_zz)
    f_w = w_t + (u * w_x + v * w_y + w * w_z) + p_z - NU * (w_xx + w_yy + w_zz)

    # The loss is the Mean Squared Error of the residuals (we want them to be exactly 0)
    loss_pde = torch.mean(f_c**2) + torch.mean(f_u**2) + torch.mean(f_v**2) + torch.mean(f_w**2)
    return loss_pde

def boundary_loss(model, domain):
    """
    Enforces boundary conditions (e.g., fluid entering the tunnel, stopping at the obstacle).
    """
    bcs = domain.sample_boundaries(BATCH_SIZE_BOUNDARY)
    
    # Dynamically grab the device the model is currently residing on
    device = next(model.parameters()).device
    
    # Inlet constraint: Fluid moves right at 1.0 m/s (u=1, v=0, w=0)
    inlet_pts = bcs["inlet"].to(device)
    inlet_pred = model(inlet_pts)
    loss_inlet = torch.mean((inlet_pred[:, 0] - 1.0)**2) + \
                 torch.mean((inlet_pred[:, 1])**2) + \
                 torch.mean((inlet_pred[:, 2])**2)
                 
    # Obstacle constraint: No-slip condition (velocity is 0 at the wall)
    obs_pts = bcs["obstacle"].to(device)
    obs_pred = model(obs_pts)
    loss_obs = torch.mean(obs_pred[:, 0:3]**2)
    
    return loss_inlet + loss_obs

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = FluidPINN().to(device)
    domain = WindTunnelDomain()

    # Adam Optimizer: standard for initial PINN convergence
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        # Sample interior points and compute PDE loss
        interior_points = domain.sample_interior(BATCH_SIZE_INTERIOR).to(device)
       
        # Calculate losses
        l_pde = navier_stokes_loss(model, interior_points)
        l_bc = boundary_loss(model, domain)

        total_loss = l_pde + l_bc
        total_loss.backward()
        optimizer.step()

        if epoch % 50  == 0:
            print(f"Epoch {epoch}/{EPOCHS} | PDE Loss: {l_pde.item():.6f} | BC Loss: {l_bc.item():.6f}")

    print("Training complete!")
    # TODO add the ONNX export logic here later
    torch.save(model.state_dict(), "pinn_fluid.pth")


if __name__ == "__main__":
    train()