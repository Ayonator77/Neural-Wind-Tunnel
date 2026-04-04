import torch
import torch.nn as nn
import numpy as np

class FourierEmbedding(nn.Module):
    """
    Embeds low-dimensional spatial/temporal inputs into a high-dimensional Fourier feature space.
    This mitigates the spectral bias of neural networks, allowing the networks to learn high-frequency details
    (chaotic eddies, sharp gradients) in fluid dynamics.

    Mathematical Formulation:
    $$\gamma(\mathbf{v})=[\cos(2\pi\mathbf{B}\mathbf{v}),\sin(2\pi\mathbf{B}\mathbf{v})]^T$$
    where $\mathbf{v}$ is the input coordinate $(x,y,z,t)$ and $\mathbf{B}$ is a random Gaussian matrix.
    """
    def __init__(self, in_features=4, mapping_size=128, scale=10.0):
        super().__init__()
        self.in_features = in_features
        self.mapping_size = mapping_size

        #Initialize a non-trainable random Gaussian matrix B
        B = torch.randn((in_features, mapping_size)) * scale
        self.register_buffer('B', B)
    
    def forward(self, x):
        # x shape: (batch_size, in_features)
        # x_proj shape: (batch_size, mapping_size)
        x_proj = (2.0 * np.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class FluidPINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for 3d Incompressible Navier-Stokes.

    Inputs : $\mathbf{x} = (x, y, z, t)$
    Outputs : $\mathbf{y} = (u, v, w, p)$

    Utilizes a Modified Residual Architecure with SiLU (Swish) activations to ensure
    non-zero second derivatives, which is crucial for computing the kinematic viscosity term
    in the momentum equations ($\nu\nabla^2\mathbf{u}$).
    """
    def __init__(self, in_dim=4, out_dim=4, fourier_features=128, hidden_dim=256, layers=6):
        super().__init__()
        self.embedding = FourierEmbedding(in_features=in_dim, mapping_size=fourier_features)

        # The embedded dimension is 2x the mapping size due to concat(sin, cos)
        embedded_dim = 2 * fourier_features

        self.fc_in = nn.Linear(embedded_dim, hidden_dim)

        #Hidden layer using a ModuleList for skip connections
        self.hidden_layers = nn.ModuleList()
        for _ in range(layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.fc_out = nn.Linear(hidden_dim, out_dim)

        # SiLU ($x\cdot\sigma(x)$) provides smooth, infinite differentiability
        self.activation = nn.SiLU()

    def forward(self, x):
        # Feature embedding
        emb = self.embedding(x)

        # Initial Linear Projection
        h = self.activation(self.fc_in(emb))

        # Modified Residual Connections
        """
        $h_{i+1}=\frac{h_i+\sigma(W_i h_i+b_i)}{\sqrt{2}}$
        """
        for layer in self.hidden_layers:
            h_new = self.activation(layer(h))
            # Skip connection: add previous state to the new state 
            h = (h + h_new) / np.sqrt(2.0) 
        
        # Final Output Projection: Returns velocity (u, v, w) and pressure (p)
        out = self.fc_out(h)
        return out
        

