import torch
import numpy as np

class WindTunnelDomain:
    def __init__(self, x_range = (-2.0, 4.0), y_range = (-1.0, 1.0), z_range = (-1.0, 1.0),
                 t_range = (0.0, 5.0), obstacle_center = (0.0, 0.0, 0.0), obstacle_radius = 0.5 ):
        
        """
        Defines the 3D spatial and temporal domain for the wind tunnel.
        """
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.t_range = t_range
        self.obs_c = obstacle_center
        self.obs_r = obstacle_radius


    def _random_tensor(self, N, range_tuple):
        """Helper to generate $N $ random uniform points within a given range."""
        return torch.rand(N,1) * (range_tuple[1]- range_tuple[0]) + range_tuple[0]
    
    def sample_interior(self, N):
        """
        Sample $N$ random points inside the wind tunnel domain, excluding the inside of the obstacle.
        Returns: tensor of shape $(N, 4) \rightarrow (x, y, z, t)$
        """
        # Generate slightly more points than needed because we will filter out points inside the obstacle
        x = self._random_tensor(int(N * 1.2), self.x_range)
        y = self._random_tensor(int(N * 1.2), self.y_range)
        z = self._random_tensor(int(N * 1.2), self.z_range)
        t = self._random_tensor(int(N * 1.2), self.t_range)

        # Filter out points inside the spherical obstacle
        distances = torch.sqrt((x - self.obs_c[0])**2 + (y - self.obs_c[1])**2 + (z - self.obs_c[2])**2)
        mask = (distances > self.obs_r).squeeze()
        
        #Apply the mask to keep exactly $N$ points
        x, y, z, t = x[mask][:N], y[mask][:N], z[mask][:N], t[mask][:N]

        # Stack into $(x, y, z, t)$ tensor
        return torch.cat([x ,y ,z ,t], dim=1).requires_grad_(True)

    def sample_boundaries(self, N_per_boundary):
        """
        Samples points on the wind tunnel walls, inlets, outlets and obstacle.
        """

        #Inlet (x = x_range[0]) - Fluid enters here
        inlet_y = self._random_tensor(N_per_boundary, self.y_range)
        inlet_z = self._random_tensor(N_per_boundary, self.z_range)
        inlet_t = self._random_tensor(N_per_boundary, self.t_range)
        inlet_x = torch.full_like(inlet_y, self.x_range[0])
        inline_pts = torch.cat([inlet_x, inlet_y, inlet_z, inlet_t], dim=1).requires_grad_(True)

        # Obstacle surface (Sphere) - No-slip condition $(u, v, w) = (0, 0, 0)$
        # Using spherical coordinates to get uniform points on the sphere surface
        theta = self._random_tensor(N_per_boundary, (0, 2 * np.pi))  # azimuthal angle
        phi = torch.acos(self._random_tensor(N_per_boundary, (-1, 1)))  # polar angle
        obs_x = self.obs_c[0] + self.obs_r * torch.sin(phi) * torch.cos(theta)
        obs_y = self.obs_c[1] + self.obs_r * torch.sin(phi) * torch.sin(theta)
        obs_z = self.obs_c[2] + self.obs_r * torch.cos(phi)
        obs_t = self._random_tensor(N_per_boundary, self.t_range)
        obs_pts = torch.cat([obs_x, obs_y, obs_z, obs_t], dim=1).requires_grad_(True)

        #For full implementation  we would sample the outlet and the walls as well, we will however keep it minimal for now to test the network.
        return {'inlet': inline_pts, 'obstacle': obs_pts}