from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import numpy as np
import pyacvd
import pyvista as pv
import skfmm
from skimage.measure import marching_cubes
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from trimesh import Trimesh

from docbrown.utilities.geometry import pv_to_trimesh


class LevelSetRepresentation:
    """A class to represent and work with level set functions."""

    def __init__(self,
                 grid: Sequence[torch.Tensor],
                 level_set_function: Union[Callable[[torch.Tensor], torch.Tensor], torch.Tensor],
                 *,
                 default_level: float = 0.0,
                 sanity_checks: bool = True,
                 ):
        """
        The grid must be evenly spaced or else numeric differencing will silently return wrong results.
        """
        self._grid = tuple(grid)
        if isinstance(level_set_function, torch.Tensor):
            self._level_set_function = level_set_function
        else:
            self._level_set_function = level_set_function(self.grid_tensor)
        self._default_level = default_level
        if sanity_checks:
            self._check_sanity()

    @property
    def aabb(self, level: Optional[float] = None) -> torch.Tensor:
        """Returns the axis-aligned bounding box of the level set function."""
        if level is None:
            level = self._default_level
        interior = self.get_interior_mask(level)
        interior_points = self.grid_tensor[interior]
        return torch.stack([interior_points.min(dim=0)[0], interior_points.max(dim=0)[0]])

    @property
    def center(self) -> torch.Tensor:
        """Returns the center of the grid."""
        return (self.grid_tensor[0, 0, 0] + self.grid_tensor[-1, -1, -1]) / 2

    @property
    def dim(self) -> int:
        return len(self._grid)

    @property
    def f(self) -> torch.Tensor:
        """Returns the level set function."""
        return self._level_set_function

    @property
    def grid_tensor(self) -> torch.Tensor:
        """Returns the meshgrid as a single torch tensor."""
        return torch.stack(self._grid, dim=-1)

    @property
    def level_set_numpy(self) -> np.ndarray:
        """Returns the current level set function as 3D numpy array."""
        return self._level_set_function.detach().cpu().numpy()

    @property
    def mean_curvature(self) -> torch.Tensor:
        """Computes the mean curvature of the level set function using numeric differentiation."""
        # TODO: Not sure if this is correct -- results seem reasonable but not exactly as expected
        dx, dy, dz = map(lambda x: x.item(), self.spacing[:, 1] - self.spacing[:, 0])
        normal = self.normals
        # div F = dF_x / dx + dF_y / dy + dF_z / dz
        kappa = sum(torch.gradient(normal[..., i], spacing=s, dim=i)[0] for s, i in zip((dx, dy, dz), range(3)))
        return kappa

    @property
    def normals(self) -> torch.Tensor:
        """Computes the normals of the level set function using numeric differentiation."""
        grad = torch.stack(torch.gradient(self._level_set_function, spacing=list(self.spacing)), dim=-1)
        n = -grad / torch.linalg.norm(grad, dim=-1, keepdim=True)
        n[torch.isnan(n)] = 0  # This is true for all points with equal distance to multiple surface points
        return n

    @property
    def spacing(self) -> torch.Tensor:
        """Returns the spacing of the grid."""
        return torch.stack((self._grid[0][:, 0, 0], self._grid[1][0, :, 0], self._grid[2][0, 0, :]), dim=0)

    def copy(self) -> LevelSetRepresentation:
        """Returns a copy of the level set representation."""
        return LevelSetRepresentation(tuple(t.clone() for t in self._grid),
                                      self._level_set_function.clone(),
                                      default_level=self._default_level)

    @torch.no_grad()
    def compute_mesh_velocity_field(self,
                                    mesh_gradients: torch.Tensor,
                                    mesh_points: torch.Tensor,
                                    max_batch_size: int = 256,
                                    variance: Optional[float] = None):
        """
        Every mesh point with a gradient defines a gaussian distribution. The velocity field is the sum of the
        distributions at all grid points.

        Note: The current implementation is inefficient. FFT could be used, but seems to suffer from numerical issues.
            Maybe, even a gaussian splatting to the implicit surface could be done, but this requires some engineering.

        :param mesh_gradients: The gradients of the mesh points. Should be of shape (N, 3).
        :param mesh_points: The mesh points. Should be of shape (N, 3).
        :param max_batch_size: The maximum number of gradients/points to process at once. Note that for a grid
            resolution of 100, a single point already takes up around 5MB of GPU memory.
        :param variance: Variance of the Gaussian distribution. If None, it is computed such that approximately 95% of
            the mesh point gradient is distributed within the closest X grid points in each dimension and direction (so,
            within a cube of 2X grid points).
        """
        nonzero_grad = (mesh_gradients != 0).all(dim=1)
        if not nonzero_grad.any():
            print("WARNING: NO CONTACT")
            return

        if variance is None:
            dx, dy, dz = map(lambda x: x.item(), self.spacing[:, 1] - self.spacing[:, 0])
            variance = (3. * min(dx, dy, dz) / 2.45) ** 2

        p = mesh_points[nonzero_grad]
        grad = mesh_gradients[nonzero_grad]

        v = torch.zeros_like(self.normals)
        batch_indices = torch.arange(p.shape[0], device=p.device).split(max_batch_size)
        for idx in batch_indices:
            gaussian = MultivariateNormal(loc=p[idx], covariance_matrix=torch.eye(3) * variance)
            density = torch.exp(gaussian.log_prob(self.grid_tensor.view(-1, 1, 3)))
            density = density / density.norm(dim=0, keepdim=True)  # Normalize it, because CDF != discrete probability
            v += torch.einsum('ij,jk->ijk', density, grad[nonzero_grad][idx]).sum(dim=1).view(self.normals.shape)
        return v / torch.linalg.norm(v, dim=-1).max()

    def _get_gradients_impulse_grid(self, p: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """
        This function takes a tensor of Nx3 gradient vectors (i.e., a sparse velocity field) and returns the dense
        impulse grid (i.e., a step size field with zeros everywhere except at the grid points).

        In contrast to other methods, the points p do not need to be on the grid exactly. This function will find the
        closest grid point automatically and apply them there.

        Usage:
        gaussian_impulse = self._get_gradients_impulse_grid(p, grad)
        mag_v = normal_distribution_density_fourier(self.grid_tensor, gaussian_impulse, variance)

        :param p: The grid points where the impulse is defined. Should be of shape (N, 3).
        :return:
        """
        # Convert centers to integer grid indices
        x = self.grid_tensor[:, 0, 0, 0]
        y = self.grid_tensor[0, :, 0, 1]
        z = self.grid_tensor[0, 0, :, 2]


        # Create impulse function at the Gaussian centers
        indices = (torch.stack([x, y, z], dim=-1).unsqueeze(1) - p.unsqueeze(0)).abs().argmin(dim=0)
        grad_grid = torch.zeros_like(self.grid_tensor)
        grad_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = grad
        impulse = torch.einsum('ijkl,ijkl->ijk', grad_grid, self.normals)
        return impulse

    def evolve(self,
               v: Union[Callable[[torch.Tensor], torch.Tensor], torch.Tensor],
               dt: float = 1.0,
               ):
        """Evolves the level set function using a velocity field."""
        if not isinstance(v, torch.Tensor):
            v = v(self.grid_tensor)
        if self.dim == 2:
            step = torch.einsum('ijk,ijk->ij', v, self.normals)
        else:
            step = torch.einsum('ijkl,ijkl->ijk', v, self.normals)
        self._level_set_function = self._level_set_function + dt * step

    def get_interior_mask(self, level: Optional[float] = None) -> torch.Tensor:
        """Returns a mask of the interior points of the level set function."""
        if level is None:
            level = self._default_level
        return self._level_set_function >= level

    def get_isosurface_mask(self, level: Optional[float] = None, eps: float = 1e-3) -> torch.Tensor:
        """Returns a mask of the grid for all values within eps of level."""
        if level is None:
            level = self._default_level
        return torch.abs(self._level_set_function - level) <= eps

    def get_mesh_data(self, level: Optional[float] = None):
        """Performs a reinitialization using the marching cubes algorithm."""
        if level is None:
            level = self._default_level
        dx, dy, dz = map(lambda x: x.item(), self.spacing[:, 1] - self.spacing[:, 0])
        verts, faces, normals, values = marching_cubes(self.level_set_numpy, level=level, spacing=(dx, dy, dz))
        verts = verts + np.array([self._grid[0].min().item(), self._grid[1].min().item(), self._grid[2].min().item()])
        return verts, faces, normals, values

    def get_trimesh(self, level: Optional[float] = None, max_num_vertices: Optional[int] = None) -> Trimesh:
        """Returns the mesh data as a trimesh object."""
        verts, faces, normals, values = self.get_mesh_data(level)
        tm = Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        if max_num_vertices is None:
            return tm
        pvm = pv.wrap(tm)

        pyacvd.clustering.LOG.setLevel(0)  # Suppress pyacvd logging
        clus = pyacvd.Clustering(pvm)
        clus.cluster(max_num_vertices)  # Using clus.subdivide() before this might help if the initial mesh is coarse
        remeshed = clus.create_mesh()  # Could also use the .decimate() method to reduce the number of vertices again
        return pv_to_trimesh(remeshed)

    def reinitialize(self):
        """Reinitializes the level set function using the fast marching method."""
        dx, dy, dz = map(lambda x: x.item(), self.spacing[:, 1] - self.spacing[:, 0])
        if not dx == dy == dz:
            raise ValueError("Reinitialization currently assumes a uniformly spaced grid in all dimensions.")
        sdf = skfmm.distance(self.level_set_numpy) * dx
        self._level_set_function = torch.tensor(sdf).to(self._level_set_function)

    def soft_surface_dirac(self, eps_sign: Optional[float] = None):
        """
        Get a relaxed version of the surface Dirac delta function.

        See Allaire et al., "Structural optimization using sensitivity analysis and a level set method", 2004, p. 22
        """
        if eps_sign is None:
            dx, dy, dz = map(lambda x: x.item(), self.spacing[:, 1] - self.spacing[:, 0])
            eps_sign = max(dx, dy, dz) / 20  # Should result in the surface being ~2 cells wide
        signum = self._level_set_function / torch.sqrt(self._level_set_function ** 2 + eps_sign ** 2)
        return sum(g.abs() for g in torch.gradient(signum)) / 3

    def visualize(self,
                  v: Optional[torch.Tensor] = None,
                  level: Optional[float] = None,
                  min_abs_magnitude: float = 0.0,
                  mesh_kwargs: Optional[dict] = None,
                  point_kwargs: Optional[dict] = None,
                  ):
        """Creates a mesh of the current level set, optionally together with the magnitude of expected change given
        a velocity field."""
        import pyvista as pv
        if point_kwargs is None:
            point_kwargs = {}
        point_kwargs.setdefault('opacity', 0.1)
        point_kwargs.setdefault('show_scalar_bar', False)
        if mesh_kwargs is None:
            mesh_kwargs = {}
        mesh_kwargs.setdefault('opacity', 0.9)
        mesh_kwargs.setdefault('show_edges', True)
        mesh_kwargs.setdefault('color', 'white')

        tm = self.get_trimesh(level)
        pv_mesh = pv.wrap(tm)

        plotter = pv.Plotter()
        plotter.add_axes()
        plotter.add_axes_at_origin()
        plotter.add_mesh(pv_mesh, **mesh_kwargs)
        if v is not None:
            magnitude = torch.einsum('ijkl,ijkl->ijk', v, self.normals)
            mask = torch.abs(magnitude) >= min_abs_magnitude
            magnitude = magnitude[mask].detach().cpu().numpy().reshape(-1, 1)
            grid = self.grid_tensor[mask].detach().cpu().numpy().reshape(-1, 3)

            pos = (magnitude >= 0).squeeze()
            if pos.any():
                plotter.add_points(grid[pos], scalars=magnitude[pos], cmap="Greens", **point_kwargs)
            if not pos.all():
                plotter.add_points(grid[~pos], scalars=magnitude[~pos], cmap="Reds", **point_kwargs)
        plotter.show()

    def _check_sanity(self):
        """Checks the sanity of the level set function, the provided dimensions etc."""
        if not isinstance(self._level_set_function, torch.Tensor):
            raise ValueError("Level set function must be a torch tensor.")
        grid_shape = self._grid[0].shape
        if not all(t.shape == grid_shape for t in self._grid):
            raise ValueError("All grid tensors must have the same shape.")
        if not self._level_set_function.shape == grid_shape:
            raise ValueError("Level set function shape does not match grid shape.")
        if not self.dim in (2, 3):
            raise ValueError("Level set function must be 2D or 3D.")