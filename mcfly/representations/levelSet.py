from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import numpy as np
from skimage.measure import marching_cubes
import torch
from trimesh import Trimesh


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
    def dim(self) -> int:
        return len(self._grid)

    @property
    def grid_tensor(self) -> torch.Tensor:
        """Returns the meshgrid as a single torch tensor."""
        return torch.stack(self._grid, dim=-1)

    @property
    def level_set_numpy(self) -> np.ndarray:
        """Returns the current level set function as 3D numpy array."""
        return self._level_set_function.detach().cpu().numpy()

    @property
    def normals(self) -> torch.Tensor:
        """Computes the normals of the level set function using numeric differentiation."""
        grad = torch.stack(torch.gradient(self._level_set_function, spacing=list(self.spacing)), dim=-1)
        return -grad / torch.linalg.norm(grad, dim=-1, keepdim=True)

    @property
    def spacing(self) -> torch.Tensor:
        """Returns the spacing of the grid."""
        return torch.stack((self._grid[0][:, 0, 0], self._grid[1][0, :, 0], self._grid[2][0, 0, :]), dim=0)

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

    def get_mesh_data(self, level: Optional[float] = None):
        """Performs a reinitialization using the marching cubes algorithm."""
        if level is None:
            level = self._default_level
        dx, dy, dz = map(lambda x: x.item(), self.spacing[:, 1] - self.spacing[:, 0])
        verts, faces, normals, values = marching_cubes(self.level_set_numpy, level=level, spacing=(dx, dy, dz))
        verts = verts + np.array([self._grid[0].min().item(), self._grid[1].min().item(), self._grid[2].min().item()])
        return verts, faces, normals, values

    def get_trimesh(self, level: Optional[float] = None) -> Trimesh:
        """Returns the mesh data as a trimesh object."""
        verts, faces, normals, values = self.get_mesh_data(level)
        return Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    def reinitialize(self):
        """Reinitializes the level set function."""
        raise NotImplementedError()  # scikit-fmm

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