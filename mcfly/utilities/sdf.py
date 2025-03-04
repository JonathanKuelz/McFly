from __future__ import annotations

from itertools import chain
from typing import Any, Callable, Iterable, List, Sequence, Tuple

from curobo.geom.sdf.world import CollisionCheckerType, WorldCollision, WorldCollisionConfig
from curobo.geom.sdf.world_mesh import WorldMeshCollision
from curobo.geom.types import Mesh, WorldConfig
from curobo.types.base import TensorDeviceType
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from tqdm import tqdm

from mcfly.utilities.curobo import get_sdf


class Sdf:
    """
    A wrapper that can be used to query signed distance fields.

    The Signed Distance is defined negative outside of volumes and positive inside.
    """

    max_query_size: int = 4096  # Will at most query this many points at once

    def __error__(self, *args, **kwargs):
        raise NotImplementedError("Your SDF must implement a callable function.")

    def __init__(self, *args, **kwargs):
        self.callable = self.__error__

    def __call__(self, pts: torch.Tensor, quiet: bool = True) -> torch.Tensor:
        """Returns the signed distance from the SDF to the given points."""
        shape = pts.shape
        pt_dim = shape[-1]
        query = pts.view(-1, pt_dim)
        dist = torch.empty(query.shape[0], device=query.device)

        splits = torch.split(query, self.max_query_size)
        for i, split in enumerate(tqdm(splits, disable=quiet or len(splits) < 20)):
            dist[i * self.max_query_size:(i + 1) * self.max_query_size] = self.callable(split)

        return dist.view(*shape[:-1])

    @classmethod
    def from_spheres(cls, sph: torch.Tensor):
        """
        Returns the signed distance field of a collection of spheres.

        Args:
            sph: A tensor of shape (N, 4) where the first three dimensions are the center of the sphere and the last
                dimension is the radius.
        """

    def approximate_inertia(self,
                            center: torch.Tensor,
                            dims: Iterable[float],
                            pts_per_dim: int,
                            r: float = 0.0
                            ):
        """
        Approximates a unitless inertia by checking where in the sampled volume the SDF is positive.
        """
        scale = (np.mean(dims) / pts_per_dim) ** 3  # Approximates the volume of a single voxel
        grid, d = self.sample(center, dims, pts_per_dim, r)
        mask = d > 0  # Inside
        pts = torch.stack(grid, dim=-1)[mask] - center
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        xs, ys, zs = x ** 2, y ** 2, z ** 2
        I = torch.tensor([
           [(ys + zs).sum() / 2, -(x * y).sum(),       -(x * z).sum()  ],
           [0,                    (xs + zs).sum() / 2, -(y * z).sum()  ],
           [0,                    0,                    (xs + ys).sum() / 2]
        ]) * scale
        return I + I.T


    def approximate_volume(self,
                           center: torch.Tensor,
                           dims: Iterable[float],
                           pts_per_dim: int,
                           r: float = 0.0
                           ) -> float:
        """Approximates the volume of the SDF by checking where in the sampled volume the SDF is positive."""
        scale = (np.mean(dims) / pts_per_dim) ** 3  # Approximates the volume of a single voxel
        grid, d = self.sample(center, dims, pts_per_dim, r)
        return (d > 0).sum().item() * scale


    def boolean_union(self, other: Sdf) -> Sdf:
        """
        Returns the signed distance field of the boolean union of two SDFs.

        Given two SDFs, A and B, this will result in a SDF the represents a geometry that would result when merging
        SDF B and SDF A.
        """

        def query_function(pts: torch.Tensor) -> torch.Tensor:
            return torch.maximum(self(pts, quiet=True), other(pts, quiet=True))

        return CallableSdf(query_function)

    def boolean_intersection(self, other: Sdf) -> Sdf:
        """
        Returns the signed distance field of the boolean intersection of two SDFs.

        Given two SDFs, A and B, this will result in a SDF the represents a geometry that would result when
        subtracting SDF B from SDF A.
        """

        def query_function(pts: torch.Tensor) -> torch.Tensor:
            return torch.minimum(self(pts, quiet=True), other(pts, quiet=True))

        return CallableSdf(query_function)

    def boolean_difference(self, other: Sdf) -> Sdf:
        """
        Returns the signed distance field of the boolean difference of two SDFs.

        Given two SDFs, A and B, this will result in a SDF the represents a geometry that would result when
        subtracting SDF B from SDF A.
        """

        def query_function(pts: torch.Tensor) -> torch.Tensor:
            return torch.minimum(self(pts, quiet=True), -other(pts, quiet=True))

        return CallableSdf(query_function)


    def plot(self,
             center: torch.Tensor,
             dims: Iterable[float],
             pts_per_dim: int,
             color: Any = None,
             alpha: float = 0.2,
             hide_threshold: float = 0.0
             ):
        """
        Visualizes the SDF as a 3D plot.
        """
        pt_radius = np.mean(dims) / pts_per_dim
        grid, d = self.sample(center, dims, pts_per_dim, r=pt_radius)
        d = d.detach().cpu().numpy()
        mask = hide_threshold < d

        grid = [g.detach().cpu().numpy()[mask] for g in grid]
        d = d[mask]

        if color is None:
            color = d

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(grid[0], grid[1], grid[2], c=color, cmap='viridis', alpha=alpha, edgecolors=None)

        # Set limits
        center = center.detach().cpu().numpy()
        ax.set_xlim(center[0] - dims[0] / 2, center[0] + dims[0] / 2)
        ax.set_ylim(center[1] - dims[1] / 2, center[1] + dims[1] / 2)
        ax.set_zlim(center[2] - dims[2] / 2, center[2] + dims[2] / 2)
        ax.set_axis_off()

        plt.tight_layout()
        plt.show()

    def pv_plot(self,
                center: torch.Tensor,
                dims: Sequence[float],
                pts_per_dim: int,
                hide_threshold: float = 0.0
                ):
        """
        Visualizes the SDF as a 3D plot.
        """
        if isinstance(dims, torch.Tensor):
            dims = dims.detach().cpu().numpy()
        pt_radius = np.mean(dims) / pts_per_dim
        grid, d = self.sample(center, dims, pts_per_dim, r=pt_radius)
        d = d.detach().cpu().numpy()
        mask = hide_threshold < d

        grid = [g.detach().cpu().numpy()[mask] for g in grid]
        d = d[mask]

        scale = np.mean(dims)
        pts = np.stack(grid, axis=-1)
        pv.plot(pts,
                scalars=d,
                style='points_gaussian',
                render_points_as_spheres=True,
                point_size=10 / scale,
                show_scalar_bar=False
                )

    def sample(self,
               center: torch.Tensor,
               dims: Iterable[float],
               pts_per_dim: int,
               r: float = 0.0
               ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Samples the SDF at a grid of points.
        """
        pts = list()
        for i, dim in enumerate(dims):
            pts.append(torch.linspace(center[i] - dim / 2, center[i] + dim / 2, pts_per_dim))
        else:
            assert len(pts) == 3, "Only 3D SDFs are supported."
        grid = torch.meshgrid(pts)
        points = torch.stack(grid, dim=-1)
        if r > 0:
            points = torch.concatenate([points, torch.ones((*points.shape[:-1], 1), device=points.device) * r], dim=-1)
        d = self(points)
        return grid, d

class CallableSdf(Sdf):
    """This class can implement arbitrarily complex SDFs."""

    def __init__(self, query_function: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.callable = query_function


class CuroboMeshSdf(Sdf):
    """A signed distance function representing a curobo mesh."""

    def __init__(self, world_collision_model: WorldMeshCollision):
        super().__init__()
        self.wcm: WorldMeshCollision = world_collision_model
        self.callable = self._callable

    def _callable(self, pts: torch.Tensor) -> torch.Tensor:
        if pts.shape[-1] == 3:
            pts = torch.concatenate([pts, torch.zeros((*pts.shape[:-1], 1), device=pts.device)], dim=-1)
        query_spheres = {'query_spheres': pts.view(1, 1, -1, 4)}
        sdf_info = get_sdf(query_spheres, self.wcm)
        return sdf_info['query_spheres'].distances

    @classmethod
    def from_meshes(cls, meshes: List[Mesh], max_distance: float):
        """Sets up a Collision World and the corresponding SDF from a list of curobo meshes."""
        cfg = WorldConfig(mesh=meshes)
        ccfg = WorldCollisionConfig(
            max_distance=max_distance,
            tensor_args=TensorDeviceType(),
            world_model=cfg,
            checker_type=CollisionCheckerType.MESH,
            cache={'mesh': len(meshes)}
        )
        world_collision = WorldMeshCollision(ccfg)
        return cls(world_collision)

    def discretize(self, pts_per_dim: int) -> SphereSdf:
        """
        Returns a discretized version of the SDF.
        """
        vertices = torch.tensor(list(chain.from_iterable(m.vertices for m in self.wcm.world_model.mesh)))
        center = (torch.max(vertices, dim=0)[0] + torch.min(vertices, dim=0)[0]) / 2
        dims = torch.max(vertices, dim=0)[0] - torch.min(vertices, dim=0)[0]
        r = dims.max() / pts_per_dim
        grid, d = self.sample(center, dims, pts_per_dim, r=r)
        include = d >= 0

        pts = torch.stack(grid, dim=-1)[include]
        r = torch.ones(pts.shape[:-1], device=pts.device) * r
        return SphereSdf(torch.concat([pts, r[..., None]], dim=-1))

    @property
    def meshes(self) -> List[Mesh]:
        return self.wcm.world_model.mesh

    @property
    def mesh_names(self) -> List[str]:
        return [m.name for m in self.wcm.world_model.mesh]

    @property
    def mesh_poses(self) -> List[List[float]]:
        return [m.pose for m in self.wcm.world_model.mesh]

class SphereSdf(Sdf):
    """A signed distance function representing a collection of spheres."""

    def __init__(self, sph: torch.Tensor):
        super().__init__()
        self.sph = sph
        self.callable = self._callable

    def boolean_union(self, other: Sdf) -> Sdf:
        """
        Returns the signed distance field of the boolean union of two SDFs.

        Given two SDFs, A and B, this will result in a SDF the represents a geometry that would result when merging
        SDF B and SDF A.
        """
        if not isinstance(other, SphereSdf):
            return super().boolean_union(other)

        sph = torch.concat([self.sph, other.sph], dim=0)
        return SphereSdf(sph)

    def _callable(self, pts: torch.Tensor) -> torch.Tensor:
        if pts.shape[-1] == 4:
            r = pts[..., 3]
            pts = pts[..., :3]
        else:
            r = torch.zeros(pts.shape[:-1], device=pts.device)
        d, idx = torch.min(torch.linalg.norm(pts[..., None, :] - self.sph[..., :3], dim=-1), dim=-1)
        d = d - self.sph[idx, 3] - r
        return -d
