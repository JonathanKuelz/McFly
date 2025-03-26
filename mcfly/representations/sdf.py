from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import chain
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

from curobo.geom.sdf.world import CollisionCheckerType, WorldCollisionConfig
from curobo.geom.sdf.world_mesh import WorldMeshCollision
from curobo.geom.transform import pose_inverse, pose_to_matrix
from curobo.geom.types import Mesh, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util.logger import setup_curobo_logger
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from pyglet.shapes import vertex_source
import pymeshfix
import pyvista as pv
import torch
from tqdm import tqdm

from mcfly.utilities.curobo import get_sdf


class Sdf(ABC):
    """
    A wrapper that can be used to query signed distance fields.

    The Signed Distance is defined negative outside of volumes and positive inside.
    """

    max_query_size: int = 4096  # Will at most query this many points at once

    def __error__(self, *args, **kwargs):
        raise NotImplementedError("Your SDF must implement a callable function.")

    def __init__(self, *args, **kwargs):
        self.callable = self.__error__
        self._is_accurate = True

    def __call__(self,
                 pts: torch.Tensor,
                 gradients: bool = False,
                 quiet: bool = True,
                 ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Returns the signed distance from the SDF to the given points."""
        shape = pts.shape
        pt_dim = shape[-1]
        query = pts.view(-1, pt_dim)
        dist = torch.empty(query.shape[0], device=query.device)
        grad = torch.empty_like(query)

        splits = torch.split(query, self.max_query_size)
        for i, split in enumerate(tqdm(splits, disable=quiet or len(splits) < 20)):
            if gradients:
                d, g = self.callable(split, gradients=True)
                dist[i * self.max_query_size:(i + 1) * self.max_query_size] = d
                grad[i * self.max_query_size:(i + 1) * self.max_query_size] = g
            else:
                dist[i * self.max_query_size:(i + 1) * self.max_query_size] = self.callable(split, gradients=False)

        if gradients:
            return dist.view(*shape[:-1]), grad.view(*shape)

        return dist.view(*shape[:-1])

    @property
    def is_accurate(self) -> bool:
        return self._is_accurate

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
        if isinstance(dims, torch.Tensor):
            dims = dims.detach().cpu().numpy()
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

        def query_function(pts: torch.Tensor, gradients: bool) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            d1 = self(pts, gradients, quiet=True)
            d2 = other(pts, gradients, quiet=True)
            if not gradients:
                return torch.maximum(d1, d2)
            d1, g1 = d1
            d2, g2 = d2
            return torch.maximum(d1, d2), torch.where((d1 >= d2).unsqueeze(1), g1, g2)

        return CallableSdf(query_function)

    def boolean_intersection(self, other: Sdf) -> Sdf:
        """
        Returns the signed distance field of the boolean intersection of two SDFs.

        Given two SDFs, A and B, this will result in a SDF the represents a geometry that would result when
        subtracting SDF B from SDF A.
        """

        def query_function(pts: torch.Tensor, gradients: bool) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            d1 = self(pts, gradients, quiet=True)
            d2 = other(pts, gradients, quiet=True)
            if not gradients:
                return torch.minimum(d1, d2)
            d1, g1 = d1
            d2, g2 = d2
            return torch.minimum(d1, d2), torch.where((d1 <= d2).unsqueeze(1), g1, g2)

        return CallableSdf(query_function)

    def boolean_difference(self, other: Sdf) -> Sdf:
        """
        Returns the signed distance field of the boolean difference of two SDFs.

        Given two SDFs, A and B, this will result in a SDF the represents a geometry that would result when
        subtracting SDF B from SDF A.
        """
        def query_function(pts: torch.Tensor, gradients: bool) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            d1 = self(pts, gradients, quiet=True)
            d2 = other(pts, gradients, quiet=True)
            if not gradients:
                return torch.minimum(d1, -d2)
            d1, g1 = d1
            d2, g2 = d2
            return torch.minimum(d1, -d2), torch.where((d1 <= -d2).unsqueeze(1), g1, -g2)

        return CallableSdf(query_function)

    def discretize(self,
                   center: torch.Tensor,
                   dims: Iterable[float],
                   pts_per_dim: int) -> SphereSdf:
        """
        Returns a discretized version of the SDF.

        TODO: This can be implemented using an octree-like structure similar to the surface detection.
        """
        r = 2 ** .5 * dims.max() / pts_per_dim
        grid, d = self.sample(center, dims, pts_per_dim, r=r)
        include = d >= 0

        pts = torch.stack(grid, dim=-1)[include]
        r = torch.ones(pts.shape[:-1], device=pts.device) * r
        return SphereSdf(torch.concat([pts, r[..., None]], dim=-1))

    def get_surface_points(self,
                           center: torch.Tensor,
                           dims: Iterable[float],
                           refinement_steps: int = 9,
                           limit_to: Optional[str] = None
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the points and their normals of the SDF.

        Args:
            center: The (approximate) center of the SDF.
            dims: The dimensions of the cube to use to sample the SDF.
            refinement_steps: How many times to refine the grid. The final resolution of the grid will be
                $\frac{dims}{8^refinement_steps}$. $6$ seems to be a good and reasonably fast default, 7 already takes
                seconds to compute. Naturally, with one more step, the computation times is multiplied by ~8.
            limit_to: If 'interior' or 'exterior' is provided, the surface points will be limited to points on this
                side of the SDF. Important for boolean operations.
        """
        sph_rad = np.sqrt(3) / 2
        if not isinstance(dims, torch.Tensor):
            dims = torch.tensor(dims)
        dims = dims + dims / 2  # Make sure the centers of all initialized octree spheres are on or inside the SDF
        surface, normals = None, None
        for step in range(refinement_steps):
            query = get_octree_children(center, dims)
            if step < refinement_steps - 1:
                d = self(query)
                mask = torch.abs(d) < sph_rad * dims.max()
                dims = dims / 2
                center = query[mask][..., :3]
            else:
                d, g = self(query, gradients=True)
                mask = torch.abs(d) < sph_rad * dims.max()
                if limit_to == 'interior':
                    mask = mask & (d > 0)
                elif limit_to == 'exterior':
                    mask = mask & (d < 0)
                surface = query[mask]
                normals = g[mask][:, :3]
                # Assuming the normals are correct, this will directly give us the surface points
                surface = surface[:, :3] - (d[mask] - query[mask, 3]).unsqueeze(-1) * normals
        return surface, normals

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

    def plot_reconstructed_surface(self,
                                   center: Optional[torch.Tensor] = None,
                                   dims: Optional[Iterable[float]] = None,
                                   refinement_steps: int = 6,
                                   ):
        """Tries to find surface points and normals and construct a mesh from them."""
        if center is None:
            raise ValueError(f"Center must be provided to plot an sdf of type {type(self)}.")
        if dims is None:
            raise ValueError(f"Dims must be provided to plot an sdf of type {type(self)}.")
        v, f = self.reconstruct_surface_poisson(center, dims, refinement_steps)
        faces = np.hstack([np.full((f.shape[0], 1), 3), f])
        pv_mesh = pv.PolyData(v.astype(np.float32), faces)
        pv_mesh.plot()

    def pv_plot(self,
                center: torch.Tensor,
                dims: Sequence[float],
                pts_per_dim: int,
                hide_threshold: float = 0.0,
                show: bool = True,
                save_at: Optional[str] = None,
                ) -> pv.Plotter:
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

        plotter = pv.Plotter()
        plotter.add_axes()
        plotter.add_axes_at_origin()
        plotter.add_points(pts,
                           scalars=d,
                           style='points_gaussian',
                           render_points_as_spheres=True,
                           point_size=10 / scale,
                           show_scalar_bar=False,
                           )
        if show:
            plotter.show()
        if save_at is not None:
            plotter.save_graphic(save_at)
        return plotter

    def reconstruct_surface_poisson(self,
                                    center: Optional[torch.Tensor] = None,
                                    dims: Optional[Iterable[float]] = None,
                                    refinement_steps: int = 6,
                                    fix_mesh: bool = True,
                                    limit_to: Optional[str] = None
                                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tries to reconstrunct the surface by sampling the SDF and using Poisson reconstruction.

        original code: https://github.com/mkazhdan/PoissonRecon
        see also: https://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html
        Args:
            center: The (approximate) center of the SDF.
            dims: The dimensions of the cube to use to sample the SDF.
            refinement_steps: How many times to refine the grid AND the reconstruction. The final resolution of the grid
            will be $\frac{dims}{8^refinement_steps}$. $6$ seems to be a good and reasonably fast default.
            fix_mesh: Whether to fix the mesh after reconstruction. This is recommended, but takes additional time.
            limit_to: You can limit the surface reconstruction to points in the 'interior' or 'exterior' of the SDF.
        Returns: (vertices, faces)
        """
        points, normals = self.get_surface_points(center, dims, refinement_steps=refinement_steps, limit_to=limit_to)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
        pcd.normals = o3d.utility.Vector3dVector(-normals.detach().cpu().numpy())
        surface, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=pcd, depth=refinement_steps)
        v, f = np.asarray(surface.vertices), np.asarray(surface.triangles)
        if fix_mesh:
            v, f = pymeshfix.clean_from_arrays(v, f)
        return v, f

    def remesh(self,
               center: torch.Tensor,
               dims: torch.Tensor,
               refinement_steps: int = 6,
               limit_to: Optional[str] = None
               ) -> CuroboMeshSdf:
        """Resamples the surface and returns the mesh for the new shape."""
        v, f = self.reconstruct_surface_poisson(center, dims, refinement_steps=refinement_steps, limit_to=limit_to)
        return CuroboMeshSdf.from_meshes(
            [Mesh(
                name='remeshed', vertices=v.astype(np.float32).tolist(),
                faces=f.tolist(), pose=[0., 0., 0., 1., 0., 0., 0.]
                )],
            max_distance=dims.max().item()
            )

    def sample(self,
               center: torch.Tensor,
               dims: Iterable[float],
               pts_per_dim: int,
               r: float = 0.0,
               gradients: bool = False
               ) -> Union[
                    Tuple[List[torch.Tensor], torch.Tensor],
                    Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]
                    ]:
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
        if gradients:
            d, g = self(points, gradients=True)
            return grid, d, g
        d = self(points)
        return grid, d

class CallableSdf(Sdf):
    """This class can implement arbitrarily complex SDFs."""

    def __init__(self,
                 query_function: Callable[[torch.Tensor, bool], torch.Tensor],
                 is_accurate: bool = False
                 ):
        """
        Args:
            query_function: A function that takes a tensor of points and returns the signed distance to the SDF.
            is_accurate: Whether the SDF is accurate. If False, the SDF provided by the query function is only an
                approximation. This commonly happens when combining SDFs, e.g., via boolean operations. See also:
                https://iquilezles.org/articles/interiordistance/
        """
        self._is_accurate = is_accurate
        super().__init__()
        self.callable = query_function


class BoundedSdf(Sdf, ABC):
    """An SDF that is aware of its boundaries."""

    @property
    def aabb(self) -> torch.Tensor:
        """Returns the axis-aligned bounding box of the SDF."""
        return torch.stack([self.get_center() - self.get_dims() / 2, self.get_center() + self.get_dims() / 2])

    @abstractmethod
    def get_center(self) -> torch.Tensor:
        """Returns the center of the SDF."""

    @abstractmethod
    def get_dims(self) -> torch.Tensor:
        """Returns the dimensions of the SDF."""

    def discretize(self,
                   center: Optional[torch.Tensor] = None,
                   dims: Optional[Iterable[float]] = None,
                   pts_per_dim: Optional[int] = 60) -> SphereSdf:
        """Returns a discretized version of the SDF."""
        center = self.get_center() if center is None else center
        dims = self.get_dims() if dims is None else dims
        return super().discretize(center, dims, pts_per_dim)

    def get_surface_points(self,
                           center: Optional[torch.Tensor] = None,
                           dims: Optional[Iterable[float]] = None,
                           refinement_steps: int = 9,
                           limit_to: Optional[str] = None
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the points and their normals of the SDF."""
        center = self.get_center() if center is None else center
        dims = self.get_dims() if dims is None else dims
        return super().get_surface_points(center, dims, refinement_steps, limit_to)

    def pv_plot(self,
                center: Optional[torch.Tensor] = None,
                dims: Optional[Sequence[float]] = None,
                pts_per_dim: int = 60,
                hide_threshold: float = 0.0,
                show: bool = True,
                save_at: Optional[str] = None,
                ) -> pv.Plotter:
        """Adds defaults to the super class plot method."""
        center = center if center is not None else self.get_center()
        dims = dims if dims is not None else self.get_dims() * 1.1
        return super().pv_plot(center, dims, pts_per_dim, hide_threshold, show, save_at)

    def plot_reconstructed_surface(self,
                                   center: Optional[torch.Tensor] = None,
                                   dims: Optional[Iterable[float]] = None,
                                   **kwargs
                                   ):
        center = self.get_center() if center is None else center
        dims = self.get_dims() if dims is None else dims
        return super().plot_reconstructed_surface(center, dims, **kwargs)

    def reconstruct_surface_poisson(self,
                                    center: Optional[torch.Tensor] = None,
                                    dims: Optional[Iterable[float]] = None,
                                    refinement_steps: int = 6,
                                    fix_mesh: bool = True,
                                    limit_to: Optional[str] = None
                                    ) -> Tuple[np.ndarray, np.ndarray]:
        center = self.get_center() if center is None else center
        dims = self.get_dims() if dims is None else dims
        return super().reconstruct_surface_poisson(center, dims, refinement_steps, fix_mesh, limit_to)


setup_curobo_logger('warning')


class CuroboMeshSdf(BoundedSdf):
    """A signed distance function representing a curobo mesh."""

    def __init__(self, world_collision_model: WorldMeshCollision):
        super().__init__()
        self.wcm: WorldMeshCollision = world_collision_model
        self.callable = self._callable
        self._is_accurate = False

    def _callable(self, pts: torch.Tensor, gradients: bool) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if gradients and not pts.requires_grad:
            pts.requires_grad = True
        if pts.shape[-1] == 3:
            pts = torch.concatenate([pts, torch.zeros((*pts.shape[:-1], 1), device=pts.device)], dim=-1)
        query_spheres = {'query_spheres': pts.view(1, 1, -1, 4)}
        sdf_info = get_sdf(query_spheres, self.wcm)
        if gradients:
            grads = sdf_info['query_spheres'].grad.view(1, 1, -1, 4)
            return sdf_info['query_spheres'].distances, grads[..., :3]
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

    def _get_vertex_positions(self) -> torch.Tensor:
        vertices = [m.vertices for m in self.wcm.world_model.mesh]
        positions = list()
        for pose, vert in zip(self.mesh_poses, vertices):
            mat = pose_to_matrix(pose[None, :3], pose[None, 3:])
            vert = torch.concat([torch.tensor(vert), torch.ones(*vert.shape[:-1], 1)], dim=-1).to(mat)
            positions.append(torch.einsum('nij,nj->ni', mat, vert)[..., :3].reshape(-1, 3))
        return torch.concat(positions, dim=0)

    def get_center(self) -> torch.Tensor:
        """Approximates the center of the SDF"""
        vertices = self._get_vertex_positions()
        return (torch.max(vertices, dim=0)[0] + torch.min(vertices, dim=0)[0]) / 2

    def get_dims(self) -> torch.Tensor:
        """Approximates the dimensions of the SDF"""
        vertices = self._get_vertex_positions()
        return torch.max(vertices, dim=0)[0] - torch.min(vertices, dim=0)[0]

    def translate(self, t: torch.Tensor):
        """Translates the whole SDF by t."""
        poses = {
            name: pose for name, pose in zip(self.mesh_names, self.mesh_poses)
        }
        for mesh, pose in poses.items():
            p = pose[:3] + t
            quat = pose[3:]
            new_pose = Pose(p, quat)
            self.wcm.update_mesh_pose(w_obj_pose=new_pose, name=mesh)

    @property
    def meshes(self) -> List[Mesh]:
        return self.wcm.world_model.mesh

    @property
    def mesh_names(self) -> List[str]:
        return [m.name for m in self.wcm.world_model.mesh]

    @property
    def mesh_poses(self) -> List[torch.Tensor]:
        env_idx = 0
        inv = [self.wcm._mesh_tensor_list[1][env_idx, self.wcm.get_mesh_idx(m, env_idx), :7] for m in self.mesh_names]
        poses = list()
        for ip in inv:
            p, quat = pose_inverse(ip[:3], ip[3:])
            poses.append(torch.cat([p, quat]))
        return poses


class SphereSdf(BoundedSdf):
    """A signed distance function representing a collection of spheres."""

    def __init__(self, sph: torch.Tensor):
        super().__init__()
        self.sph = sph
        self.callable = self._callable

    @property
    def aabb(self) -> torch.Tensor:
        """Returns the axis-aligned bounding box of the SDF."""
        sph = self.sph.view(-1, 4)[:, :3]
        bb = torch.tensor([
            [sph[:, 0].min(), sph[:, 1].min(), sph[:, 2].min()],
            [sph[:, 0].max(), sph[:, 1].max(), sph[:, 2].max()]
        ])
        r = self.sph[..., 3].max()
        bb[0] -= r
        bb[1] += r
        return bb

    @property
    def is_accurate(self):
        """
        SphereSdf is accurate only for a single sphere. As multiple spheres are considered a boolean union it becomes
        an approximation in this case.
        """
        return self.sph.shape[0] == 1


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

    def get_center(self) -> torch.Tensor:
        """Approximates the center of the SDF"""
        return self.aabb.mean(dim=0)

    def get_dims(self) -> torch.Tensor:
        """Approximates the dimensions of the SDF"""
        return self.aabb[1] - self.aabb[0]

    def _callable(self, pts: torch.Tensor, gradients: bool) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if pts.shape[-1] == 4:
            r = pts[..., 3]
            pts = pts[..., :3]
        else:
            r = torch.zeros(pts.shape[:-1], device=pts.device)
        deltas = pts[..., None, :] - self.sph[..., :3]
        delta_norm = torch.linalg.norm(deltas, dim=-1)
        d = delta_norm - self.sph[..., 3]
        d, idx = torch.min(d, dim=-1)
        d = d - r
        if gradients:
            g = torch.zeros_like(deltas)
            compute = delta_norm > 0
            g[compute] = deltas[compute] / delta_norm.unsqueeze(-1)[compute]
            grads = torch.concat([-g[torch.arange(g.shape[0]), idx], torch.ones((g.shape[0], 1)).to(g)], dim=-1)
            return -d, grads
        return -d


def get_octree_children(c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Assuming a center c and box dimensions d, this returns the 8 children of the box octree."""
    radius = d.max() / 4
    if c.dim() == 1:
        c = c.unsqueeze(0)
    children = c.unsqueeze(1).repeat_interleave(8, dim=1)
    offset = torch.tensor([
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1]
    ]) * d / 4
    pts = (children + offset).view(-1, 3)
    spheres = torch.concatenate([pts, torch.ones((pts.shape[0], 1), device=pts.device) * radius], dim=-1)
    return spheres