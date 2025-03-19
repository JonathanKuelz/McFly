import hashlib
from typing import List, Optional, Tuple, Union

from curobo.types.math import Pose
import numpy as np
import torch

from biff.optimization.pygmo import UdpWithGradient
from mcfly.representations.sdf import CuroboMeshSdf, Sdf, SphereSdf

class MoveToContactPoint(UdpWithGradient):
    """
    A UDP that moves a portable SDF towards another SDF (represented by spheres) until they are just in contact.
    """

    def __init__(self,
                 moveable_sdf: CuroboMeshSdf,
                 static_sdf: SphereSdf,
                 bounds: Union[float, np.ndarray],
                 move_direction: Optional[torch.Tensor] = None,
                 threshold: float = 0.02,
                 ):
        """
        :param moveable_sdf: The SDF to move, e.g. a gripper finger.
        :param static_sdf: The SDF to move towards, e.g. an object to grasp.
        :param bounds: The bounds of the optimization problem. If a single float is given, the bounds are
            [-bounds, bounds] across all dimensions. If a 1x2 array is given, the bounds are broadcast across
            all dimensions. Lastly, this can be a 3x2 array.
        :param move_direction: The direction to move in. If None, the direction will not be constrained.
        :param threshold: The distance at which the two SDFs are considered to be in contact. Setting this to a
            reasonable value increases convergence speed.
        """
        super().__init__(stores_grad_during_fitness=True)
        self.moveable_sdf = moveable_sdf
        self.static_sdf = static_sdf
        if move_direction is None:
            self.move_direction = None
        else:
            self.move_direction = move_direction.detach().cpu().numpy()
            self.move_direction = self.move_direction / np.linalg.norm(self.move_direction)
        self.threshold = threshold

        if np.isscalar(bounds):
            bounds = np.array([[-bounds, bounds]] * 3)
        elif isinstance(bounds, np.ndarray):
            if bounds.shape in [(2,), (1, 2)]:
                bounds = np.tile(bounds, (3, 1)).reshape(3, 2)
            elif bounds.shape != (3, 2):
                raise ValueError(f"Invalid bounds shape: {bounds.shape}")
        else:
            raise ValueError(f"Invalid bounds type: {type(bounds)}")
        self._bounds = bounds
        self._start_poses = {name: pose.clone() for name, pose in zip(moveable_sdf.mesh_names, moveable_sdf.mesh_poses)}
        self._query_cache = dict()

    def _eval_fitness(self, x: np.ndarray) -> np.ndarray:
        """
        The fitness function is the absolute of the signed distance between the two SDFs.

        :param x: The displacement of the moveable SDF.
        """
        d, g = self._query(x)
        idx = torch.argmax(d)  # Find closest point, then take abs because it should not be inside
        self._store_gradient(x, g[idx, :3].detach().cpu().numpy(), where="fitness")
        return np.array([torch.abs(d[idx]).item()])

    def _eval_equality_constraints(self, x: np.ndarray) -> Optional[np.ndarray]:
        """
        The translation needs to happen along the specified direction. Also, make sure there is no penetration of
        the two SDFs.
        """
        if self.move_direction is None:
            return None
        x_projected = np.dot(x, self.move_direction) * self.move_direction
        v = x_projected - x

        d, g = self._query(x)
        penetrate = d > 0
        if penetrate.any():
            penetration = np.array((penetrate.sum() / penetrate.numel()).item())
            v_remove = g[:,  :3].mean(dim=0).detach().cpu().numpy()
        else:
            penetration = np.array([0])
            v_remove = np.zeros_like(x)

        grad = -np.hstack([v, v_remove])
        c_eq = np.hstack([np.linalg.norm(v, keepdims=1), penetration])
        self._store_gradient(x, grad, where="eq")
        return c_eq

    def _query(self, x: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Queries the signed distance, except it has already been computed."""
        hash_ = hashlib.sha256(x.tobytes()).hexdigest()
        if hash_ not in self._query_cache:
            pos = torch.tensor(x, dtype=torch.float32)
            self._translate_mesh_to(pos)
            d, g = self.moveable_sdf(self.static_sdf.sph, gradients=True)
            self._query_cache[hash_] = (d, g)
        return self._query_cache[hash_]

    def _translate_mesh_to(self, new_pos: torch.Tensor):
        """Places the mesh at the new position."""
        for name, ref in self._start_poses.items():
            new_pos = ref[:3] + new_pos
            pose = Pose(new_pos, ref[3:])
            self.moveable_sdf.wcm.update_mesh_pose(pose, name=name)

    def get_bounds(self) -> Tuple[List[float], List[float]]:
        return self._bounds[:, 0].tolist(), self._bounds[:, 1].tolist()

    def get_nec(self) -> int:
        return 2 if self.move_direction is not None else 1


class DeformGrasp(MoveToContactPoint):
    
    def __init__(self,
                 finger: CuroboMeshSdf,
                 workpiece: SphereSdf,
                 bounds: Union[float, np.ndarray],
                 weighting: Optional[np.ndarray] = None,
                 *,
                 resolution: int = 60,
                 ):
        super().__init__(finger, workpiece, bounds)
        self.resolution = resolution
        self._base_inertia = self.__get_inertia(workpiece)
        self._gripper_surface_pts = self.__get_surface(finger)[0].shape[0]
        if weighting is None:
            weighting = np.array([1., 1.])
        self.w = weighting

    def _eval_fitness(self, x: np.ndarray) -> np.ndarray:
        f, g = self._query(x)
        wg = (self.w[:, None] * g.detach().cpu().numpy().reshape(-1, x.size)).sum(axis=0)
        self._store_gradient(x, wg, where="fitness")
        return np.dot(f.detach().cpu().numpy(), self.w)

    def _query(self, x: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        hash_ = hashlib.sha256(x.tobytes()).hexdigest()
        if hash_ not in self._query_cache:
            pos = torch.tensor(x, dtype=torch.float32)
            self._translate_mesh_to(pos)
            carved = self.static_sdf.boolean_difference(self.moveable_sdf)
            intersected = self.static_sdf.boolean_intersection(self.moveable_sdf)
            fo, go = self._object_quality(carved, intersected)
            fg, gg = self._grasp_quality(intersected)
            f = -torch.hstack([fo, fg])
            g = torch.hstack([go, gg])
            self._query_cache[hash_] = (f, g)
        return self._query_cache[hash_]

    def _grasp_quality(self, obj: Sdf) -> Tuple[torch.Tensor, torch.Tensor]:
        """Grasp quality increases with the surface area of the object."""
        surface, _ = self.__get_surface(obj)
        _, g = self.moveable_sdf(self.static_sdf.sph, gradients=True)
        contact_points = surface.shape[0]
        quality = torch.tensor([contact_points / self._gripper_surface_pts])
        return quality, g[..., :3].mean(dim=0)

    def _object_quality(self, obj: Sdf, cut: Sdf) -> Tuple[torch.Tensor, torch.Tensor]:
        inertia = self.__get_inertia(obj)
        delta = inertia - self._base_inertia
        c1, c2, c3 = delta[0, 1].abs(), delta[1, 0].abs(), delta[2, 0].abs()
        quality = -sum((c1, c2, c3))
        cutout = cut.discretize(self.static_sdf.get_center(), self.static_sdf.get_dims(), self.resolution)
        d, g = self.static_sdf(cutout.sph, gradients=True)
        grad = g[..., :3].mean(dim=0)  # Pushing out the finger is gonna reduce the inertia offset
        return quality * 5e4, grad

    def __get_inertia(self, obj: Sdf) -> torch.Tensor:
        center = self.static_sdf.get_center()
        dims = self.static_sdf.get_dims()
        return obj.approximate_inertia(center, dims, self.resolution)

    def __get_surface(self, obj: Sdf) -> Tuple[torch.Tensor, torch.Tensor]:
        center = self.moveable_sdf.get_center()
        dims = self.moveable_sdf.get_dims()
        return obj.get_surface_points(center, dims)
    
    def get_nec(self) -> int:
        return 0