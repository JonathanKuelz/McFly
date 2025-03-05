import hashlib
from typing import List, Optional, Tuple, Union

from curobo.types.math import Pose
import numpy as np
import torch

from biff.optimization.pygmo import UdpWithGradient
from mcfly.utilities.sdf import CuroboMeshSdf, SphereSdf

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
            self._translate_mesh(pos)
            d, g = self.moveable_sdf(self.static_sdf.sph, gradients=True)
            self._query_cache[hash_] = (d, g)
        return self._query_cache[hash_]

    def _translate_mesh(self, new_pos: torch.Tensor):
        """Places the mesh at the new position."""
        for name, ref in self._start_poses.items():
            new_pos = ref[:3] + new_pos
            pose = Pose(new_pos, ref[3:])
            self.moveable_sdf.wcm.update_mesh_pose(pose, name=name)

    def get_bounds(self) -> Tuple[List[float], List[float]]:
        return self._bounds[:, 0].tolist(), self._bounds[:, 1].tolist()

    def get_nec(self) -> int:
        return 2 if self.move_direction is not None else 1

    # def generate(self) -> UdpWithGradient:
    #     """
    #     pygmo requires deepcopy-able objects. This class is not, due to the CuRobo SDF references. This method
    #     returns a deepcopy-able functional copy of this class.
    #     """
    #     this = self
    #     class _UDP(UdpWithGradient):
    #
    #         def _eval_fitness(self, x: np.ndarray) -> np.ndarray:
    #             return this._eval_fitness(x)
    #
    #         def _eval_equality_constraints(self, x: np.ndarray) -> Optional[np.ndarray]:
    #             return this._eval_equality_constraints(x)
    #
    #         def get_bounds(self) -> Tuple[List[float], List[float]]:
    #             return this.get_bounds()
    #
    #         def get_nec(self) -> int:
    #             return this.get_nec()
    #
    #     return _UDP()
