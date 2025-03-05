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
        self.move_direction = move_direction.detach().cpu().numpy() if move_direction is not None else None
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

    def _eval_fitness(self, x: np.ndarray) -> np.ndarray:
        """
        The fitness function is the absolute of the signed distance between the two SDFs.

        :param x: The displacement of the moveable SDF.
        """
        pos = torch.tensor(x, dtype=torch.float32)
        self._translate_mesh(pos)
        d, g = self.moveable_sdf(self.static_sdf.sph, gradients=True)
        idx = torch.argmin(torch.abs(d))

        self._store_gradient(x, g[idx, :3].detach().cpu().numpy())
        return np.array([torch.abs(d[idx]).item()])

    def _eval_equality_constraints(self, x: np.ndarray) -> Optional[np.ndarray]:
        """
        The translation needs to happen along the specified direction.
        """
        if self.move_direction is None:
            return None
        delta = x - np.dot(x, self.move_direction)
        return np.linalg.norm(delta, keepdims=1)

    def _translate_mesh(self, new_pos: torch.Tensor):
        """Places the mesh at the new position."""
        for name, ref in self._start_poses.items():
            new_pos = ref[:3] + new_pos
            pose = Pose(new_pos, ref[3:])
            self.moveable_sdf.wcm.update_mesh_pose(pose, name=name)

    def get_bounds(self) -> Tuple[List[float], List[float]]:
        return self._bounds[:, 0].tolist(), self._bounds[:, 1].tolist()

    def get_nec(self) -> int:
        return 1 if self.move_direction is not None else 0

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
