from abc import abstractmethod
from typing import List, Optional, Union

# CuRobo imports
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose

# IsaacSim imports
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.api.scenes.scene import Scene

import numpy as np


class ManipulationTask(BaseTask):
    """Boilerplate code for manipulating an object with a single robot in IsaacSim."""

    def __init__(self,
                 name: str,
                 static_position_offset: Optional[Union[np.array, List[float]]] = None,
                 static_rotation_offset: Optional[Union[np.array, List[float]]] = None,
                 ):
        """The task can be setup with just a name.

        Args:
            name (str): A unique name for this task.
            static_position_offset (Optional[Union[np.array, List[float]]], optional): The static position offset to
                apply to the target object to obtain a grasp pose. Defaults to None.
            static_rotation_offset (Optional[Union[np.array, List[float]]], optional): The static rotation offset to
                apply to the target object to obtain a grasp pose. Defaults to None.
        """
        super().__init__(name=name)
        self.object_grasped = False
        self.target_object_name = None
        self.tensor_args = TensorDeviceType()

        if static_position_offset is None:
            static_position_offset = np.array([0., 0., 0.])
        elif not isinstance(static_position_offset, np.ndarray):
            static_position_offset = np.array(static_position_offset)
        self.static_position_offset: np.array = static_position_offset

        if static_rotation_offset is None:
            static_rotation_offset = np.array([1., 0., 0., 0.])
        elif not isinstance(static_rotation_offset, np.ndarray):
            static_rotation_offset = np.array(static_rotation_offset)
        self.static_rotation_offset: np.array = static_rotation_offset

        self.drop_position = np.array([0., 0., 0.])
        self.drop_orientation = np.array([1., 0., 0., 0.])

    @property
    def offset(self) -> Pose:
        return Pose.from_list(
            self.static_position_offset.tolist() + self.static_rotation_offset.tolist(),
            self.tensor_args
        )

    def get_observations(self) -> dict:
        """Make sure all objects that need to be kept track off are in the observations.

        Returns:
            dict: The current observations.
        """
        joints_state = self._robot.get_joints_state()
        end_effector_position, _ = self._robot.end_effector.get_local_pose()
        observations = {
            self._robot.name: {
                "joint_positions": joints_state.positions,
                "end_effector_position": end_effector_position,
            }
        }
        for name, task_object in self._task_objects.items():
            if name == self._robot.name:
                continue
            pos, ori = task_object.get_local_pose()
            observations[name] = {
                "position": pos,
                "orientation": ori,
            }
        return observations

    def reset(self) -> None:
        """Resets all task related variables."""
        self.object_grasped = False
        self.target_position = np.array([0., 0., 0.])
        self.target_orientation = np.array([1., 0., 0., 0.])

    def set_goal(self, observations: dict) -> None:
        """Gets the pick position for the target object, assuming the object position IS the pick position.

        Args:
            observations (dict): World observations.
        """
        if self.object_grasped:
            self.target_position = self.drop_position
            self.target_orientation = self.drop_orientation
        else:
            pos = observations[self.target_object]["position"]
            ori = observations[self.target_object]["orientation"]
            pose = Pose.from_list(pos.tolist() + ori.tolist(), self.tensor_args)
            goal = pose.multiply(self.offset)
            self.target_position = goal.position.detach().cpu().numpy()
            self.target_orientation = goal.quaternion.detach().cpu().numpy()

    @abstractmethod
    def set_up_scene(self, scene: Scene):
        """Set up the scene for the task. This method needs to populate the following properties:
            - self._robot: The robot to be used for this task.
            - self._task_objects: The objects to be manipulated.
            - self.target_object: The current object to be manipulated.
            - self.drop_position: The current position to drop the object.
            - self.drop_orientation: The current orientation to drop the object.

        Args:
            scene (Scene): The current scene.

        """
        return super().set_up_scene(scene)
