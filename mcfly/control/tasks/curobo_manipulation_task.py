from typing import List, Optional, Union

# IsaacSim imports
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.string import find_unique_string_name
from omni.isaac.franka import Franka

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
        """
        super().__init__(name=name)
        self.target_object = None
        self.target_position = None
        self._target_orientation = None
        self.object_grasped: Optional[str] = None
        self.target_object = 'cube'  # Todo: Remove
        self.target_position = np.array([0.55, -0.3, 0.5])  # Todo: Remove

        self._robot = None  # Todo: Set it properly
        self._cube_size = np.array([0.0515, 0.0515, 0.0515])  # TODO: get rid of this

        if static_position_offset is None:
            static_position_offset = [0, 0, 0]
        elif isinstance(static_position_offset, np.array):
            static_position_offset = static_position_offset.tolist()
        self.static_position_offset = static_position_offset

        if static_rotation_offset is None:
            static_rotation_offset = [0, 0, 0]
        elif isinstance(static_rotation_offset, np.array):
            static_rotation_offset = static_rotation_offset.tolist()
        self.static_rotation_offset = static_rotation_offset

    @property
    def target_orientation(self):
        if self._target_orientation is None:
            return np.array([0, 0, 0, 1])
        return self._target_orientation

    def reset(self) -> None:
        """Resets all task related variables."""
        self.target_object = None
        self._target_orientation = None
        self.object_grasped = False
        self.target_object = 'cube'  # Todo: Remove
        self.target_position = np.array([0.55, -0.3, 0.5])  # Todo: Remove

    def get_pick_position(self, observations: dict) -> None:
        """Gets the pick position for the target object, assuming the object position IS the pick position.

        Args:
            observations (dict): World observations.
        """
        assert not self.object_grasped, "There's already an object grasped."
        self.target_position = observations[self.target_object]["position"] + np.array([
            0,
            0,
            self._cube_size[2] / 2 + 0.092,
        ])

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
                "target_position": np.array(  # TODO
                    [
                        self.target_position[0],
                        self.target_position[1],
                        (self._cube_size[2]) + self._cube_size[2] / 2.0,
                    ]
                ),
            }
        return observations

    # TODO: The following code should be replaced

    def set_robot(self, scene):
        franka_prim_path = find_unique_string_name(
            initial_name="/World/Franka", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        franka_robot_name = find_unique_string_name(
            initial_name="my_franka", is_unique_fn=lambda x: not scene.object_exists(x)
        )
        return Franka(
            prim_path=franka_prim_path, name=franka_robot_name, end_effector_prim_name="panda_hand"
        )

    def set_up_scene(self, scene: Scene):
        """Extracts the setup_scene shipped with the CuRobo Stacking controller."""
        super().set_up_scene(scene)
        scene.add_default_ground_plane()
        for i in range(2):
            color = np.random.uniform(size=(3,))
            cube_prim_path = find_unique_string_name(
                initial_name="/World/Cube", is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
            cube_name = find_unique_string_name(
                initial_name="cube", is_unique_fn=lambda x: not scene.object_exists(x)
            )
            cube = scene.add(
                DynamicCuboid(
                    name=cube_name,
                    position=[i / 3, .3, .1],
                    orientation=None,
                    prim_path=cube_prim_path,
                    scale=np.array([0.0515, 0.0515, 0.0515]) / get_stage_units(),
                    size=1.0,
                    color=color,
                )
            )
            self._task_objects[cube_name] = cube
        robot = self.set_robot(scene)
        scene.add(robot)
        self._robot = robot
        self._task_objects[robot.name] = robot
        self._move_task_objects_to_their_frame()
