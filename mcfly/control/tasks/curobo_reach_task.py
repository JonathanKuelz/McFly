# IsaacSim imports
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.string import find_unique_string_name
from omni.isaac.franka import Franka

import numpy as np


class ReachTask(BaseTask):
    # TODO: It could be cleaner to implement this as mg.MotionPolicyController
    """Boilerplate code for manipulating an object with a single robot in IsaacSim."""

    def __init__(self, name: str):
        """The task can be setup with just a name.

        Args:
            name (str): A unique name for this task.
        """
        super().__init__(name=name)
        self.goal: str = 'GoalPose'
        self.target_position = np.array([.3, 0, .5])
        self.target_orientation = np.array([0, 0, 0, 1])
        self.goal_prim = None
        self._robot = None

    def reset(self) -> None:
        """Resets all task related variables."""
        pass

    def set_goal(self, observations: dict) -> None:
        """Gets the pick position for the target object, assuming the object position IS the pick position.

        Args:
            observations (dict): World observations.
        """
        self.target_position = observations[self.goal]["position"]
        self.target_orientation = observations[self.goal]["orientation"]

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
        pos, ori = self.goal_prim.get_world_pose()
        observations[self.goal] = {
            "position": pos,
            "orientation": ori,
        }
        return observations

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
        self.prim_path = find_unique_string_name(initial_name=f'/World/{self.goal}',
                                                 is_unique_fn=lambda x: not is_prim_path_valid(x))
        self.goal_prim = SingleXFormPrim(name=self.goal,
                                         prim_path=self.prim_path)
        self.goal_prim.set_world_pose(position=self.target_position, orientation=self.target_orientation)
        scene.add(self.goal_prim)

        wall = FixedCuboid(name='wall', prim_path='/World/wall', position=[0.2, 0.25, 0.2], scale=[0.6, 0.05, 0.4])
        scene.add(wall)

        robot = self.set_robot(scene)
        self._robot = robot
        scene.add(robot)
