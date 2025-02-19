from abc import abstractmethod

# IsaacSim imports
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.api.scenes.scene import Scene

import numpy as np


class ReachTask(BaseTask):
    """Boilerplate code for moving to a goal with a single robot in IsaacSim."""

    def __init__(self, name: str):
        """The task can be setup with just a name.

        Args:
            name (str): A unique name for this task.
        """
        super().__init__(name=name)
        self.goal: str = 'GoalPose'
        self.target_position = np.array([0., 0., 0.])
        self.target_orientation = np.array([1., 0., 0., 0.])
        self.goal_prim = None
        self._robot = None

    def reset(self) -> None:
        """Resets all task related variables."""
        pass

    def set_goal(self, observations: dict) -> None:
        """Gets the position if the target object.

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

    @abstractmethod
    def set_up_scene(self, scene: Scene):
        """Set up the scene for the task. This method needs to populate the following properties:
            - self.goal_prim: The goal primitive.
            - self._robot: The robot to be used for this task.

        Args:
            scene (Scene): The current scene.

        """
        return super().set_up_scene(scene)
