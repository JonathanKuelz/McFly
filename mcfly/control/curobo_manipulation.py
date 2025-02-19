import logging
from typing import List, Optional

# CuRobo imports
from curobo.types.math import Pose
from curobo.types.state import JointState
import curobo.wrap.reacher.motion_gen as cmg

from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.world import World

from mcfly.control.curobo_moveto import MoveToController


class ManipulationController(MoveToController):
    """A controller inspired from the CuRobo stacking example but more general. It allows manipulating objects in the
    scene.
    """

    gripper_action_steps: int = 20  # Number of world steps to perform to open/close the gripper.

    def attach_obj(
        self,
        joint_state: JointState,
        joint_names: List[str],
        obj_name: str,
        z_offset: float = 0.01,
    ) -> None:
        """Attaches an object to the robot.

        Args:
            current_js (JointState): The current joint state of the robot under control.
            joint_names (List[str]): The name of the joints corresponding to the joint state.
            obj_name (str): The name of the object to attach.
            z_offset (float, optional): The z offset to apply to the object before being attached. Defaults to 0.01.
                This should avoid collisions immediately after attaching the object.
        """
        world = World()
        scene = world.scene

        self.robot.gripper.close()
        for _ in range(self.gripper_action_steps):
            world.step(render=True)

        obj_prim = scene.get_object(obj_name).prim_path
        cu_js = JointState(
            position=self.tensor_args.to_device(joint_state.positions),
            velocity=self.tensor_args.to_device(joint_state.velocities) * 0.0,
            acceleration=self.tensor_args.to_device(joint_state.velocities) * 0.0,
            jerk=self.tensor_args.to_device(joint_state.velocities) * 0.0,
            joint_names=joint_names,
        )
        self.motion_gen.attach_objects_to_robot(
            cu_js,
            [obj_prim],
            world_objects_pose_offset=Pose.from_list([0, 0, z_offset, 1, 0, 0, 0], self.tensor_args),
        )
        self.task.object_grasped = True

    def detach_obj(self) -> None:
        """Detach the currently held object."""
        self.robot.gripper.open()
        for _ in range(self.gripper_action_steps):
            World().step(render=True)

        self.motion_gen.detach_object_from_robot()
        self.task.reset()

    def forward(
        self,
        joint_state: JointState,
        joint_names: list,
        force_replan: bool = False,
        observations: dict = None
    ) -> Optional[ArticulationAction]:
        """This calls the forward function of the moveto controller. In addition, it can force replanning if the last
            motion plan is terminated, but did not lead to termination of the task.

        Args:
            observations (dict): The current world observations.
        """
        if not force_replan:
            force_replan = self.cmd_plan is None \
                and not self.reached_target(observations) \
                and self.task.has_goal
            if force_replan:
                logging.warning('Replanning manipulation motion, last plan did not terminate successfully.')
        return super().forward(joint_state, joint_names, force_replan)

    def plan(
        self,
        goal_pose: Pose,
        joint_state: JointState,
        joint_names: list,
    ) -> cmg.MotionGenResult:
        if not self.task.has_goal:
            logging.warning('Skipping plan step -- no goal set.')
            return cmg.MotionGenResult(success=self.tensor_args.to_device([False]),
                                       status=cmg.MotionGenStatus.NOT_ATTEMPTED)
        return super().plan(goal_pose, joint_state, joint_names)
