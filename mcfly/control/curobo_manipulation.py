from typing import List

# CuRobo imports
from curobo.types.math import Pose
from curobo.types.state import JointState

from mcfly.control.curobo_moveto import MoveToController


class ManipulationController(MoveToController):
    """A controller inspired from the CuRobo stacking example but more general. It allows manipulating objects in the
    scene.
    """

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
        cu_js = JointState(
            position=self.tensor_args.to_device(joint_state.positions),
            velocity=self.tensor_args.to_device(joint_state.velocities) * 0.0,
            acceleration=self.tensor_args.to_device(joint_state.velocities) * 0.0,
            jerk=self.tensor_args.to_device(joint_state.velocities) * 0.0,
            joint_names=joint_names,
        )
        self.motion_gen.attach_objects_to_robot(
            cu_js,
            [obj_name],
            world_objects_pose_offset=Pose.from_list([0, 0, z_offset, 1, 0, 0, 0], self.tensor_args),
        )
        self.tasl.object_grasped = True

    def detach_obj(self) -> None:
        """Detach the currently held object."""
        self.motion_gen.detach_object_from_robot()
        self.tasl.object_grasped = False
