import logging
from pathlib import Path
from typing import List, Optional, Union

# CuRobo imports
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.util.usd_helper import UsdHelper
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util_file import load_yaml
import curobo.wrap.reacher.motion_gen as cmg

# IsaacSim imports
from isaacsim.core.api import World
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.api.robots import Robot

import numpy as np
from omni.isaac.core.utils.types import ArticulationAction

from mcfly.control.curobo_moveto import MoveToController


class ManipulationController(MoveToController):
    """A controller inspired from the CuRobo stacking example but more general. It allows manipulating objects in the
    scene.
    """

    def __init__(
        self,
        world: World,
        task: BaseTask,
        robot: Robot,
        robot_config_path: Union[str, Path],
        name: str = "manipulation_controller",
        cmd_joint_names: Optional[List[str]] = None,
        constrain_grasp_approach: bool = False,
    ) -> None:
        """Initializes the controller.

        Args:
            world (World): The current world.
            task (BaseTask): The manipulation task.
            robot (Robot): The robot to control.
            robot_config_path (Union[str, Path]): The path to the CuRobo robot configuration file.
            name (str, optional): The name to register this controller under. Defaults to "curobo_controller".
            cmd_joint_names (List[str], optional): The joint names to use for the commands. If not provided, this will
                default to all dof names of the robot.
            constrain_grasp_approach (bool, optional): _description_. Defaults to False. // TODO: what does this do?
        """
        super().__init__(name=name)
        self._save_log = False
        self.world = world
        self.task = task

        self.robot = robot
        self.robot_name: str = robot.name  # TODO: this can be a property

        self._step_idx = 0
        self._control_freq = 3  # Every 3rd step, we send an AirtuculationAction

        # warmup curobo instance
        self.usd_helper = UsdHelper()
        self.usd_helper.load_stage(self.world.stage)
        self.init_curobo = False

        if cmd_joint_names is None:
            cmd_joint_names = robot.dof_names
        self.cmd_joint_names = cmd_joint_names
        self.tensor_args = TensorDeviceType()  # TODO: this defaults to cuda:0
        self.robot_cfg = load_yaml(robot_config_path)["robot_cfg"]
        self.robot_cfg["kinematics"]["collision_spheres"] = "spheres/franka_collision_mesh.yml"

        self._world_cfg = self.usd_helper.get_obstacles_from_stage(ignore_substring=[self.robot.prim_path])

        motion_gen_config = cmg.MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            self._world_cfg,
            self.tensor_args,
            trajopt_tsteps=32,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=True,  # TODO: Think about when this needs to be deactivated (see docs for hint)
            interpolation_dt=0.01,
            collision_cache={"obb": 10, "mesh": 4},  # TODO: come up with reasonable values
            store_ik_debug=self._save_log,
            store_trajopt_debug=self._save_log,
        )
        self.motion_gen = cmg.MotionGen(motion_gen_config)

        logging.info("CuRobo Controller warming up...")
        self.motion_gen.warmup(parallel_finetune=True)
        pose_metric = None  # TODO: cmg.PoseCostMetric seems to be useful

        self.plan_config = cmg.MotionGenPlanConfig(
            enable_graph=False,  # TODO: why?
            max_attempts=10,
            enable_graph_attempt=None,
            enable_finetune_trajopt=True,
            partial_ik_opt=False,
            parallel_finetune=True,
            pose_cost_metric=pose_metric,
            time_dilation_factor=0.75,
        )
        self.cmd_plan = None
        self.cmd_idx = 0
        self._step_idx = 0
        self.idx_list = None

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

    def detach_obj(self) -> None:
        """Detach the currently held object."""
        self.motion_gen.detach_object_from_robot()

    def plan(
        self,
        ee_translation_goal: np.array,
        ee_orientation_goal: np.array,
        joint_state: JointState,
        joint_names: list,
    ) -> cmg.MotionGenResult:
        """Plans a motion to reach the given goal.

        Usually, this doesn't need to be called directly, as the `forward` method will call this internally.
        Args:
            ee_translation_goal (np.array): The desired end effector translation.
            ee_orientation_goal (np.array): The desired end effector orientation as quaternion.
            joint_state (JointState): The current joint state of the robot.
            joint_names (list): The names of the joints corresponding to the joint state.
        """
        ik_goal = Pose(
            position=self.tensor_args.to_device(ee_translation_goal),
            quaternion=self.tensor_args.to_device(ee_orientation_goal),
        )
        cu_js = JointState(
            position=self.tensor_args.to_device(joint_state.positions),
            velocity=self.tensor_args.to_device(joint_state.velocities) * 0.0,
            acceleration=self.tensor_args.to_device(joint_state.velocities) * 0.0,
            jerk=self.tensor_args.to_device(joint_state.velocities) * 0.0,
            joint_names=joint_names,
        )
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)
        result = self.motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, self.plan_config.clone())
        if self._save_log:  # and not result.success.item(): # logging for debugging
            UsdHelper.write_motion_gen_log(
                result,
                {"robot_cfg": self.robot_cfg},
                self._world_cfg,
                cu_js,
                ik_goal,
                "home/chicken/curobo_debug",  # TODO: replace by properly configured constant
                write_ik=False,
                write_trajopt=True,
                visualize_robot_spheres=True,
                link_spheres=self.motion_gen.kinematics.kinematics_config.link_spheres,
                grid_space=2,
                write_robot_usd_path="log/usd/assets",
            )
        return result

    def forward(
        self,
        joint_state: JointState,
        joint_names: list,
    ) -> ArticulationAction:
        """Computes the next joint articulation.

        Args:
            joint_state (JointState): Current joint state.
            joint_names (list): Joint names corresponding to the joint state.

        Returns:
            ArticulationAction: The next joint articulation.
        """
        assert self.task.target_position is not None
        assert self.task.target_object is not None

        if self.cmd_plan is None:
            self.cmd_idx = 0
            self._step_idx = 0
            # Set EE goals
            ee_translation_goal = self.task.target_position
            ee_orientation_goal = self.task.target_orientation
            # compute curobo solution:
            result = self.plan(ee_translation_goal, ee_orientation_goal, joint_state, joint_names)
            succ = result.success.item()
            if succ:
                cmd_plan = result.get_interpolated_plan()
                self.idx_list = [i for i in range(len(self.cmd_joint_names))]
                self.cmd_plan = cmd_plan.get_ordered_joint_state(self.cmd_joint_names)
            else:
                logging.warning(f"CuRobo planning did not converge to a solution. {result.status}")
                return None

        if self._step_idx % 3 == self._control_freq:
            cmd_state = self.cmd_plan[self.cmd_idx]
            self.cmd_idx += 1

            # get full dof state
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy() * 0.0,
                joint_indices=self.idx_list,
            )
            if self.cmd_idx >= len(self.cmd_plan.position):
                self.cmd_idx = 0
                self.cmd_plan = None
        else:
            art_action = None
        self._step_idx += 1
        return art_action

    def reached_target(self, observations: dict) -> bool:
        """Boolean function to check if the target has been reached.

        Args:
            observations (dict): Current world observations. The robot needs to be in there.

        Returns:
            bool: True if the target has been reached, False otherwise.
        """
        curr_ee_position = observations[self.robot_name]["end_effector_position"]  # TODO: is end_effector_position in there?
        if np.linalg.norm(self.task.target_position - curr_ee_position) < 0.04 \
           and (self.cmd_plan is None):  # TODO: replace by non-hardcoded value
            return True
        else:
            return False

    def reset(
        self,
        ignore_substring: str,
        robot_prim_path: str,
    ) -> None:
        """Resets the controller by updating the world configuration and resetting the plan.

        Args:
            ignore_substring (str): All prims with this substring in their path will be ignored. Case sensitive.
            robot_prim_path (str): The root path of the robot in the USD stage.
        """
        self.update(ignore_substring, robot_prim_path)
        self.init_curobo = True
        self.cmd_plan = None
        self.cmd_idx = 0

    def update(
        self,
        ignore_substring: str,
        robot_prim_path: str,
    ) -> None:
        """Updates the world configuration.

        Args:
            ignore_substring (str): All prims with this substring in their path will be ignored. Case sensitive.
            robot_prim_path (str): The root path of the robot in the USD stage.
        """
        obstacles = self.usd_helper.get_obstacles_from_stage(
            ignore_substring=ignore_substring, reference_prim_path=robot_prim_path
        ).get_collision_check_world()
        # add ground plane as it's not readable:
        obstacles.add_obstacle(self._world_cfg.cuboid[0])  # TODO - this is uncool
        self.motion_gen.update_world(obstacles)
        self._world_cfg = obstacles
