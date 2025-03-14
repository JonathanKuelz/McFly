import logging

from isaacsim import SimulationApp
import numpy as np
_sim = None
if __name__ == '__main__':
    _sim = SimulationApp(
        {
            "headless": False,
            "width": "1920",
            "height": "1080",
        }
    )

# CuRobo imports
from curobo.util_file import get_robot_configs_path, join_path

# IsaacSim imports
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.string import find_unique_string_name
from omni.isaac.franka import Franka

# OmniKit imports
from omni.isaac.core.utils.extensions import enable_extension

from mcfly.control.curobo_manipulation import ManipulationController
from mcfly.control.tasks.curobo_manipulation_task import ManipulationTask


def add_extensions(sim: SimulationApp, headless_mode: bool = False):
    ext_list = [
        "omni.kit.asset_converter",
        "omni.kit.tool.asset_importer",
        "omni.isaac.asset_browser",
    ]
    if headless_mode:
        logging.warning("Running in headless mode: " + headless_mode)
        ext_list += ["omni.kit.livestream." + headless_mode]
    [enable_extension(x) for x in ext_list]
    sim.update()


class ControlFrankaTask(ManipulationTask):
    """This class presents a sample implementation of a reach task.
    """

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

    def set_up_scene(self, scene):
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
                    position=[.3, i / 3, .1],
                    orientation=[1, 0, 0, 0],
                    prim_path=cube_prim_path,
                    scale=np.array([0.0515, 0.0515, 0.0515]),
                    size=1.0,
                    color=color,
                )
            )
            self._task_objects[cube_name] = cube
        robot = self.set_robot(scene)
        scene.add(robot)
        self._robot = robot
        self._task_objects[robot.name] = robot
        self.drop_position = np.array([0.5, 0., 0.2])
        self.drop_orientation = np.array([0., 1., 0., 0.])


def main(sim: SimulationApp):
    robot_name = 'my_franka'
    robot_prim_path = "/World/Franka/panda_link0"
    my_world = World(stage_units_in_meters=1.0)
    wait_steps = 8
    ignore_substring = ["Franka", "TargetCube", "material", "Plane"]
    stage = my_world.stage
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    my_task = ControlFrankaTask(name='Learning to Manipulate',
                                static_position_offset=np.array([0.0, 0.0, 0.1]),
                                static_rotation_offset=np.array([0.0, 1.0, 0.0, 0.0])
                                )
    my_world.add_task(my_task)
    my_world.reset()
    cubes = my_task.object_names

    my_franka = my_world.scene.get_object(robot_name)
    franka_config = join_path(get_robot_configs_path(), "franka.yml")
    cmd_joint_names = my_franka.dof_names[:-2]
    my_controller = ManipulationController(
        world=my_world,
        task=my_task,
        robot=my_franka,
        cmd_joint_names=cmd_joint_names,
        robot_config_path=franka_config,
        name='FrankaController')

    articulation_controller = my_franka.get_articulation_controller()

    my_franka.set_solver_velocity_iteration_count(4)
    my_franka.set_solver_position_iteration_count(124)
    my_world._physics_context.set_solver_type("TGS")
    initial_steps = 240

    my_franka.gripper.open()
    for _ in range(wait_steps):
        my_world.step(render=True)
    my_task.reset()
    my_task.set_target_object(cubes[0])
    observations = my_world.get_observations()
    my_task.update(observations)
    add_extensions(sim)

    i = 0
    first_cube_placed = False
    while sim.is_running():
        my_world.step(render=True)  # necessary to visualize changes
        i += 1

        if i < initial_steps:
            continue

        if not my_controller.is_initialized:
            my_controller.reset(ignore_substring, robot_prim_path)

        observations = my_world.get_observations()
        my_task.update(observations)

        sim_js = my_franka.get_joints_state()
        art_action = my_controller.forward(sim_js, cmd_joint_names, observations=observations)
        if art_action is not None:
            articulation_controller.apply_action(art_action)

        if my_controller.reached_target(observations):
            if my_task.object_grasped:
                my_controller.detach_obj()
                if not first_cube_placed:
                    my_task.drop_position = my_task.drop_position + np.array([0., 0., .07])
                first_cube_placed = True
                my_task.set_target_object(cubes[1])
            else:
                grasp_obj = cubes[1] if first_cube_placed else cubes[0]
                my_controller.attach_obj(sim_js, cmd_joint_names, grasp_obj)
                my_task.set_target_object(grasp_obj)

    sim.close()


if __name__ == '__main__':
    main(_sim)
