import logging

from isaacsim import SimulationApp
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
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.string import find_unique_string_name
from omni.isaac.franka import Franka

# OmniKit imports
from omni.isaac.core.utils.extensions import enable_extension

from mcfly.control.curobo_moveto import MoveToController
from mcfly.control.tasks.curobo_reach_task import ReachTask


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


class ControlFrankaTask(ReachTask):
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
        prim_path = find_unique_string_name(initial_name=f'/World/{self.goal}',
                                            is_unique_fn=lambda x: not is_prim_path_valid(x))
        self.goal_prim = SingleXFormPrim(name=self.goal,
                                         prim_path=prim_path)
        scene.add(self.goal_prim)

        wall = FixedCuboid(name='wall', prim_path='/World/wall', position=[0.2, 0.25, 0.2], scale=[0.6, 0.05, 0.4])
        scene.add(wall)

        robot = self.set_robot(scene)
        self._robot = robot
        scene.add(robot)


def main(sim: SimulationApp):
    robot_name = 'my_franka'
    robot_prim_path = "/World/Franka/panda_link0"
    my_world = World(stage_units_in_meters=1.0)
    wait_steps = 8
    ignore_substring = ["Franka", "TargetCube", "material", "Plane"]
    stage = my_world.stage
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    my_task = ControlFrankaTask(name='Learning to Manipulate')
    my_world.add_task(my_task)
    my_world.reset()

    my_franka = my_world.scene.get_object(robot_name)
    franka_config = join_path(get_robot_configs_path(), "franka.yml")
    cmd_joint_names = my_franka.dof_names[:-2]
    my_controller = MoveToController(
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
    initial_steps = 100

    my_franka.gripper.open()
    for _ in range(wait_steps):
        my_world.step(render=True)
    my_task.reset()
    observations = my_world.get_observations()
    my_task.set_goal(observations)
    add_extensions(sim)

    i = 0
    while sim.is_running():
        my_world.step(render=True)  # necessary to visualize changes
        my_task.set_goal(observations)
        i += 1

        if i < initial_steps:
            continue

        if not my_controller.is_initialized:
            my_controller.reset(ignore_substring, robot_prim_path)

        observations = my_world.get_observations()
        sim_js = my_franka.get_joints_state()
        art_action = my_controller.forward(sim_js, cmd_joint_names)
        if art_action is not None:
            articulation_controller.apply_action(art_action)

    sim.close()


if __name__ == '__main__':
    main(_sim)
