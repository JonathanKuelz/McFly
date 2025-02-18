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


def main(sim: SimulationApp):
    robot_name = 'my_franka'
    robot_prim_path = "/World/Franka/panda_link0"
    my_world = World(stage_units_in_meters=1.0)
    wait_steps = 8
    ignore_substring = ["Franka", "TargetCube", "material", "Plane"]
    stage = my_world.stage
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    my_task = ReachTask(name='Learning to Manipulate')
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
    ################################################################
    print("Start simulation...")
    robot = my_franka
    print(
        my_world._physics_context.get_solver_type(),
        robot.get_solver_position_iteration_count(),
        robot.get_solver_velocity_iteration_count(),
    )
    print(my_world._physics_context.use_gpu_pipeline)
    print(articulation_controller.get_gains())
    print(articulation_controller.get_max_efforts())
    robot = my_franka
    print("**********************")

    my_franka.gripper.open()
    for _ in range(wait_steps):
        my_world.step(render=True)
    my_task.reset()
    task_finished = False
    observations = my_world.get_observations()
    my_task.set_goal(observations)

    i = 0

    add_extensions(sim)

    while sim.is_running():
        my_world.step(render=True)  # necessary to visualize changes
        my_task.set_goal(observations)
        i += 1

        if task_finished or i < initial_steps:
            continue

        if not my_controller.init_curobo:
            my_controller.reset(ignore_substring, robot_prim_path)

        observations = my_world.get_observations()
        sim_js = my_franka.get_joints_state()
        art_action = my_controller.forward(sim_js, cmd_joint_names)
        if art_action is not None:
            articulation_controller.apply_action(art_action)

    sim.close()


if __name__ == '__main__':
    main(_sim)
