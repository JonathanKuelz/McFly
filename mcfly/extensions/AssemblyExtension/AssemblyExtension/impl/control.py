import asyncio
import logging

from pxr import Sdf, UsdPhysics
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.api import World
import omni.usd

from mcfly.control.rmpflow import FrankaRMPFlowController


class AssemblyControl:

    def __init__(self):
        self._goal_prim = None
        self._world = None

    @property
    def scene(self) -> Scene:
        return self.world.scene

    @property
    def world(self):
        if self._world is None:
            self._world = World(**self._world_settings)
            self._world.reset()
        return self._world

    def _setup_post_load(self):
        robot_name = 'Franka'  # TODO: Remove hardcoded value
        robot = self.scene.get_object(robot_name)  # TODO: Why is the robot not found?
        logging.warning(f"Robot: {robot}")
        return
        self._controller = FrankaRMPFlowController(name="target_follower_controller", robot_articulation=robot)
        self._articulation_controller = robot.get_articulation_controller()

    def _on_set_goal_prim(self, goal: Sdf.Path):
        """Function called when clicking set goal prim button"""
        logging.info(f"Goal set to prim {goal}")
        self._goal_prim = goal

    def _on_move(self, val):
        async def _on_move_async(val):
            if val:
                await self.world.play_async()
                self.world.add_physics_callback("sim_step", self._on_move_simulation_step)
            else:
                self.world.remove_physics_callback("sim_step")
        asyncio.ensure_future(_on_move_async(val))

    def _on_move_simulation_step(self, step_size):
        observations = self.world.get_observations()
        logging.warning(f"Observations: {observations}")
        actions = self._controller.forward(
            target_end_effector_position=observations[self._goal_prim]["position"],
            target_end_effector_orientation=observations[self._goal_prim]["orientation"],
        )
        self._articulation_controller.apply_action(actions)
