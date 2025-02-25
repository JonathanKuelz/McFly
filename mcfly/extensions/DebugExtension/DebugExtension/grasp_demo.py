import asyncio
import logging

from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.core.prims import Articulation, SingleRigidPrim
from isaacsim.core.utils.types import ArticulationAction

import numpy as np

from omni.isaac.core import World

from mcfly.utilities.debugging import DebugInterfaceBaseClass


"""
Learnings:

- Only rigid bodies can be connected by joints.
- Rigid bodies behave weirdly (no mass etc.) if no collider is attached.
- Order in which things need to be done in the stage:
    1. Add the rigid body
    2. Add the joints
    3. Add drives to the joints
    4. Create an articulation root
"""

GRIPPER_PRIM = '/World/Gripper'
OBJECT_PRIM = '/World/doublepipe'


def setup_scene(world: World):
    """Gets the gripper and gripping object from the stage and registers them."""
    scene = world.scene
    articulation = Articulation(
        prim_paths_expr=GRIPPER_PRIM,
        name='Gripper',
    )
    scene.add(articulation)
    obj = SingleRigidPrim(
        OBJECT_PRIM,
        name='Object'
    )
    scene.add(obj)

    # Don't ask me why, but the pause only works if done asynchronously
    async def setup_():
        await world.reset_async()
        await world.pause_async()

    asyncio.ensure_future(setup_())


def close_gripper(world: World):
    scene = world.scene
    gripper = scene.get_object('Gripper')
    gripper.initialize()
    controller = ArticulationController()
    controller.initialize(gripper)
    action = ArticulationAction(joint_positions=np.array([0.25, 0.25]), joint_indices=np.array([0, 1]))

    def callback(dt: float):
        controller.apply_action(action)

    world.add_physics_callback('CloseGripper', callback)
    world.add_physics_callback('LogMe', lambda _: logging.warning(f"{action}"))

    asyncio.ensure_future(world.play_async())


def main():
    setup_scene()
    close_gripper()


class DebugInterface(DebugInterfaceBaseClass):
    """_summary_

    Args:
        DebugInterfaceBaseClass (_type_): _description_
    """

    def __init__(self, world: World):
        super().__init__()
        self._world = world
        self._world.get_physics_context().set_gravity(1e-6)

    def setup_scene(self):
        setup_scene(self._world)

    def execute(self):
        close_gripper(self._world)
