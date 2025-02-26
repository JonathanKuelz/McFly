import asyncio
from copy import deepcopy
import logging
import sys
sys.path.append('/home/chicken/isaacsim/exts/isaacsim.asset.exporter.urdf/pip_prebundle')  # CuRobo needs these

from curobo.geom.sdf.world import CollisionCheckerType, CollisionQueryBuffer, WorldCollisionConfig
from curobo.geom.sdf.world_mesh import WorldMeshCollision
from curobo.geom.sphere_fit import SphereFitType
from curobo.types.base import TensorDeviceType
from curobo.util.usd_helper import UsdHelper
from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.core.prims import Articulation, SingleRigidPrim
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.util.debug_draw import _debug_draw
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.prims import RigidPrimView
import omni.physx
import torch

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
- Usfeul CuRobo links:
    - https://curobo.org/get_started/2c_world_collision.html
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
    fingers = RigidPrimView(prim_paths_expr='/World/Gripper/*finger', name='Fingers',
                            track_contact_forces=True)  # Todo: Try what happens when I prepare_contact_sensors

    # Requires local import because it will not be loaded succesfully without first enabling extensions
    usd_helper = UsdHelper()
    usd_helper.load_stage(world.stage)
    obstacle = usd_helper.get_obstacles_from_stage(only_substring=['doublepipe'])
    cfg = WorldCollisionConfig(tensor_args=TensorDeviceType(), world_model=obstacle,
                               checker_type=CollisionCheckerType.MESH,
                               cache={'mesh': len(obstacle.mesh)})
    wmc = WorldMeshCollision(cfg)

    def control_callback(dt: float):
        controller.apply_action(action)

    sphere_approximations = approximate_fingers(world, n=2048, requires_grad=False)
    spheres = {f: t.clone().detach() for f, t in sphere_approximations.items()}
    p0 = get_finger_transforms(fingers)

    def force_log_callback(dt: float):
        f = fingers.get_net_contact_forces()
        logging.info(f"Contact forces: {f}")

        if np.linalg.norm(f) > 1e-6:
            # Note that this simple translation does not work anymore as soon as rotations come into play
            p_new = get_finger_transforms(fingers)
            for finger in spheres:
                spheres[finger].requires_grad = False  # A bit hacky, but we only need them for the sdf computation
                spheres[finger][..., :3] = sphere_approximations[finger][..., :3] + (
                    torch.tensor(p_new[finger]['position']).to(sphere_approximations[finger])
                    - torch.tensor(p0[finger]['position']).to(sphere_approximations[finger])
                )
                spheres[finger].requires_grad = True
            sdf_info = get_sdf_per_finger(spheres, wmc)
            debubg_draw_sdf(sdf_info)

    world.add_physics_callback('CloseGripper', control_callback)
    world.add_physics_callback('CuroboSdf', force_log_callback)
    world.add_physics_callback('LogMe', lambda _: logging.warning(f"{action}"))

    asyncio.ensure_future(world.play_async())


def get_finger_transforms(fingers: RigidPrimView) -> dict:
    transforms = dict()
    physx_interface = omni.physx.get_physx_interface()
    for prim in fingers.prims:
        transforms[prim.GetName()] = physx_interface.get_rigidbody_transformation(str(prim.GetPrimPath()))
    return transforms


def approximate_fingers(world: World, n: int = 512, requires_grad: bool = True):
    """Loads the finger meshes and approximates them with n spheres.

    Args:
        world (World): The current simulation world.
        n (int, optional): Number of spheres to use to approximate the fingers. Defaults to 512.
        requires_grad (bool, optional): If this is set, gradients will later be available for the sdf. Defaults to True.
    """
    usd_helper = UsdHelper()
    usd_helper.load_stage(world.stage)
    finger_meshes = usd_helper.get_obstacles_from_stage(only_substring=['finger'])
    spheres = dict()
    for finger in ('left_finger', 'right_finger'):
        query_spheres = list()
        for mesh in finger_meshes.mesh:
            if finger in mesh.name:
                sph = mesh.get_bounding_spheres(n_spheres=n,
                                                fit_type=SphereFitType.VOXEL_SURFACE,
                                                surface_sphere_radius=1e-2)
                query_spheres.extend(sph)
        spheres[finger] = torch.tensor([qs.position + [qs.radius] for qs in query_spheres],
                                       device=TensorDeviceType().device, requires_grad=requires_grad).view(1, 1, -1, 4)
    return spheres


def get_sdf_per_finger(  # Todo: Precompute the query spheres
    bounding_spheres: dict,
    world_mesh_collision: WorldMeshCollision,
):
    """This method parses the current forld for the finger meshes and queries their signed distance against the provided
    worl mesh collision object.

    Args:
        bounding_spheres (dict): A dictionary containing the bounding spheres for each finger.
        world_mesh_collision (WorldMeshCollision): The collision representation to query against.

    Returns:
        dict: A dictionary containing sampled signed distance field for each finger. If gradients are required, the
        dictionary will contain an additional key-value pair for them.
    """
    device = TensorDeviceType().device

    info = {}
    for finger, sph_tensor in bounding_spheres.items():
        query_buffer = CollisionQueryBuffer.initialize_from_shape(sph_tensor.shape, TensorDeviceType(), {'mesh': True})
        d = world_mesh_collision.get_sphere_distance(sph_tensor, query_buffer, weight=torch.tensor([1.], device=device),
                                                     activation_distance=torch.tensor([2e-2], device=device),
                                                     compute_esdf=True)
        info[finger + '_distances'] = d.squeeze()
        info[finger + '_spheres'] = sph_tensor.squeeze()
        if sph_tensor.requires_grad:
            info[finger + '_grad'] = query_buffer.get_gradient_buffer().squeeze()
    return info


def clear_debug_draws():
    """Remove all debug points and lines from the stage."""
    draw = _debug_draw.acquire_debug_draw_interface()
    draw.clear_points()
    draw.clear_lines()


def debubg_draw_sdf(sdf_info, s: float = 15., w: float = 1.):
    draw = _debug_draw.acquire_debug_draw_interface()
    clear_debug_draws()
    for finger in ('left_finger', 'right_finger'):
        dist = sdf_info[finger + '_distances'].detach().cpu().numpy()
        mask = dist > -0.099
        dist = dist[mask]
        pts = sdf_info[finger + '_spheres'][:, :3].detach().cpu().numpy()[mask]
        grads = sdf_info[finger + '_grad'][:, :3].detach().cpu().numpy()[mask]

        c = .3 + .7 * ((dist - dist.min()) / np.max((dist - dist.min()))) ** 8
        colors = np.array([[1., 0., .3, 1.]] * len(pts))
        colors[:, 0] = c
        draw.draw_points(pts, colors, [s] * len(pts))

        endpoints = pts + grads * .1
        draw.draw_lines(pts, endpoints, [(0., 0., 1., 1.)] * len(pts), [w] * len(pts))
    logging.info("Drawing SDF done")


def reset_articulation(world: World):
    scene = world.scene
    gripper = scene.get_object('Gripper')
    if gripper.get_applied_actions() is None:
        return  # Nothing to reset
    controller = ArticulationController()
    controller.initialize(gripper)
    action = ArticulationAction(joint_positions=gripper._default_joints_state.positions[0])
    controller.apply_action(action)


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

    def cleanup(self):
        reset_articulation(self._world)
        clear_debug_draws()
