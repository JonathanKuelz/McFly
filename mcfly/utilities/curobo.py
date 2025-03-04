from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from curobo.geom.sdf.world import CollisionQueryBuffer, WorldCollision
from curobo.geom.sdf.world_mesh import WorldMeshCollision
from curobo.geom.types import Mesh, Sphere
from curobo.types.base import TensorDeviceType
import torch
import trimesh


@dataclass
class SdfInfo:
    distances: torch.Tensor
    spheres: torch.Tensor
    grad: Optional[torch.Tensor] = None

def get_sdf(
    query_spheres: Dict[str, torch.Tensor],
    world_collision: WorldCollision,
    tensor_device_type: Optional[TensorDeviceType] = None,
    *,
    collision_types: Iterable[str] = ('mesh',),
    activation_distance: float = 0.,
) -> Dict[str, SdfInfo]:
    """Computes an sdf between arbitrary meshes approximated by spheres and a world collision representation.

    Args:
        query_spheres (dict): A dictionary containing object names and spheres representing them.
        world_collision (WorldCollision): The collision representation to query against.
        tensor_device_type (TensorDeviceType): The device type for tensors. If None, defaults to TensorDeviceType.
        collision_types (Iterable[str]): The types of collisions objects to consider in the world mesh collision object.
        activation_distance (float): All distances above this value will be ignored / set to this value.

    Returns:
        dict: A dictionary containing signed distance field information. If the input tensor requires gradients, the
            gradient will be included in the output.
    """
    if tensor_device_type is None:
        tensor_device_type = TensorDeviceType()

    wm = world_collision.world_model
    device = tensor_device_type.device
    ret = {}
    for name, spheres in query_spheres.items():
        gradients = None
        if len(wm.sphere)  > 0:
            if not len(wm.mesh + wm.cuboid + wm.capsule + wm.cylinder + wm.blox) == 0:
                raise ValueError("World model must be either only spheres or only non-spheres.")
            spheres = spheres.squeeze()
            assert spheres.dim() == 2, "Not implemented"
            col = stack_spheres(wm.sphere).unsqueeze(0)
            delta = spheres.unsqueeze(1)[..., :3] - col[..., :3]
            d, idx = torch.min(torch.norm(delta, dim=-1), dim=-1)
            d = -(d - spheres[..., 3] - col[0, idx, 3])
            if spheres.requires_grad:
                direction = -delta[torch.arange(d.numel()), idx]
                direction = direction / torch.norm(direction, dim=-1, keepdim=True)
                gradients = torch.zeros(d.numel(), 4, device=device)
                gradients[:, :3] = direction
        else:
            query_buffer = CollisionQueryBuffer.initialize_from_shape(spheres.shape, TensorDeviceType(),
                                                                      {t: True for t in collision_types})
            d = world_collision.get_sphere_distance(spheres, query_buffer, weight=torch.tensor([1.], device=device),
                                                    activation_distance=torch.tensor([activation_distance], device=device),
                                                    compute_esdf=True)
            if spheres.requires_grad:
                gradients = query_buffer.get_gradient_buffer().squeeze()

        ret[name] = SdfInfo(distances=d, spheres=spheres, grad=gradients)

    return ret

def stack_spheres(spheres: Iterable[Sphere]) -> torch.Tensor:
    """Stacks spheres into a single tensor."""
    return torch.tensor(
            [qs.position + [qs.radius] for qs in spheres],
            device=TensorDeviceType().device)

def trimesh_to_curobo_mesh(tm: trimesh.Trimesh, name) -> Mesh:
    return Mesh(
        pose=[0, 0, 0, 1, 0, 0, 0],
        name=name,
        vertices=tm.vertices.tolist(),
        faces=tm.faces.tolist(),
    )