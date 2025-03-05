import time

from curobo.types.math import Pose
import numpy as np
import pyvista as pv
import torch

from biff.geometry.boolean_ball_sdf import make_finger_mesh, plot_inertia_deltas
from biff.optimization.grasp import MoveToContactPoint
from biff.optimization.pygmo import optimize_udp, udp_sgd
from docbrown.utilities.geometry import pv_to_trimesh
from mcfly.utilities.curobo import get_sdf, trimesh_to_curobo_mesh
from mcfly.utilities.sdf import CuroboMeshSdf


def main(discretize: bool = False):
    r = .5
    t0 = 0.75
    finger_mesh = make_finger_mesh().translate((0., 0., .3))
    left_finger = finger_mesh.copy(deep=True).rotate_z(180).translate((-t0, 0, 0))
    right_finger = finger_mesh.translate((t0, 0, 0))
    rf_sdf = CuroboMeshSdf.from_meshes([trimesh_to_curobo_mesh(pv_to_trimesh(right_finger), f'right')],
                                       max_distance=10 * r)

    sphere_mesh = trimesh_to_curobo_mesh(pv_to_trimesh(pv.Sphere(radius=r)), 'object')
    sphere_sdf = CuroboMeshSdf.from_meshes([sphere_mesh], max_distance=r)

    query_spheres = sphere_sdf.discretize(100)

    project_onto = torch.tensor([1., 0, 0], dtype=torch.float32)

    # TODO: Why does this not find a valid solution?
    # p_contact = MoveToContactPoint(rf_sdf, query_spheres, bounds=np.array([-.5, 2]), move_direction=project_onto)
    # udp_sgd(p_contact, x0=np.array([0., 0., 0.]), step_size=1e-1, num_iter=500)
    # population = optimize_udp(p_contact.reference_clone(), x0=[np.array([0., 0., 0.])], verbosity=1)

    # First step: establish contact
    d, g = rf_sdf(query_spheres.sph, gradients=True)
    d_max = d.max().item()
    while abs(d_max) > 1e-2:
        # Note: We compute the sdf of the obstacle w.r.t. the finger -- the gradient thus points "in the wrong direction"
        direction = torch.dot(g[..., :3].mean(dim=0), project_onto) * project_onto
        direction /= torch.norm(direction)
        step = direction * d_max
        rf_sdf.translate(step)
        d, g = rf_sdf(query_spheres.sph, gradients=True)
        d_max = d.max().item()

    # Second step: Move inwards, optimizing arbitrary objectives
    # TODO: Implement me

    surface, normals = rf_sdf.get_surface_points(rf_sdf.get_center(), rf_sdf.get_dims(), 6)
    reconstruction = pv.wrap(surface.detach().cpu().numpy())
    reconstruction.point_data.active_normals = normals.detach().cpu().numpy()
    surface = reconstruction.reconstruct_surface()
    surface.plot()

if __name__ == '__main__':
    torch.set_default_device('cuda')
    main()