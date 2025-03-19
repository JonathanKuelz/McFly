import time

from curobo.types.math import Pose
import numpy as np
import pyvista as pv
import torch

from biff.geometry.boolean_ball_sdf import make_finger_mesh, plot_inertia_deltas
from docbrown.utilities.geometry import pv_to_trimesh
from mcfly.utilities.curobo import trimesh_to_curobo_mesh
from mcfly.representations.sdf import CuroboMeshSdf


def main(discretize: bool = False):
    UQ = torch.tensor([1., 0., 0., 0.])  # Unit Quaternion for CuRobo
    r = .5
    t0 = 0.75
    finger_mesh = make_finger_mesh().translate((0., 0., .3))
    left_finger = finger_mesh.copy(deep=True).rotate_z(180).translate((-t0, 0, 0))
    right_finger = finger_mesh.translate((t0, 0, 0))
    finger_sdf = CuroboMeshSdf.from_meshes([trimesh_to_curobo_mesh(pv_to_trimesh(m), f'{i}')
                                            for i, m in enumerate((left_finger, right_finger))],
                                           max_distance=r)

    sphere_mesh = trimesh_to_curobo_mesh(pv_to_trimesh(pv.Sphere(radius=r)), 'object')
    sphere_sdf = CuroboMeshSdf.from_meshes([sphere_mesh], max_distance=r)

    offsets = np.linspace(.25, .7, 20)

    resolutions = (20, 30, 50, 100, 150)
    inertia_deltas = []
    penetration_volume = []

    for resolution in resolutions:
        t0 = time.time()
        id_ = []
        pv_ = []

        kwargs = dict(center=torch.tensor([0, 0, 0]), dims=[2.] * 3, pts_per_dim=resolution)
        def get_inert(sdf):
            return sdf.approximate_inertia(**kwargs)

        inertia = get_inert(sphere_sdf)
        base_volume = (finger_sdf.sample(**kwargs)[1] > 0).sum().item()
        for delta in offsets:
            for i, name in enumerate(finger_sdf.mesh_names):
                sign = 1 if i == 0 else -1
                pos = torch.tensor([sign * delta, 0, 0], dtype=torch.float32)
                finger_sdf.wcm.update_mesh_pose(Pose(pos, UQ), name=name)
            new_obj = sphere_sdf.boolean_difference(finger_sdf)
            new_inertia = get_inert(new_obj)
            id_.append((100 * (new_inertia - inertia) / inertia).detach().cpu().numpy())

            new_volume = (sphere_sdf.boolean_intersection(finger_sdf).sample(**kwargs)[1] > 0).sum().item()
            if base_volume == 0:
                pv_.append(0)
            else:
                pv_.append(100 * new_volume / base_volume)

        if resolution == resolutions[-1]:
            new_obj.pv_plot(**kwargs)

        inertia_deltas.append(id_)
        penetration_volume.append(pv_)
        print(f'{resolution}: {time.time() - t0:.2f}s')

    plot_inertia_deltas(offsets, resolutions, inertia_deltas, penetration_volume)

if __name__ == '__main__':
    torch.set_default_device('cuda')
    main()