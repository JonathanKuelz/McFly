from copy import deepcopy
import time

from curobo.geom.sdf.world import CollisionCheckerType, WorldCollisionConfig
from curobo.geom.sdf.world_mesh import WorldMeshCollision
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import trimesh
import pyvista as pv

from docbrown.utilities.geometry import pv_to_trimesh, show_pv_mesh
from mcfly.utilities.curobo import trimesh_to_curobo_mesh
from mcfly.representations.sdf import CuroboMeshSdf, SphereSdf


def make_finger_mesh():
    w, h = .1, .4
    h_tip = .15
    finger_base = pv.Cube(center=(h_tip + w / 2, 0, (h - w) / 2), x_length=w, y_length=w, z_length=h).triangulate()
    finger_tip = pv.Cube(center=(h_tip / 2, 0, 0), x_length=h_tip, y_length=w, z_length=w).triangulate()
    finger_base = pv_to_trimesh(finger_base)
    finger_tip = pv_to_trimesh(finger_tip)
    tm = trimesh.boolean.union([finger_base, finger_tip])
    return pv.wrap(tm)

def plot_pv(object_spheres, finger_spheres):
    lf = [pv.Sphere(t[-1].item(), center=t[:3].detach().cpu().numpy()) for t in finger_spheres['left'].squeeze()]
    rf = [pv.Sphere(t[-1].item(), center=t[:3].detach().cpu().numpy()) for t in finger_spheres['right'].squeeze()]
    ws = [pv.Sphere(s.radius, center=s.position) for s in object_spheres]
    show_pv_mesh(lf + rf + ws)

def plot_inertia_deltas(offsets,
                        resolutions,
                        inertia_deltas_,
                        contact_fraction_):
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(8, 6), sharex=True, sharey='col')
    sns.despine()

    for resolution, inertia_deltas, contact_fraction in zip(resolutions, inertia_deltas_, contact_fraction_):
        for i, axis in zip(range(3), 'xyz'):
            name = '{' + 2 * axis + '}'
            ax[i, 0].plot(offsets, [di[i, i] for di in inertia_deltas], label=str(resolution))
            ax[i, 0].set_ylabel(fr"$I_{name}$ offset (%)")

            ax[i, 1].plot(offsets, contact_fraction, label=str(resolution))
            ax[i, 1].set_ylabel("% penetration volume")

    for i in range(2):
        ax[2, i].set_xlabel("Finger offset")

    ax[2, 0].legend(fontsize='small', fancybox=True, title='Sample Resolution', loc='lower left')
    plt.tight_layout()
    plt.show()
    print()


def main():
    r = .5
    finger = make_finger_mesh().translate((0., 0., .3))
    left_finger = finger.copy(deep=True).rotate_z(180).translate((-0.75, 0, 0))
    right_finger = finger.translate((0.75, 0, 0))

    finger_sdfs = list()
    for name, mesh in zip(('left', 'right'), (left_finger, right_finger)):
        cfg = WorldConfig(mesh=[trimesh_to_curobo_mesh(pv_to_trimesh(mesh), name)])
        ccfg = WorldCollisionConfig(
            tensor_args=TensorDeviceType(),
            world_model=cfg,
            checker_type=CollisionCheckerType.MESH,
            max_distance=r
        )
        world_collision = WorldMeshCollision(ccfg)
        finger_sdfs.append(CuroboMeshSdf(world_collision))

    inertia_deltas = []
    penetration_volume = []
    offsets = np.linspace(.25, .7, 20)
    obj_sdf = SphereSdf(torch.tensor([[0, 0, 0, r]]))

    resolutions = (20, 30, 50, 100, 150)
    finger_resolutions = (10, 10, 10, 10, 10)
    # Can't discretize the fingers too fine, this will cause memory issues

    for resolution, fr in zip(resolutions, finger_resolutions):
        t0 = time.time()
        lf_ = finger_sdfs[0].discretize(fr)
        rf_ = finger_sdfs[1].discretize(fr)
        obj_inertia = obj_sdf.approximate_inertia(center=torch.tensor([0, 0, 0]), dims=[r * 2.2] * 3, pts_per_dim=resolution)

        id_ = []
        pv_ = []
        for delta in offsets:
            translation = torch.tensor([delta, 0, 0, 0])
            lf = deepcopy(lf_)
            rf = deepcopy(rf_)
            lf.sph[..., :4] += translation
            rf.sph[..., :4] -= translation
            finger_sdf = lf.boolean_union(rf)
            sdf = obj_sdf.boolean_difference(finger_sdf)
            new_inertia = sdf.approximate_inertia(center=torch.tensor([0, 0, 0]), dims=[r * 2.2] * 3, pts_per_dim=resolution)
            id_.append(((new_inertia - obj_inertia) / obj_inertia).detach().cpu().numpy())
            pv_.append(100 * (obj_sdf(finger_sdf.sph) > 0).sum().item() / finger_sdf.sph.shape[0])

        inertia_deltas.append(id_)
        penetration_volume.append(pv_)
        print(f'{resolution}: {time.time() - t0:.2f}s')

    plot_inertia_deltas(offsets, resolutions, inertia_deltas, penetration_volume)


if __name__ == '__main__':
    torch.set_default_device('cuda')
    main()