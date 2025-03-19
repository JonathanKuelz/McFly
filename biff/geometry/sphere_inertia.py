import matplotlib.pyplot as plt
import pyvista as pv
import torch

from docbrown.utilities.geometry import pv_to_trimesh
from mcfly.utilities.curobo import trimesh_to_curobo_mesh
from mcfly.representations.sdf import CuroboMeshSdf


def main():
    r = 1.
    density = 1.
    mass = density * 4 / 3 * torch.pi * r**3
    ixx_gt = 2 / 5 * mass * r**2
    vol_gt = 4 / 3 * torch.pi * r**3

    mesh = trimesh_to_curobo_mesh(pv_to_trimesh(pv.Sphere(radius=r)), 'object')
    sphere_sdf = CuroboMeshSdf.from_meshes([mesh], max_distance=r)

    resolutions = (20, 30, 50, 100, 200, 300)
    inertia_errors = []
    volume_errors = []
    for res in resolutions:
        volume = sphere_sdf.approximate_volume(center=torch.tensor([0, 0, 0]), dims=[2 * r] * 3, pts_per_dim=res)
        volume_errors.append(100 * abs(volume - vol_gt) / vol_gt)

        inertia = sphere_sdf.approximate_inertia(center=torch.tensor([0, 0, 0]), dims=[2 * r] * 3, pts_per_dim=res)
        ixx = inertia[0, 0].item()
        inertia_errors.append(100 * abs(ixx - ixx_gt) / ixx_gt)

    res = 200
    voxel_width = 2 * r / res
    deltas = torch.tensor([0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]) * voxel_width
    perspective_inertia_errors = []
    for delta in deltas:
        inertia = sphere_sdf.approximate_inertia(center=torch.tensor([delta, 0, 0]), dims=[2 * (r + voxel_width)] * 3,
                                                 pts_per_dim=res)
        ixx = inertia[0, 0].item()
        perspective_inertia_errors.append(100 * abs(ixx - ixx_gt) / ixx_gt)

    fig, (axi, axv, axp) = plt.subplots(nrows=3, figsize=(8, 8))
    fig.suptitle('Approximation Errors due to Sphere Discretization')
    axi.plot(resolutions, inertia_errors)
    axi.set_xlabel('Sample Resolution')
    axi.set_ylabel('$I_{xx}$ Error (%)')

    axv.plot(resolutions, volume_errors)
    axv.set_xlabel('Sample Resolution')
    axv.set_ylabel('Volume Error (%)')

    axp.plot(deltas.detach().cpu().numpy(), perspective_inertia_errors)
    axp.set_xlabel('Perspective Offset')
    axp.set_ylabel('$I_{xx}$ Error (%) due to Perspective')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    torch.set_default_device('cuda')
    main()
