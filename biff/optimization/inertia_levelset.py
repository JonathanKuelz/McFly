import torch

from mcfly.representations.sdf import BoundedSdf, CuroboMeshSdf, SphereSdf
from mcfly.representations.levelSet import LevelSetRepresentation
from mcfly.utilities.meshviz import animate, show_hist
from mcfly.utilities.pointmass import PointMass


def main():
    radius = 0.2
    shape_gt: BoundedSdf = SphereSdf(torch.tensor([[0., 0., 0., radius]]))
    dims = torch.tensor([1., 1., 1.])

    grid, sd = shape_gt.sample(shape_gt.get_center(), dims, pts_per_dim=40)
    level_set = LevelSetRepresentation(grid, sd)

    meshes = list()
    point_masses = PointMass(level_set.grid_tensor, 1.0)
    inertia = point_masses.inertia(torch.tensor([0., 0., 0.])).sum(dim=(0, 1, 2))
    loss = (2.5 * inertia[..., 0, 0] - inertia[..., 1, 1] - inertia[..., 2, 2])
    loss.backward()
    v = point_masses.p.grad
    # v = level_set.normals * torch.einsum('ijkl,ijkl->ijk', v, level_set.normals).unsqueeze(-1)
    # v = v / torch.linalg.norm(v, dim=-1, keepdim=True)
    v = v / torch.linalg.norm(v, dim=-1).max() * 1.5

    for i in range(36):
        level_set.evolve(v, 0.01)
        if i % 3 == 0:
            meshes.append(level_set.get_trimesh())

    show_hist(meshes, grid)


if __name__ == "__main__":
    torch.set_default_device('cuda')
    torch.manual_seed(0)
    main()