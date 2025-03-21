import torch

from mcfly.representations.levelSet import LevelSetRepresentation
from mcfly.utilities.meshviz import show_hist
from mcfly.utilities.pointmass import PointMass


def is_interior_ellipsoid(p: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    """https://en.wikipedia.org/wiki/Ellipsoid"""
    x, y, z = p.unbind(dim=-1)
    return ((x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2) <= 1.0

def n_velocities(v: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return v / torch.linalg.norm(v, dim=-1).abs().max()

def n_velocities_ratios(v: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return v / torch.linalg.norm(v, dim=-1, keepdim=True)

def n_effects(v: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    effect = torch.einsum('ijkl,ijkl->ijk', v, n)
    return v / effect.abs().max()

def n_effects_ratios(v: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    effect = torch.einsum('ijkl,ijkl->ijk', v, n).unsqueeze(-1)
    return v / effect.abs()

def main():
    hw = 1.
    a, b, c = .7, .5, .3
    pts = [torch.linspace(-hw, hw, 40)] * 3
    grid = torch.meshgrid(pts)
    grid_tensor = torch.stack(grid, dim=-1)
    level_set = -torch.ones(grid_tensor.shape[:-1], device=grid_tensor.device)
    level_set[is_interior_ellipsoid(grid_tensor, a, b, c)] = 1.0

    options = {
        "n_velocities": n_velocities,
        "n_velocities_ratios": n_velocities_ratios,
        "n_effects": n_effects,
        "n_effects_ratios": n_effects_ratios,
    }

    for name, normalize in options.items():
        repr = LevelSetRepresentation(grid, level_set)
        repr.reinitialize()

        meshes = list()
        point_masses = PointMass(repr.grid_tensor, 1.0)
        inertia = point_masses.inertia(torch.tensor([0., 0., 0.])).sum(dim=(0, 1, 2))
        loss = point_masses.m.sum() - (inertia[..., 1, 1] + inertia[..., 2, 2])
        loss.backward()

        v = -point_masses.p.grad
        v = v - point_masses.m_nominal.grad.unsqueeze(-1) * repr.normals

        for i in range(600):
            v = normalize(v, repr.normals)
            repr.evolve(v, 0.01)
            if i % 50 == 0:
                repr.reinitialize()
                meshes.append(repr.get_trimesh())

        show_hist(meshes, grid_tensor, title=name)


if __name__ == "__main__":
    torch.set_default_device('cuda')
    torch.manual_seed(0)
    main()