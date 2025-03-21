from typing import Union

import torch

from biff.optimization.addremove import parallel_axis_theorem


class PointMass:

    def __init__(self, p: torch.Tensor, m_nominal: Union[torch.Tensor, float]):
        """
        Initializes point masses, intended as a discrete representation for Level Set Methods.

        Args:
            p (torch.Tensor): Position of the point masses. Should be of shape (..., 3).
            m_nominal (Union[torch.Tensor, float]): Mass of the point mass. Either a single value that is broadcast to
                all point masses or a tensor of the same shape as p.
        """
        p = p.detach().clone().requires_grad_()
        self.p = p
        if not isinstance(m_nominal, torch.Tensor):
            m_nominal = m_nominal * torch.ones_like(p[..., 0])
        self.m_nominal = m_nominal.requires_grad_(True)

    @property
    def p_flat(self) -> torch.Tensor:
        """Flattened position of the point masses."""
        return self.p.view(-1, 3)

    @property
    def m_flat(self) -> torch.Tensor:
        """Flattened mass of the point masses."""
        return self.m.view(-1)

    @property
    def num_points(self) -> int:
        """Number of point masses."""
        return self.p_flat.shape[0]

    def com(self, center: torch.Tensor):
        """Compute the center of mass contribution of the point mass with respect to a coordinate system placed in center."""
        com = self.p / self.m.unsqueeze(-1)
        return com - center

    def inertia(self, center: torch.Tensor):
        """Compute the inertia of the point mass with respect to a coordinate system placed in center."""
        d = self.p_flat - center
        i_com = torch.zeros(self.num_points, 3, 3, device=self.p.device)
        i = parallel_axis_theorem(i_com, self.m_flat.unsqueeze(-1), d)
        return i.view(*self.p.shape, 3)

    @property
    def m(self):
        return self.m_nominal
    #
    # def mass(self, level_set_values: torch.Tensor):
    #     """
    #     You could think the mass of a pointmass is constant, but it is not that easy: In Level Set Methods, points
    #     either belong to a surface, or they do not. The mass plays an important role in computing other attributes like
    #     inertia. We need to consider a "continuous relaxation" of the "mass contribution" to the implicit shape to
    #     pass gradients through things like inertia computation back to the level set function value of the point, which
    #     we do by implementing a soft mass.
    #     """
    #     # Linear interpolation
    #     min_score = 0.2
    #     max_score = 1.
    #     scores =
