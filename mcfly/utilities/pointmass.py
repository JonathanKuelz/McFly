from typing import Union

import torch

from biff.optimization.addremove import parallel_axis_theorem


class PointMass:

    def __init__(self, p: torch.Tensor, m: Union[torch.Tensor, float]):
        """
        Initializes point masses.

        Args:
            p (torch.Tensor): Position of the point masses. Should be of shape (..., 3).
            m (Union[torch.Tensor, float]): Mass of the point mass. Either a single value that is broadcast to all
                point masses or a tensor of the same shape as p.
        """
        p = p.detach().clone().requires_grad_()
        self.p = p
        if not isinstance(m, torch.Tensor):
            m = m * torch.ones_like(p[..., 0])
        self.m = m

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

    def inertia(self, center: torch.Tensor):
        """Compute the inertia of the worker with respect to a coordinate system placed in center."""
        d = self.p_flat - center
        i_com = torch.zeros(self.num_points, 3, 3, device=self.p.device)
        i = parallel_axis_theorem(i_com, self.m_flat.unsqueeze(-1), d)
        return i.view(*self.p.shape, 3)
