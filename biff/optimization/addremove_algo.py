from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.autograd import Function
from tqdm import tqdm

from mcfly.utilities.sdf import BoundedSdf, CuroboMeshSdf, Sdf, SphereSdf


def parallel_axis_theorem(i_com: torch.Tensor, m: torch.Tensor, d: torch.Tensor):
    """Computes the inertia of a body with translated by d using the parallel axis theorem."""
    eye = torch.eye(3).to(d).unsqueeze(0).repeat(d.shape[0], 1, 1)
    return i_com + m.unsqueeze(-1) * (torch.norm(d, dim=1).unsqueeze(-1).unsqueeze(-1) ** 2 * eye
                                      - torch.einsum('bi,bj->bij', d, d))

@dataclass
class Worker:

    center: torch.Tensor
    radius: Union[torch.Tensor, float]
    density: float = 1.
    add_material: bool = False
    carve_material: bool = False

    def __post_init__(self):
        self.center.requires_grad = True
        if isinstance(self.radius, float):
            self.radius: torch.Tensor = torch.ones((self.center.shape[0], 1)).to(self.center) * self.radius
        self.radius.requires_grad = True

    @property
    def aabb(self) -> torch.Tensor:
        """Compute the aabb of all workers."""
        return self.sdf.aabb

    @property
    def com_inertia(self) -> torch.Tensor:
        """Compute the inertia of the worker with respect to their own center of mass."""
        inertia = torch.zeros((self.num_workers, 3, 3)).to(self.center)
        i = (2 / 5) * self.mass * self.radius ** 2
        inertia[:, 0, 0] = i.squeeze()
        inertia[:, 1, 1] = i.squeeze()
        inertia[:, 2, 2] = i.squeeze()
        return inertia

    @property
    def mass(self) -> torch.Tensor:
        """Compute the mass of the worker."""
        return self.density * self.volume

    @property
    def num_workers(self) -> int:
        """Number of workers."""
        return self.center.shape[0]

    @property
    def query_shapes(self) -> torch.Tensor:
        """Returns geometric primitives describing the workers."""
        return torch.concat((self.center, self.radius), dim=1)

    @property
    def volume(self) -> torch.Tensor:
        """Compute the volume of the worker."""
        vol = (4. / 3.) * torch.pi * self.radius ** 3
        if self.add_material:
            return vol
        else:
            return -vol

    @property
    def sdf(self):
        return SphereSdf(self.query_shapes.detach().clone())

    def enable_addition(self):
        """Enable addition of material."""
        self.add_material = True
        self.carve_material = False

    def enable_carving(self):
        """Enable carving of material."""
        self.add_material = False
        self.carve_material = True

    def get_relevance_score(self, shape: Sdf) -> torch.Tensor:
        """
        Computes a score indicating how much the worker is changing the material.
        """
        d, _ = QuerySdf.apply(self.query_shapes, shape)
        with torch.no_grad():
            weight =  1 / self.radius.squeeze()
        if self.add_material:
            return torch.sigmoid(-d * weight)  # Higher scores fore being outside
        else:
            return torch.sigmoid(d * weight)  # Higher scores fore being inside

    def get_worker_inertia(self, origin: torch.Tensor) -> torch.Tensor:
        """Compute the approximated inertia that the workers add to a shape with reference coordinates at origin."""
        d = self.center - origin
        return parallel_axis_theorem(self.com_inertia, self.mass, d)

    def work_on_shape(self, shape: Sdf) -> Sdf:
        """Add the workers body to the shape if it is in the exterior."""
        if self.add_material:
            return self._add_material(shape)
        elif self.carve_material:
            return self._carve_material(shape)
        else:
            raise ValueError("Workers are not enabled for addition or carving.")

    def _add_material(self, shape: Sdf):
        """Add the workers body to the shape if it is in the exterior."""
        mask = shape(self.query_shapes) < self.radius.squeeze()
        if mask.sum() == 0:
            return shape
        worker_sdf = SphereSdf(self.query_shapes.detach().clone()[mask])
        return shape.boolean_union(worker_sdf)

    def _carve_material(self, shape: Sdf):
        """Remove the workers body from the shape if it is in the exterior."""
        mask = shape(self.query_shapes) > 0
        if mask.sum() == 0:
            return shape
        worker_sdf = SphereSdf(self.query_shapes.detach().clone()[mask])
        return shape.boolean_difference(worker_sdf)


class QuerySdf(Function):
    """Query SDF with autograd support. Returns the distance and the gradient."""

    @staticmethod
    def forward(query_shapes: torch.Tensor, shape: BoundedSdf) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the gradient in the forward pass because not all SDFs support autograd."""
        d, g = shape(query_shapes, gradients=True)
        return d, g

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save the gradient for later"""
        d, g = output
        ctx.save_for_backward(g)

    @staticmethod
    def backward(ctx, grad_query, grad_shape):
        """Compute the backward gradient using a simple chain rule."""
        if grad_query is not None:
            grad_query = grad_query.unsqueeze(-1) * ctx.saved_tensors[0]
        return grad_query, None


class FleeingWorkerException(Exception):
    """Exception raised when the workers are out of bounds."""
    pass


class AddRemoveAlgorithm(ABC):

    oversample: int = 10

    def __init__(self,
                 base_shape: BoundedSdf,
                 max_workers: int,
                 worker_radius: float,
                 n_cycles: int,
                 steps_per_cycle: int,
                 bounds: Optional[torch.Tensor] = None,
                 *,
                 get_optimizer: Callable[[AddRemoveAlgorithm, List[torch.Tensor]], torch.optim.Optimizer] = None,
                 worker_sampling: str = 'grid'
                 ):
        """
        Args:
            base_shape (BoundedSdf): The base shape to optimize.
            max_workers (int): The maximum number of workers.
            worker_radius (float): The radius of the workers.
            n_cycles (int): The number of add/remove cycles to run.
            steps_per_cycle (int): The number of steps per cycle.
            bounds (Optional[torch.Tensor]): The bounds for the workers. Should be 2x3 with the min and max bounds.
            get_optimizer (Callable[[List[torch.Tensor]], torch.optim.Optimizer]): A function that takes parameters
                and returns an optimizer.
            worker_sampling (str): The sampling strategy to use. Either 'grid' or 'random'.
        """
        self.base_shape = base_shape
        self.bounds = bounds
        self.center = self.base_shape.get_center()
        self.shape = base_shape

        self.random_sampling = worker_sampling == 'random'
        self.max_workers = max_workers
        self.worker_radius = worker_radius
        self.workers = None

        self.add_steps: int = steps_per_cycle
        self.remove_steps: int = steps_per_cycle
        self.n_cycles: int = n_cycles

        if get_optimizer is None:
            get_optimizer = self.default_optim
        self.get_optimizer: Callable[[AddRemoveAlgorithm, List[torch.Tensor]], torch.optim.Optimizer] = get_optimizer

    @abstractmethod
    def loss(self) -> torch.Tensor:
        """Compute the loss for the optimization."""
        pass

    @staticmethod
    def default_optim(algo, params: List[torch.Tensor]) -> torch.optim.Optimizer:
        """Default optimizer for the workers."""
        return torch.optim.SGD([
            {'params': params[0], 'lr': 1e-3, 'name': 'center'},
            {'params': params[1], 'lr': 1e-4, 'name': 'radius'}
        ])

    def check_bounds(self):
        """Makes sure the workers are not working outside their bounds."""
        if self.bounds is None or self.workers.carve_material:
            return
        aabb = self.workers.aabb
        if (aabb[0].min() < self.bounds[0].min()).any() or (aabb[1].max() > self.bounds[1].max()).any():
            raise FleeingWorkerException("Workers are out of bounds.")

    def optimization_cycle(self):
        """Moves the workers to the exterior to expand the current shape."""
        optim = self.get_optimizer(self, [self.workers.center, self.workers.radius])

        for step in tqdm(range(self.add_steps), desc="Adding material"):
            optim.zero_grad()
            loss = self.loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.workers.center, self.workers.radius], self.max_workers)
            optim.step()
            self.shape = self.workers.work_on_shape(self.shape)
            try:
                self.check_bounds()
            except FleeingWorkerException:
                break

    def add_material(self):
        """Add material to the shape."""
        self.workers = Worker(self.sample(interior=True), radius=self.worker_radius)
        self.workers.enable_addition()
        self.optimization_cycle()

    def remove_material(self):
        """Add material to the shape."""
        self.workers = Worker(self.sample(interior=False), radius=self.worker_radius)
        self.workers.enable_carving()
        self.optimization_cycle()

    def sample(self, interior: bool = True) -> torch.Tensor:
        """Sample points to spawn the workers."""
        dim = self.bounds[1] - self.bounds[0]
        num_sample = self.max_workers * self.oversample
        if self.random_sampling:
            pts = torch.rand((num_sample, 3))
            grid = self.center + dim * (pts - .5)
            d = self.shape(grid)
        else:
            pts_per_dim = int(np.ceil(num_sample ** (1 / 3)))
            grid, d = self.shape.sample(self.center, dim, pts_per_dim)
        if interior:
            mask = d >= 0
        else:
            mask = d <= 0
        if not mask.any():
            raise ValueError("No valid points found for the workers.")
        valid_samples = torch.stack(grid, dim=-1)[mask]
        if valid_samples.shape[0] <= self.max_workers:
            return valid_samples
        else:
            choice = torch.randperm(valid_samples.shape[0])[:self.max_workers]
            return valid_samples[choice]

    def optimize(self):
        """Optimize the workers to add material to the shape."""
        for cycle in range(self.n_cycles):
            print(f"Cycle {cycle + 1}/{self.n_cycles}")
            self.add_material()
            self.remove_material()
            if cycle < self.n_cycles - 1:
                pass
                # self.shape = self.shape.remesh(self.center, self.bounds[1] - self.bounds[0], 7)

    def visualize(self):
        """Visualize the current shape."""
        dims = self.bounds[1] - self.bounds[0]
        self.shape.pv_plot(self.center, dims, pts_per_dim=120)


class XxRollingBehavior(AddRemoveAlgorithm):

    def loss(self):
        inertia = self.workers.get_worker_inertia(self.center)
        scores = self.workers.get_relevance_score(self.shape)
        workers_reward = (3 * inertia[:, 0, 0] - inertia[:, 1, 1] - inertia[:, 2, 2]) * scores
        reward = workers_reward.sum() * 1e3
        return -reward


def main():
    radius = 0.5
    shape_gt: BoundedSdf = SphereSdf(torch.tensor([[0., 0., 0., radius]]))
    # shape = remesh(shape_gt, origin, shape_gt.get_dims())
    shape = shape_gt

    max_workers = 1000
    n_cycles: int = 10
    steps_per_iteration: int = 120
    bounds = torch.tensor([[-1., -1., -1.], [1., 1., 1.]])
    algo = XxRollingBehavior(shape, max_workers, 0.03, n_cycles, steps_per_iteration, bounds=bounds)
    algo.optimize()
    algo.visualize()
    print(algo.workers.aabb)
    # TODO: Spawn them on the surface, not in the interior. Also, can I somehow "close" the shape again?
    # TODO: See how it works if theres really A LOT of them
    # TODO: Stop the workers before they reach the bounds, leave the others running


if __name__ == "__main__":
    torch.set_default_device('cuda')
    main()