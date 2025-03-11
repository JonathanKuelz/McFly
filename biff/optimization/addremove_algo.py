from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable,  List, Optional, Tuple, Union

from faker import Faker
import numpy as np
import pyvista as pv
import torch
from torch.autograd import Function
from tqdm import tqdm

from mcfly.utilities.file_locations import PLOTS
from mcfly.utilities.sdf import BoundedSdf, Sdf, SphereSdf


def parallel_axis_theorem(i_com: torch.Tensor, m: torch.Tensor, d: torch.Tensor):
    """Computes the inertia of a body with translated by d using the parallel axis theorem."""
    eye = torch.eye(3).to(d).unsqueeze(0).repeat(d.shape[0], 1, 1)
    return i_com + m.unsqueeze(-1) * (torch.norm(d, dim=1).unsqueeze(-1).unsqueeze(-1) ** 2 * eye
                                      - torch.einsum('bi,bj->bij', d, d))

class Worker:

    def __init__(self,
                 center: torch.Tensor,
                 radius: Union[torch.Tensor, float],
                 density: float = 1.,
                 *,
                 min_radius: float = 0.01,
                 ):
        self.center: torch.Tensor = center.detach().requires_grad_(True)
        if isinstance(radius, float):
            radius: torch.Tensor = torch.ones((self.center.shape[0], 1)).to(self.center) * radius
        if not (radius >= min_radius).all():
            raise ValueError('radius must be >= min_radius')
        self._radius: torch.Tensor = radius.requires_grad_(True)
        self._min_radius: torch.Tensor = torch.ones_like(radius).requires_grad_(False) * min_radius
        self.density: float = density
        self.active = torch.ones(self.center.shape[0], dtype=torch.bool).to(self.center.device)
        self.add_material: bool = False
        self.carve_material: bool = False

    @property
    def aabb(self) -> torch.Tensor:
        """Compute the aabb of all workers."""
        return self.sdf.aabb

    @property
    def com_inertia(self) -> torch.Tensor:
        """Compute the inertia of the worker with respect to their own center of mass."""
        inertia = torch.zeros((self.active.sum(), 3, 3)).to(self.center)
        # i = (2 / 5) * self.mass * self.radius[self.active] ** 2
        i = (2 / 5)  * (4. / 3.) * self.radius[self.active] ** 5 * self.density * torch.pi
        if self.carve_material:
            i = -i
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
    def optimization_parameters(self) -> List[torch.Tensor]:
        """Returns the leaf parameters to optimize."""
        return [self.center, self._radius]

    @property
    def query_shapes(self) -> torch.Tensor:
        """Returns geometric primitives describing the workers."""
        return torch.concat((self.center[self.active], self.radius[self.active]), dim=1)

    @property
    def radius(self) -> torch.Tensor:
        """Makes sure the radius of the workers is always larger than the defined minimum."""
        return torch.maximum(self._radius, self._min_radius)

    @property
    def volume(self) -> torch.Tensor:
        """Compute the volume of the worker."""
        vol = (4. / 3.) * torch.pi * self.radius[self.active] ** 3
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

        Args:
            shape (Sdf): The shape that the signed distance should be computed for. This should typicalle NOT be the
                shape that is changed in every iteration, because this would lead to unstable gradients.
        """
        d, _g = QuerySdf.apply(self.query_shapes, shape, shape, True)
        with torch.no_grad():
            weight =  1 / self.radius[self.active].squeeze()
        if self.add_material:
            return torch.sigmoid(-d * weight) / torch.sigmoid(torch.tensor([-1])) # Higher scores fore being outside
        else:
            if d.abs().mean() < 0.0001:
                print()
            shape(self.query_shapes)
            return torch.sigmoid(d * weight) / torch.sigmoid(torch.tensor([1])) # Higher scores fore being inside

    def get_worker_inertia(self, origin: torch.Tensor) -> torch.Tensor:
        """Compute the approximated inertia that the workers add to a shape with reference coordinates at origin."""
        d = self.center[self.active] - origin
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
        """
        Add the workers body to the shape if it is in the exterior.

        Args:
            shape (Sdf): The shape to add the workers body to.
        """
        mask = shape(self.query_shapes) < self.radius[self.active].squeeze()
        if mask.sum() == 0:
            return shape
        worker_sdf = SphereSdf(self.query_shapes.detach().clone()[mask])
        return shape.boolean_union(worker_sdf)

    def _carve_material(self, shape: Sdf):
        """
        Remove the workers body from the shape if it is in the exterior.

        Args:
            shape (Sdf): The shape to remove the workers body
        """
        mask = shape(self.query_shapes) > 0
        if mask.sum() == 0:
            return shape
        worker_sdf = SphereSdf(self.query_shapes.detach().clone()[mask])
        return shape.boolean_difference(worker_sdf)


class QuerySdf(Function):
    """Query SDF with autograd support. Returns the distance and the gradient."""

    @staticmethod
    def forward(query_shapes: torch.Tensor,
                shape: BoundedSdf,
                gradient_shape: BoundedSdf,
                guess_missing_gradients: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the gradient in the forward pass because not all SDFs support autograd."""
        d, _ = shape(query_shapes, gradients=True)
        _, g = gradient_shape(query_shapes, gradients=True)
        return d, g

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save the gradient for later"""
        query, _, _, guess_gradients = inputs
        d, g = output
        ctx.save_for_backward(g, torch.tensor([guess_gradients]), query)

    @staticmethod
    def backward(ctx, grad_query, grad_shape):
        """Compute the backward gradient using a simple chain rule."""
        grad, guess_grad, query = ctx.saved_tensors
        if grad_query is not None:
            grad_query = grad_query.unsqueeze(-1) * grad

            if guess_grad.item():
                mask = torch.norm(grad[:, :3], dim=1) == 0.
                if mask.any():
                    grad_query[mask, :3] = (- query[:, :3] / query[:, :3].norm(dim=1).unsqueeze(-1))[mask]
        return grad_query, None, None, None


class FleeingWorkerException(Exception):
    """Exception raised when the workers are out of bounds."""
    pass


class AddRemoveAlgorithm(ABC):

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
        self.active_workers = None
        self.workers = None

        self.add_steps: int = steps_per_cycle
        self.remove_steps: int = steps_per_cycle
        self.n_cycles: int = n_cycles

        if get_optimizer is None:
            get_optimizer = self.default_optim
        self.get_optimizer: Callable[[AddRemoveAlgorithm, List[torch.Tensor]], torch.optim.Optimizer] = get_optimizer
        self.name = Faker().first_name()

    @abstractmethod
    def loss(self) -> torch.Tensor:
        """Compute the loss for the optimization."""
        pass

    @staticmethod
    def default_optim(algo, params: List[torch.Tensor]) -> torch.optim.Optimizer:
        """Default optimizer for the workers."""
        return torch.optim.SGD([
            {'params': params[0], 'lr': 1e-2, 'name': 'center'},
            {'params': params[1], 'lr': 1e-4, 'name': 'radius'}
        ])

    def optimize(self):
        """Optimize the workers to add material to the shape."""
        for cycle in range(self.n_cycles):
            print(f"Cycle {cycle + 1}/{self.n_cycles}")
            # self._add_material()
            # self.visualize_with_workers()
            # self.shape.plot_reconstructed_surface(self.center, self.bounds[1] - self.bounds[0])
            #
            # self.shape = self.shape.remesh(self.center, self.bounds[1] - self.bounds[0], 7, limit_to='exterior')

            self._remove_material()
            self.visualize_with_workers()
            self.shape.plot_reconstructed_surface(self.center, self.bounds[1] - self.bounds[0], refinement_steps=7)
            self.shape = self.shape.remesh(self.center, self.bounds[1] - self.bounds[0], 7, limit_to='interior')

    def optimization_cycle(self):
        """Moves the workers to the exterior to expand the current shape."""
        optim = self.get_optimizer(self, self.workers.optimization_parameters)
        for step in tqdm(range(self.add_steps), desc=f"{'Adding' if self.workers.add_material else 'Removing'} material"):
            optim.zero_grad()
            loss = self.loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.workers.optimization_parameters, 1.)
            optim.step()
            self.shape = self.workers.work_on_shape(self.shape)
            self._set_worker_active()

    def visualize(self):
        """Visualize the current shape."""
        dims = self.bounds[1] - self.bounds[0]
        self.shape.pv_plot(self.center, dims * 1.1, pts_per_dim=120)

    def visualize_with_workers(self):
        """Visualize the current shape with the workers."""
        dims = (self.bounds[1] - self.bounds[0]).detach().cpu().numpy()
        pt_radius = np.mean(dims) / 120
        grid, d = self.shape.sample(self.center, dims, 120, r=pt_radius)
        grid, d_worker = self.workers.sdf.sample(self.center, dims, 120, r=pt_radius)

        d = d.detach().cpu().numpy()
        d_worker = d_worker.detach().cpu().numpy()
        mask = d > 0.0
        mask_worker = d_worker > 0.0

        pts_shape = np.stack([g.detach().cpu().numpy()[mask] for g in grid], axis=-1)
        pts_worker = np.stack([g.detach().cpu().numpy()[mask_worker] for g in grid], axis=-1)
        d = d[mask]

        scale = np.mean(dims)

        plotter = pv.Plotter()
        plotter.add_axes()
        plotter.add_points(
            pts_shape,
            scalars=d,
            style='points_gaussian',
            render_points_as_spheres=True,
            point_size=10 / scale,
            show_scalar_bar=False,
            )
        plotter.add_points(
            pts_worker,
            color='red',
            style='points_gaussian',
            render_points_as_spheres=True,
            point_size=10 / scale,
        )
        plotter.show()

    def save_screenshot(self, postfix: str = ''):
        """Save a screenshot of the current shape."""
        dims = self.bounds[1] - self.bounds[0]
        self.shape.pv_plot(self.center, dims * 1.1, pts_per_dim=120, show=False,
                           save_at=PLOTS / 'addremove' / f'{self.name}{postfix}.eps')

    def _add_material(self):
        """Add material to the shape."""
        self.workers = Worker(self._sample(interior=True), radius=self.worker_radius)
        self.workers.enable_addition()
        self._set_worker_active()
        self.optimization_cycle()

    def _remove_material(self):
        """Add material to the shape."""
        self.workers = Worker(self._sample(interior=False), radius=self.worker_radius)
        self.workers.enable_carving()
        self._set_worker_active()
        self.optimization_cycle()

    def _check_bounds(self):
        """Makes sure the workers are not working outside their bounds."""
        if self.bounds is None or self.workers.carve_material:
            return

        if not self._get_workers_in_bounds(add_phase_threshold=1.)[self.workers.active].all():
            raise FleeingWorkerException("Workers are out of bounds.")

    def _get_workers_in_bounds(self, add_phase_threshold: float = 0.95):
        """Checks which of the current workers are within the bounds."""
        outer_positions = self.workers.center
        bounds = self.bounds
        if self.workers.add_material:  # Worker body contributes towards shape
            outer_positions = outer_positions + torch.sign(self.workers.center) * self.workers.radius
            bounds = bounds * add_phase_threshold
        return torch.logical_and((outer_positions > bounds[0]).all(dim=1), (outer_positions < bounds[1]).all(dim=1))

    def _set_worker_active(self, add_phase_threshold: float = 0.95):
        """Set the active workers to be within the bounds."""
        self.workers.active = self._get_workers_in_bounds(add_phase_threshold=add_phase_threshold)

    def _sample(self, interior: bool = True) -> torch.Tensor:
        """Sample points to spawn the workers."""
        dim = self.bounds[1] - self.bounds[0]
        surf, normals = self.shape.get_surface_points(self.center, dim, 9)
        d = self.shape(surf)
        if self.random_sampling:
            on_surface = torch.abs(d) < 1e-6
            if on_surface.sum() > self.max_workers:
                candidates = torch.arange(surf.shape[0])[on_surface]
                selection = candidates[torch.randperm(candidates.shape[0])][:self.max_workers]
            else:
                selection = torch.argsort(torch.abs(d))[:self.max_workers]
        else:
            # Try to select evenly along the grid
            every_nth = surf.shape[0] // self.max_workers
            selection = torch.arange(surf.shape[0])[::every_nth][:self.max_workers]

        # Move the workers just to the inside/outside of the shape
        safety_factor = 1.1
        surface = surf[selection]
        if interior:
            workers = surface + (self.worker_radius - d[selection]).unsqueeze(-1) * safety_factor * normals[selection]
        else:
            workers = surface - (self.worker_radius - d[selection]).unsqueeze(-1) * safety_factor * normals[selection]
        return workers



class XxRollingBehavior(AddRemoveAlgorithm):

    def loss(self):
        inertia = self.workers.get_worker_inertia(self.center)
        scores = self.workers.get_relevance_score(self.base_shape)
        workers_reward = (3 * inertia[:, 0, 0] - inertia[:, 1, 1] - inertia[:, 2, 2]) * scores
        reward = workers_reward.sum() * 1e3
        return -reward


def main():
    radius = 0.5
    shape_gt: BoundedSdf = SphereSdf(torch.tensor([[0., 0., 0., radius]]))
    # shape = remesh(shape_gt, origin, shape_gt.get_dims())
    shape = shape_gt

    max_workers = 3
    n_cycles: int = 5
    steps_per_iteration: int = 500
    bounds = torch.tensor([[-1., -1., -1.], [1., 1., 1.]])
    algo = XxRollingBehavior(shape, max_workers, 0.1, n_cycles, steps_per_iteration, bounds=bounds)
    algo.optimize()
    algo.visualize()
    print(algo.workers.aabb)
    # TODO: Can I somehow "close" the shape again?
    # TODO: Currently, there is a problem because the inertia decreases quadratically with the "relevance" that increases
    #   slower, so once the carving workers touch the surface, they refuse to go in and instead just grow.


if __name__ == "__main__":
    torch.set_default_device('cuda')
    torch.manual_seed(0)
    main()