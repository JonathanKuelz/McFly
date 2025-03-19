from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union

from faker import Faker
import numpy as np
import pyvista as pv
import torch
from torch.autograd import Function
from tqdm import tqdm

from docbrown.templates.warp_sim import WarpSim
from mcfly.utilities.file_locations import PLOTS
from mcfly.representations.sdf import BoundedSdf, Sdf, SphereSdf


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
        self._score: torch.Tensor = torch.ones(self.num_workers).to(self.center.device)

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
    def score(self) -> torch.Tensor:
        """The individual score (think: reward) the worker."""
        return self._score[self.active]

    @score.setter
    def score(self, value: torch.Tensor):
        """Set the individual score (think: reward) the worker."""
        self._score[self.active] = value

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

    def get_relevance(self, shape: Sdf) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes a score indicating how much the worker is changing the material. This assumes that a worker maximally
        contributes to the shape if it is exactly at the surface of the shape. (which side depends on the mode)

        Args:
            shape (Sdf): The shape that the signed distance should be computed for. This should typicalle NOT be the
                shape that is changed in every iteration, because this would lead to unstable gradients.
        """
        d, _g = QuerySdf.apply(self.query_shapes, shape, shape, True)
        contributes = d > 0

        if self.add_material:
            optimal_d = torch.zeros_like(d)
        else:
            optimal_d = self.radius[self.active].squeeze()  # Intentionally just half interior to allow remeshing
        return 1 - (d - optimal_d) ** 2, contributes

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
        mask = torch.logical_and(mask, self.score > 0)
        if mask.sum() == 0:
            return shape
        worker_sdf = SphereSdf(self.query_shapes.detach().clone()[mask])
        return shape.boolean_union(worker_sdf)

    def _carve_material(self, shape: Sdf):
        """
        Remove the workers body from the shape if it is in the interior.

        Args:
            shape (Sdf): The shape to remove the workers body
        """
        mask = shape(self.query_shapes) > 0
        mask = torch.logical_and(mask, self.score > 0)
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
                 interactive: bool = False,
                 *,
                 perform_add_material: bool = True,
                 perform_carve_material: bool = True,
                 viz_every: int = 0,
                 get_optimizer: Callable[[AddRemoveAlgorithm, List[torch.Tensor]], torch.optim.Optimizer] = None,
                 worker_sampling: str = 'grid'
                 ):
        """
        Args:
            base_shape (BoundedSdf): The base shape to optimize.
            max_workers (int): The maximum number of workers.
            worker_radius (float): The radius of the workers that are going to add their body to the shape. The removing
                workers will have twice the radius, but are only allowed to enter the material with half their body.
            n_cycles (int): The number of add/remove cycles to run.
            steps_per_cycle (int): The number of steps per cycle.
            bounds (Optional[torch.Tensor]): The bounds for the workers. Should be 2x3 with the min and max bounds.
            perform_add_material (bool): If set to false, the "adding material" part of the algorithm is disabled.
            perform_carve_material (bool): If set to false, the "removing material" part of the algorithm is disabled.
            interactive (bool): If set to true, the algorithm will open intermediate visualizations. They need to be
                closed manually for the algorithm to continue.
            viz_every (int): The number of cycles to visualize the shape.
            get_optimizer (Callable[[List[torch.Tensor]], torch.optim.Optimizer]): A function that takes parameters
                and returns an optimizer.
            worker_sampling (str): The sampling strategy to use. Either 'grid' or 'random'.
        """
        self.base_shape = base_shape
        self.bounds = bounds
        self.center = self.base_shape.get_center()
        self.shape = base_shape
        self.last_shape = self.base_shape
        self.perform_add_material = perform_add_material
        self.perform_carve_material = perform_carve_material
        self.interactive = interactive
        self.viz_every = viz_every

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
            {'params': params[0], 'lr': 1e-3, 'name': 'center'},
            {'params': params[1], 'lr': 1e-4, 'name': 'radius'}
        ])

    def optimize(self):
        """Optimize the workers to add material to the shape."""
        for cycle in range(self.n_cycles):
            self.step_one_cycle(cycle)

    def step_one_cycle(self, cycle: int):
        print(f"Cycle {cycle + 1}/{self.n_cycles}")
        if self.perform_add_material:
            self._add_material()
            if self.interactive and self.viz_every > 0 and cycle % self.viz_every == 0:
                self.visualize_with_workers()
            self.shape = self.shape.remesh(self.center, self.bounds[1] - self.bounds[0], 7, limit_to='exterior')

        if self.perform_carve_material:
            self._remove_material()
            if self.interactive and self.viz_every > 0 and cycle % self.viz_every == 0:
                self.visualize_with_workers()
                self.shape.plot_reconstructed_surface(self.center, self.bounds[1] - self.bounds[0], refinement_steps=7)
            self.shape = self.shape.remesh(self.center, self.bounds[1] - self.bounds[0], 7, limit_to='interior')
        self.last_shape = self.shape


    def let_them_work(self):
        """Moves the workers to the exterior to expand the current shape."""
        optim = self.get_optimizer(self, self.workers.optimization_parameters)
        for step in tqdm(range(self.add_steps), desc=f"{'Adding' if self.workers.add_material else 'Removing'} material"):
            optim.zero_grad()
            loss = self.loss()
            if step == 0:
                tqdm.write('Initial scores: ' + str(self.workers.score.sum().item()))
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.workers.optimization_parameters, 1.)
            optim.step()
            self.shape = self.workers.work_on_shape(self.shape)
            self._set_worker_active()
        tqdm.write('EOC scores: ' + str(self.workers.score.sum().item()))

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
        self.let_them_work()

    def _remove_material(self):
        """Add material to the shape."""
        self.workers = Worker(self._sample(interior=False), radius=self.worker_radius * 2)
        self.workers.enable_carving()
        self._set_worker_active()
        self.let_them_work()

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
        if self.workers.add_material:
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
            # For factor 2 see docstring of this class
            workers = surface - (self.worker_radius * 2 - d[selection]).unsqueeze(-1) * safety_factor * normals[selection]
        return workers


class AddRemoveSimulateAlgorithm(AddRemoveAlgorithm):
    """An extension of the addremove algorithm that backpropagates gradients from a simulation."""

    def __init__(self, sim_kwargs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_sim_gradients = {}
        self.sim: Optional[WarpSim] = None
        self.sim_name = sim_kwargs.pop('name')
        self.sim_kwargs = sim_kwargs

    @abstractmethod
    def construct_builder(self):
        pass

    @abstractmethod
    def make_sim(self, cycle: int):
        """Create the simulation."""
        pass

    def optimize(self):
        for cycle in range(self.n_cycles):
            self.make_sim(cycle)
            self.sim.launch()
            self.latest_sim_gradients = self.get_sim_gradients()
            self.step_one_cycle(cycle)

    def let_them_work(self):
        """Moves the workers to the exterior to expand the current shape."""
        optim = self.get_optimizer(self, self.workers.optimization_parameters)
        for step in tqdm(range(self.add_steps), desc=f"{'Adding' if self.workers.add_material else 'Removing'} material"):
            optim.zero_grad()
            loss = self.loss()
            if step == 0:
                tqdm.write('Initial scores: ' + str(self.workers.score.sum().item()))
            if loss is not None:
                loss.backward()
            self.write_simulation_gradients()
            for param in self.workers.optimization_parameters:
                # Scale the parameters s.t. the worker with the largest gradient has gradnorm 1
                norms = param.grad.norm(dim=-1)
                param.grad = param.grad / norms.max()
            optim.step()
            self.shape = self.workers.work_on_shape(self.shape)
            self._set_worker_active()
        tqdm.write('EOC scores: ' + str(self.workers.score.sum().item()))

    @abstractmethod
    def get_sim_gradients(self) -> Dict[str, torch.Tensor]:
        """Get the shape gradients from the latest simulation."""
        pass

    def write_simulation_gradients(self):
        # TODO: Default + inheritance
        if 'inertia' in self.latest_sim_gradients:
            inertia = self.workers.get_worker_inertia(self.center)
            incoming_grads = self.latest_sim_gradients['inertia'].broadcast_to(inertia.shape)
            inertia.backward(incoming_grads)
        if 'mass' in self.latest_sim_gradients:
            mass = self.workers.mass
            incoming_grads = self.latest_sim_gradients['mass'].broadcast_to(mass.shape)
            mass.backward(incoming_grads)
        if 'com' in self.latest_sim_gradients:
            com = self.workers.center
            incoming_grads = self.latest_sim_gradients['com'].broadcast_to(com.shape)
            com.backward(incoming_grads)
