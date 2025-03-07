from dataclasses import dataclass
from typing import Tuple, Union

from curobo.geom.types import Mesh
import numpy as np
import torch
from torch.autograd import Function
from tqdm import tqdm

from mcfly.utilities.sdf import BoundedSdf, CuroboMeshSdf, Sdf, SphereSdf

@dataclass
class Worker:

    center: torch.Tensor
    radius: Union[torch.Tensor, float]
    density: float = 1.

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
        return (4. / 3.) * torch.pi * self.radius ** 3

    @property
    def sdf(self):
        return SphereSdf(self.query_shapes.detach().clone())

    def get_exterior_score(self, shape: Sdf) -> torch.Tensor:
        """
        Computes the exterior score of the worker with respect to the shape.
        """
        # TODO: Add meaningful weighting to the distance
        # TODO: How do I want to get the gradients of the distance computation? Automatically? Implement a manual backprop?
        d, _ = QuerySdf.apply(self.query_shapes, shape)
        debug, debug = shape(self.query_shapes, gradients=True)
        weight =  1 / self.query_shapes[..., 3]
        return torch.sigmoid(-d * weight)


    def work_on_shape(self, shape: Sdf) -> Sdf:
        """Add the workers body to the shape if it is in the exterior."""
        # TODO: Or remove it when it is in the interior
        mask = shape(self.query_shapes) < self.radius.squeeze()
        if mask.sum() == 0:
            return shape
        worker_sdf = SphereSdf(self.query_shapes.detach().clone()[mask])
        return shape.boolean_union(worker_sdf)
    

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

def parallel_axis_theorem(i_com: torch.Tensor, m: torch.Tensor, d: torch.Tensor):
    """Computes the inertia of a body with translated by d using the parallel axis theorem."""
    eye = torch.eye(3).to(d).unsqueeze(0).repeat(d.shape[0], 1, 1)
    return i_com + m.unsqueeze(-1) * (torch.norm(d, dim=1).unsqueeze(-1).unsqueeze(-1) ** 2 * eye
                                      - torch.einsum('bi,bj->bij', d, d))

def worker_inertia(worker: Worker, origin: torch.Tensor) -> torch.Tensor:
    """Compute the approximated inertia that a worker adds to the shape."""
    d = worker.center - origin
    return parallel_axis_theorem(worker.com_inertia, worker.mass, d)

def remesh(shape: Sdf, center: torch.Tensor, dims: torch.Tensor) -> CuroboMeshSdf:
    """Resamples the surface and returns the mesh for the new shape."""
    v, f = shape.reconstruct_surface_poisson(center, dims, refinement_steps=7)
    return CuroboMeshSdf.from_meshes([Mesh(name='o', vertices=v.astype(np.float32).tolist(),
                                           faces=f.tolist(), pose=[0., 0., 0., 1., 0., 0., 0.])],
                                     max_distance=dims.max().item())

def main():
    radius = 0.5
    shape_gt: BoundedSdf = SphereSdf(torch.tensor([[0., 0., 0., radius]]))
    origin = shape_gt.get_center()
    # shape = remesh(shape_gt, origin, shape_gt.get_dims())
    shape = shape_gt

    max_workers = 50
    n_cycles: int = 1
    steps_per_iteration: int = 5000

    for cycle in range(n_cycles):
        workers = Worker(shape.sample_interior(max_points=max_workers), radius=.1)
        optim = torch.optim.SGD([workers.center], lr=1e-3, maximize=True)

        for step in tqdm(range(steps_per_iteration)):
            optim.zero_grad()
            inertia = worker_inertia(workers, origin)
            scores = workers.get_exterior_score(shape)
            workers_reward = (3 * inertia[:, 0, 0] - inertia[:, 1, 1] - inertia[:, 2, 2]) * scores
            reward = workers_reward.sum() * 1e3
            reward.backward()

            torch.nn.utils.clip_grad_norm_(workers.center, 5.0)
            optim.step()
            shape = workers.work_on_shape(shape)


            if workers.center.max() > 0.8:
                print("Worker out of bounds, stopping optimization.")
                break

        worker_dims = workers.aabb[1] - workers.aabb[0]
        dims = torch.maximum(shape_gt.get_dims(), worker_dims)
        shape.pv_plot(shape_gt.get_center(), dims, pts_per_dim=100)
        print(workers.center)
        # shape = remesh(shape, origin, dims)


if __name__ == "__main__":
    torch.set_default_device('cuda')
    main()