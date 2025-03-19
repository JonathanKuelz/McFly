import torch

from biff.optimization.addremove import AddRemoveAlgorithm
from mcfly.representations.sdf import BoundedSdf, SphereSdf


class XxRollingBehavior(AddRemoveAlgorithm):

    def loss(self):
        """
        A weighted sum of inertia terms. Those that contribute in a "good" way are pushed to become more relevant, those
        that contribute in a "bad" way are pushed to become less relevant.

        Workers can "contribute" (be in contact with) a shape, and/or have desireable properties or not.
        - Does not contribute: Push the worker closer to the surface
        - Contributes: If the worker is "good" (i.e. has a positive inertia), try increase its relevance. If it is "bad",
            prioritize its reward.
        """
        inertia = self.workers.get_worker_inertia(self.center)
        score = 1.2 * inertia[:, 0, 0] - inertia[:, 1, 1] - inertia[:, 2, 2]
        self.workers.score = score.detach()
        good_reward = self.workers.score > 0
        relevance, contributes = self.workers.get_relevance(self.last_shape)
        optimize_score = torch.logical_or(~contributes, good_reward)
        reward = relevance[optimize_score].sum() + score[~optimize_score].sum()
        return -reward


def main():
    radius = 0.5
    shape_gt: BoundedSdf = SphereSdf(torch.tensor([[0., 0., 0., radius]]))
    shape = shape_gt

    max_workers = 256
    n_cycles: int = 15
    steps_per_iteration: int = 250
    bounds = torch.tensor([[-1., -1., -1.], [1., 1., 1.]])
    algo = XxRollingBehavior(shape, max_workers, 0.1, n_cycles, steps_per_iteration, bounds=bounds,
                             viz_every=3)
    algo.optimize()
    algo.shape.plot_reconstructed_surface(algo.center, algo.bounds[1] - algo.bounds[0], refinement_steps=7)

if __name__ == "__main__":
    torch.set_default_device('cuda')
    torch.manual_seed(0)
    main()