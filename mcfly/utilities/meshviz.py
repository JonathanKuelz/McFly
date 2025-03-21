from typing import Optional, Sequence

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch
from trimesh import Trimesh


def animate(meshes: Sequence[Trimesh],
            grid: Sequence[torch.Tensor],
            save_path: str,
            dt: float = 0.1):
    """Creates an animation by plotting the meshes one after the other."""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(i):
        ax.clear()
        mesh = Poly3DCollection(meshes[i].vertices[meshes[i].faces], alpha=0.8)
        mesh.set_facecolor('cyan')
        mesh.set_edgecolor('black')
        ax.add_collection3d(mesh)
        setup_axis(ax, grid)

    ani = FuncAnimation(fig, update, frames=len(meshes), interval=dt * 1000)
    ani.save(save_path)

def show_hist(history: Sequence[Trimesh],
              grid: torch.Tensor,
              max_columns: int = 4,
              title: Optional[str] = None,
              ):
    """Visualizes the history of meshes."""
    history = tuple(history)
    n = len(history)
    rows = np.ceil(n / max_columns).astype(int)
    cols = min(max_columns, n)

    fig = plt.figure(figsize=(4 + cols * 3.5, 5 + rows * 3.5))
    for i, mesh in enumerate(history):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        verts = mesh.vertices
        faces = mesh.faces
        mesh = Poly3DCollection(verts[faces], alpha=0.8)
        mesh.set_edgecolor('black')
        mesh.set_facecolor('cyan')
        ax.add_collection3d(mesh)
        setup_axis(ax, grid)

    if title is not None:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.show()

def setup_axis(ax, grid: torch.Tensor):
    """Sets up the axis for the 3D plot."""
    x_min, x_max = grid[0].min().item(), grid[0].max().item()
    y_min, y_max = grid[1].min().item(), grid[1].max().item()
    z_min, z_max = grid[2].min().item(), grid[2].max().item()

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')