import numpy as np

from functools import partial

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation, colors

from . import Vortex, VortexManager

rng = np.random.default_rng()

matplotlib.use('tkagg')
fig, ax = plt.subplots(figsize=(20,12))

N = 1
manager = VortexManager(default_size=N, boundaries={
    'left': 0,
    'right': 10,
    'bottom': 0,
    'top': 10
})
# vortices = [manager.add(rng.random()*10, rng.random()*10, 2 * np.sign(rng.random() - 0.5)) for i in range(N)]
manager.add(1, 1, 2)

dt = 0.1
x = np.linspace(0, 10, 50)
y = np.linspace(0, 10, 50)

xs, ys = np.meshgrid(x, y)

vel_field = manager.velocity_at(xs, ys)
norms = np.linalg.norm(vel_field, axis=0)

vel_field /= norms

field = ax.quiver(xs, ys, *vel_field, norms, cmap='plasma', norm=colors.LogNorm(), angles='xy', scale_units='xy', units='xy', minshaft=2, pivot='tail')
positions = manager.plot_positions(ax, color='tab:red')
# m_positions = manager.mirror_vortices().plot_positions(ax, color='tab:orange')


def update(frame, ax, field, positions, m_positions, manager: VortexManager):
    """Update animation on each frame
    
    Args:
        frame: index of the frame being rendered
        ax: The axis on which the animation is plotted
        contour: The velocity field magnitude contour
        field: The velocity field directions quiver
        positions: The scatter plot of vortex positions
        m_positions: Positions of the mirror vortices
        manager: VortexManager for the vortices
    """

    manager.advect(dt, mirror_vortices=True)

    velocities = manager.velocity_at(xs, ys)
    norms = np.linalg.norm(velocities, axis=0)
    velocities /= norms

    field.set_UVC(*velocities, norms)
    # positions = manager.plot_positions(ax, color='r')
    positions.set_offsets(manager.positions.T)
    # m_positions.set_offsets(manager.mirror_vortices().positions.T)
    
    return field, positions, manager


anim = animation.FuncAnimation(
    fig,
    partial(update, ax=ax, field=field, positions=positions, m_positions=None, manager=manager),
    # frames=np.arange(0, 65),
    repeat=False,
    cache_frame_data=False,
    interval=50
)

# anim.save('./success.gif')

plt.show()
