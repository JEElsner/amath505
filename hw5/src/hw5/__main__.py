import numpy as np

from functools import partial

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation, colors

from . import Vortex, VortexManager

rng = np.random.default_rng()

matplotlib.use('tkagg')
fig, ax = plt.subplots(figsize=(20,12))

N = 10
manager = VortexManager(default_size=N)
vortices = [manager.add(rng.random()*10, rng.random()*10, 2 * np.sign(rng.random() - 0.5)) for i in range(N)]

dt = 0.1
x = np.linspace(0, 10, 50)
y = np.linspace(0, 10, 50)

xs, ys = np.meshgrid(x, y)

vel_field = manager.velocity_at(xs, ys)
norms = np.linalg.norm(vel_field, axis=0)

vel_field /= norms

field = ax.quiver(xs, ys, *vel_field, norms, cmap='plasma', norm=colors.LogNorm(), angles='xy', scale_units='xy', units='xy', minshaft=2, pivot='tail')
positions = manager.plot_positions(ax, color='r')


def update(frame, ax, field, positions, manager: VortexManager):
    """Update animation on each frame
    
    Args:
        frame: index of the frame being rendered
        ax: The axis on which the animation is plotted
        contour: The velocity field magnitude contour
        field: The velocity field directions quiver
        positions: The scatter plot of vortex positions
        manager: VortexManager for the vortices
    """

    manager.advect(dt)

    velocities = manager.velocity_at(xs, ys)
    norms = np.linalg.norm(velocities, axis=0)
    velocities /= norms

    field.set_UVC(*velocities, norms)
    # positions = manager.plot_positions(ax, color='r')
    positions.set_offsets(manager.positions.T)
    
    return field, positions, manager


anim = animation.FuncAnimation(
    fig,
    partial(update, ax=ax, field=field, positions=positions, manager=manager),
    # frames=np.arange(0, 65),
    repeat=False,
    cache_frame_data=False,
    interval=50
)

# anim.save('./success.gif')

plt.show()
