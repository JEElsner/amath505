import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation, colors

from functools import partial

from . import VortexManager

def update(frame, dt, xs, ys, field, positions, manager: VortexManager):
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
    
    return dt, xs, ys, field, positions, manager

def animate(manager, grid_x, grid_y, dt, save_loc=None, duration=None, **kwargs):
    matplotlib.use('tkagg')
    fig, ax = plt.subplots(figsize=(20,12))

    xs, ys = np.meshgrid(grid_x, grid_y)

    vel_field = manager.velocity_at(xs, ys)
    norms = np.linalg.norm(vel_field, axis=0)

    vel_field /= norms

    field = ax.quiver(xs, ys, *vel_field, norms, cmap='plasma', norm=colors.LogNorm(), angles='xy', scale_units='xy', units='xy', minshaft=2, pivot='tail')
    positions = manager.plot_positions(ax, color='tab:red')
    
    # Set some defaults for animation arguments
    kwargs['repeat'] = kwargs.get('repeat', False)
    kwargs['cache_frame_data'] = kwargs.get('cache_frame_data', False)
    kwargs['interval'] = kwargs.get('interval', 50)
    
    if duration is not None:
        kwargs['frames'] = np.arange(duration)

    anim = animation.FuncAnimation(
        fig,
        partial(update, dt=dt, xs=xs, ys=ys, field=field, positions=positions,  manager=manager),
        **kwargs
    )

    if save_loc is not None:
        anim.save(save_loc)
    else:
        plt.show()