from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from typing import Dict, Set

def tangential_vel(r, circulation, zero_cutoff=1e-10):
    """Compute the tangential component of velocity away from a vortex
    
    Args:
        r: The radius away from the vortex center
        circulation: The circulation of the vortex
        zero_cutoff: The tolerance under which to treat the radius as zero
        
    Returns:
        The tangential component of velocity at a distance `r` from a vortex
        with circulation `circulation`.
    """
    # TODO: I'm guessing this 'with' paradigm isn't very performant in loops
    with np.errstate(divide='ignore'):
        return np.where(r > 1e-10, circulation / (r * 2 * np.pi), 0)

def velocity_from_vortex(vortex_x, vortex_y, circulation, x, y):
    """
    Calculate the velocity at a point given the position and circulation of a
    vortex.

    Below: m is a scalar natural number, N is a (possibly) shape tuple, or a
    scalar.

    Args:
        vortex_x: (m,) x coordinates of vortices
        vortex_y: (m,) y coordinates of vortices
        circulation: (m,) circulations
        x: (N,) x points
        y: (N,) y points

    Returns:
        (2, N) matrix of velocities. In 2D, the first row is the x-coordinate,
        and the second row is the y-coordinate. If N is a shape tuple, then the
        first array contains the x-coordinates and the second the y-coordinates.
    """

    # Vectorize everything!
    vortex_x = np.atleast_1d(np.array(vortex_x))
    vortex_y = np.atleast_1d(np.array(vortex_y))
    circulation = np.atleast_1d(np.array(circulation))
    x = np.atleast_1d(np.array(x))
    y = np.atleast_1d(np.array(y))

    # x - vortex_x: (m, N)
    x_dist = x - vortex_x[(...,) + (np.newaxis,) * x.ndim]
    # y - vortex_y: (m, N)
    y_dist = y - vortex_y[(...,) + (np.newaxis,) * y.ndim]

    # (m, N) -> (m, N)
    distance = np.sqrt(x_dist**2 + y_dist**2)
    # dist: (m, N); circ: (m,) -> vel: (m, N)
    speed = tangential_vel(distance, circulation[(...,) + (np.newaxis,) * x.ndim])
    # (m, N) -> (m, N)
    angle = np.arctan2(y_dist, x_dist)
    # speed: (m, N); angle: (m, N) -cos/sin-> (2, m, N) -sum-> (2, N)
    return np.sum(speed * np.array([-np.sin(angle), np.cos(angle)]) / np.sqrt(2), axis=1)

class VortexManager:
    def __init__(self, default_size=1, max_deleted=10, boundaries=None):
        self.default_size = default_size
        self.max_deleted = max_deleted
        self.deleted: Set[int] = set()
        
        self.index_mapping: Dict[int, Vortex] = dict()

        # rows are coordinate axes, columns are coordinate pairs
        self.positions = np.zeros((2, default_size), dtype=float)
        self.circulations = 0 * np.ones(default_size, dtype=float)
        self.core_radii = np.zeros(default_size, dtype=float)
        self.end = 0

        # boundaries for mirror vortices
        self.boundaries = boundaries

    def add(self, x: float, y: float, circulation: float, core_radius=1, name="") -> Vortex:
        return Vortex(x, y, circulation, core_radius, name, manager=self)
        
    def ensure_space(self):
        if self.end == self.positions.shape[1]:
            self.positions = np.concat((self.positions, np.zeros_like(self.positions)), axis=1)
            self.circulations = np.concat((self.circulations, 0 * np.ones(len(self.circulations))))
            self.core_radii = np.concat((self.core_radii, np.zeros_like(self.core_radii)))
            
    def register(self, vortex: Vortex) -> int:
        # Ensure there is enough room in the arrays
        self.ensure_space()
        
        # Add to index mapping registry
        self.index_mapping[self.end] = vortex

        # This is kinda goofy: add 1 and then subtract it
        self.end += 1
        return self.end - 1
    
    def deregister(self, vortex: Vortex | int) -> int:
        if isinstance(vortex, int):
            pass
        else:
            vortex = vortex._i

        self.circulations[vortex] = np.nan
        self.positions[:, vortex] = np.nan
        self.core_radii[vortex] = np.nan
        
        del self.index_mapping[vortex]._i
        del self.index_mapping[vortex]
        self.deleted.add(vortex)

        if len(self.deleted) > self.max_deleted:
            self.minimize_space()

    def minimize_space(self):
        raise NotImplementedError()
    
    @property
    def non_nan_positions(self) -> NDArray[np.float64]:
        return self.positions[:, ~np.isnan(self.circulations)]
    
    @property
    def non_nan_circulations(self) -> NDArray[np.float64]:
        return self.circulations[~np.isnan(self.circulations)]
    
    @staticmethod
    def __reflection_matrix(angle: float) -> NDArray[np.float64]:
        """2D Reflection matrix for reflection across axis with the given
        angle.
        """
        return np.array([[np.cos(2*angle), np.sin(2*angle)],
                         [np.sin(2*angle), -np.cos(2*angle)]])
        
    @staticmethod
    def reflect_across(pts, axis, angle):
        """Reflect points across an axis.

        Args:
            pts: (2, n) The points to reflect.
            axis: (2, 1) A point on the axis of reflection
            angle: The angle of the axis of reflection
            
        Returns:
            A (2, n) array of reflected points.
        """

        reflection = VortexManager.__reflection_matrix(angle)
        return reflection @ (pts - axis) + axis
    
    def mirror_vortices(self, recurse=1) -> VortexManager:
        l_mirror = self.reflect_across(self.positions, np.vstack([self.boundaries['left'], 0]), np.pi/2)
        r_mirror = self.reflect_across(self.positions, np.vstack([self.boundaries['right'], 0]), np.pi/2)
        
        b_mirror = self.reflect_across(self.positions, np.vstack([0, self.boundaries['bottom']]), 0)
        t_mirror = self.reflect_across(self.positions, np.vstack([0, self.boundaries['top']]), 0)

        mirrors = np.concat((l_mirror, r_mirror, b_mirror, t_mirror), axis=1)
        circulations = -np.repeat(self.circulations, 4)
        
        # I'm kind of bastardizing my own code here. VortexManager isn't
        # designed to be used like this
        mirror_manager = VortexManager(boundaries=self.boundaries)
        mirror_manager.positions = mirrors
        mirror_manager.circulations = circulations
        
        if recurse > 0:
            deeper = mirror_manager.mirror_vortices(recurse=recurse-1)

            mirror_manager.positions = np.concat((mirrors, deeper.positions), axis=1)
            mirror_manager.circulations = np.concat((circulations, deeper.circulations))
        
        return mirror_manager
    
    def velocity_at(self, x: ArrayLike[np.float64], y: ArrayLike[np.float64], mirror_vortices=False):
        # xs, ys = np.meshgrid(self.positions[0, :], self.positions[1, :])
        vel = velocity_from_vortex(*self.non_nan_positions, self.non_nan_circulations, x, y)
        
        if mirror_vortices:
            vel += self.mirror_vortices().velocity_at(x, y)

        return vel
    
    def advect(self, dt, mirror_vortices=True):
        """Move all the vortices based on their interactions with each other
        
        Args:
            dt: Timestep
        """
        self.positions += dt * self.velocity_at(*self.positions)
        
        if mirror_vortices and self.boundaries is not None:
            self.positions += dt * self.mirror_vortices().velocity_at(*self.positions)
            
    def step_forward(self, dt, steps):
        """Advect the vortices by a number of steps.

        Args:
            dt: The timestep to use for advection
            steps: The number of times to step forward

        Returns:
            An `(n, 2, steps+1)` array of points, where `n` is the number of
            points in the vortex manager. The second axis is x and y
            coordinates, and the last axis is the time. This works well for
            plotting, because you can iterate through the first dimension,
            calling `plot(*point)` on the remaining dimensions, which results in
            line plots of the positions of all the particles.
        """
        history = np.zeros((steps+1,) + self.positions.shape)
        history[0, :] = self.positions

        for i in range(1, steps+1):
            self.advect(dt)
            history[i, :] = self.positions
            
        # (times, 2, points) -> (points, 2, times)
        return np.transpose(history, (2, 1, 0))
    
    def plot_positions(self, ax, *args, **kwargs):
        return ax.scatter(*self.non_nan_positions, *args, **kwargs)

DEFAULT_VORTEX_MANAGER = VortexManager()

class Vortex:
    """Simple model of a Point Vortex.
    """

    def __init__(self, x: float, y: float, circulation: float, core_radius: float=1, name="", manager=None):
        if manager is None:
            global DEFAULT_VORTEX_MANAGER
            self.manager = DEFAULT_VORTEX_MANAGER
        else:
            self.manager = manager

        self._i = manager.register(self)
        
        manager.positions[:, self._i] = [x, y]
        manager.circulations[self._i] = circulation
        manager.core_radii[self._i] = core_radius

        self.name = name

    def tangential_vel(self, r):
        return tangential_vel(r, self.circulation)
    
    def velocity_from_vortex(self, x, y):
        return velocity_from_vortex(self.x, self.y, self.circulation, x, y)
    
    @property
    def vorticity(self):
        # TODO calculate vorticity
        return self.circulation
        
    @property
    def x(self) -> float:
        return self.manager.positions[0, self._i]
    
    @x.setter
    def x(self, val: float):
        self.manager.positions[0, self._i] = val
        
    @property
    def y(self) -> float:
        return self.manager.positions[1, self._i]
    
    @y.setter
    def y(self, val: float) -> float:
        self.manager.positions[1, self._i] = val
        
    @property
    def position(self) -> NDArray[np.float64]:
        return self.manager.positions[:, self._i]
        
    @property
    def circulation(self) -> float:
        return self.manager.circulations[self._i]
    
    @circulation.setter
    def circulation(self, val: float):
        self.manager.circulations[self._i] = val
        
    @property
    def core_radius(self) -> float:
        return self.manager.core_radii[self._i]
    
    @core_radius.setter
    def core_radius(self, val: float):
        self.manager.core_radii[self._i] = val
