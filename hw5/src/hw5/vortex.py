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
    def __init__(self, default_size=1, max_deleted=10):
        self.default_size = default_size
        self.max_deleted = max_deleted
        self.deleted: Set[int] = set()
        
        self.index_mapping: Dict[int, Vortex] = dict()

        # rows are coordinate axes, columns are coordinate pairs
        self.positions = np.zeros((2, default_size), dtype=float)
        self.circulations = 0 * np.ones(default_size, dtype=float)
        self.core_radii = np.zeros(default_size, dtype=float)
        self.end = 0

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
    
    def velocity_at(self, x: ArrayLike[np.float64], y: ArrayLike[np.float64]):
        # xs, ys = np.meshgrid(self.positions[0, :], self.positions[1, :])
        return velocity_from_vortex(*self.non_nan_positions, self.non_nan_circulations, x, y)
    
    def plot_positions(self, ax, *args, **kwargs):
        return ax.scatter(*self.non_nan_positions, *args, **kwargs)

DEFAULT_VORTEX_MANAGER = VortexManager()

class Vortex:
    """Simple model of a Rankine Vortex.
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
        # TODO: may need to guard against the case where r = 0
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
