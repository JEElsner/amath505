from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from typing import Dict, Set

def tangential_vel(r, circulation):
    # TODO: may need to guard against the case where r = 0
    return circulation / (r * 2 * np.pi)

def velocity_from_vortex(vortex_x, vortex_y, circulation, x, y):
    # Vectorize everything!
    vortex_x = np.array(vortex_x)[:, np.newaxis]
    vortex_y = np.array(vortex_y)[:, np.newaxis]
    circulation = np.array(circulation)[:, np.newaxis]
    x = np.array(x)
    y = np.array(y)

    distance = np.sqrt((x - vortex_x)**2 + (y - vortex_y)**2)
    speed = tangential_vel(distance, circulation)

    angle = np.pi/2 + np.arctan2(y - vortex_y, x - vortex_x)
    return np.sum(speed * np.array([np.cos(angle), np.sin(angle)]) / np.sqrt(2), axis=1)

class VortexManager:
    def __init__(self, default_size=1, max_deleted=10):
        self.default_size = default_size
        self.max_deleted = max_deleted
        self.deleted: Set[int] = set()
        
        self.index_mapping: Dict[int, Vortex] = dict()

        # rows are coordinate axes, columns are coordinate pairs
        self.positions = np.zeros((2, default_size), dtype=float)
        self.circulations = np.nan * np.ones(default_size, dtype=float)
        self.core_radii = np.zeros(default_size, dtype=float)
        self.end = 0

    def add(self, x: float, y: float, circulation: float, core_radius=1, name="") -> Vortex:
        return Vortex(x, y, circulation, core_radius, name, manager=self)
        
    def ensure_space(self):
        if self.end == self.positions.shape[1]:
            self.positions = np.concat((self.positions, np.zeros_like(self.positions)), axis=1)
            self.circulations = np.concat((self.circulations, np.nan * np.ones(len(self.circulations))))
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
    
    def velocity_at(self, x: ArrayLike[np.float64], y: ArrayLike[np.float64]):
        # xs, ys = np.meshgrid(self.positions[0, :], self.positions[1, :])
        return velocity_from_vortex(*self.positions, self.circulations, x, y)
    
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
