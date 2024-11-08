from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

class VortexManager:
    def __init__(self, default_size=1, max_deleted=10):
        self.default_size = default_size
        self.max_deleted = max_deleted
        self.deleted = set()
        
        self.index_mapping = dict()

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
    
    @property
    def non_nan_positions(self) -> NDArray[np.float64]:
        return self.positions[:, ~np.isnan(self.circulations)]
    
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
        # TODO: I think we need to change np.sign(vorticity) to something useful once we actually calculate vorticity
        # TODO: may need to guard against the case where r = 0
        return self.circulation * np.where(r <= self.core_radius, r * self.core_radius**2, 1/r) / (2 * np.pi)
    
    def velocity_from_vortex(self, x, y):
        distance = np.sqrt((x - self.x)**2 + (y - self.y)**2)
        tangential_vel = self.tangential_vel(distance)

        angle = np.pi/2 + np.arctan2(y - self.y, x - self.x)
        return tangential_vel * np.array([np.cos(angle), np.sin(angle)]) / np.sqrt(2)
    
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

    def __del__(self):
        # Really we should shift all the vortices down an index so that we don't infinitely expand memory, but that's a problem for later.
        self.manager.circulations[self._i] = np.nan
