import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_SIZE = 1

class Vortex:
    """Simple model of a Rankine Vortex.
    """
    
    # rows are coordinate axes, columns are coordinate pairs
    positions = np.zeros((2, DEFAULT_SIZE), dtype=float)
    circulations = np.nan * np.ones(DEFAULT_SIZE, dtype=float)
    core_radii = np.zeros(DEFAULT_SIZE, dtype=float)
    end = 0
    
    @staticmethod
    def plot_positions(ax, *args, **kwargs):
        ax.scatter(*Vortex.positions, *args, **kwargs)

    def __init__(self, x: float, y: float, circulation: float, core_radius: float=1, name=""):
        # Ensure there is enough room in the arrays
        if Vortex.end == Vortex.positions.shape[1]:
            Vortex.positions = np.concat((Vortex.positions, np.zeros_like(Vortex.positions)), axis=1)
            Vortex.circulations = np.concat((Vortex.circulations, np.zeros(len(Vortex.circulations))))
            Vortex.core_radii = np.concat((Vortex.core_radii, np.zeros_like(Vortex.core_radii)))

        self._i = Vortex.end
        Vortex.end += 1
        
        Vortex.positions[:, self._i] = [x, y]
        Vortex.circulations[self._i] = circulation
        Vortex.core_radii[self._i] = core_radius

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
        return Vortex.positions[0, self._i]
    
    @x.setter
    def x(self, val: float):
        Vortex.positions[0, self._i] = val
        
    @property
    def y(self) -> float:
        return Vortex.positions[1, self._i]
    
    @y.setter
    def y(self, val: float) -> float:
        Vortex.positions[1, self._i] = val
        
    @property
    def position(self) -> NDArray[np.float64]:
        return Vortex.positions[:, self._i]
        
    @property
    def circulation(self) -> float:
        return Vortex.circulations[self._i]
    
    @circulation.setter
    def circulation(self, val: float):
        Vortex.circulations[self._i] = val
        
    @property
    def core_radius(self) -> float:
        return Vortex.core_radii[self._i]
    
    @core_radius.setter
    def core_radius(self, val: float):
        Vortex.core_radii[self._i] = val

    def __del__(self):
        # Really we should shift all the vortices down an index so that we don't infinitely expand memory, but that's a problem for later.
        Vortex.circulations[self._i] = np.nan
