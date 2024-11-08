import numpy as np

DEFAULT_SIZE = 10

class Vortex:
    """Simple model of a Rankine Vortex.
    """

    # This is called a flyweight pattern, right?
    xs = np.zeros(DEFAULT_SIZE, dtype=float)
    ys = np.zeros(DEFAULT_SIZE, dtype=float)
    circulations = np.zeros(DEFAULT_SIZE, dtype=float)
    core_radii = np.zeros(DEFAULT_SIZE, dtype=float)
    end = 0
    
    @staticmethod
    def positions_vector():
        return np.vstack((Vortex.xs, Vortex.ys))

    def __init__(self, x: float, y: float, circulation: float, core_radius: float=1):
        # Ensure there is enough room in the arrays
        if Vortex.end == len(Vortex.xs):
            Vortex.xs = np.concat(Vortex.xs, np.zeros(len(Vortex.xs)))
            Vortex.ys = np.concat(Vortex.ys, np.zeros(len(Vortex.ys)))
            Vortex.circulations = np.concat(Vortex.circulations, np.zeros(len(Vortex.circulations)))
            Vortex.core_radii = np.concat(Vortex.core_radii, np.zeros_like(Vortex.core_radii))

        self._i = Vortex.end
        Vortex.end += 1
        
        Vortex.xs[self._i] = x
        Vortex.ys[self._i] = y
        Vortex.circulations[self._i] = circulation
        Vortex.core_radii[self._i] = core_radius

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
        return Vortex.xs[self._i]
    
    @x.setter
    def x(self, val: float):
        Vortex.xs[self._i] = val
        
    @property
    def y(self) -> float:
        return Vortex.ys[self._i]
    
    @y.setter
    def y(self, val: float) -> float:
        Vortex.ys[self._i] = val
        
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
        Vortex.circulations[self._i] = 0
