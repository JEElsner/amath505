import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

from hw5 import Vortex, VortexManager

@pytest.fixture
def manager() -> VortexManager:
    return VortexManager()

def test_add_vortex(manager):
    v1 = Vortex(1, 2, 3, manager=manager)
    assert v1.x == 1
    assert v1.y == 2
    assert v1.circulation == 3
    assert v1._i == 0

    v2 = Vortex(4, 5, 6, manager=manager)
    assert v2.x == 4
    assert v2.y == 5
    assert v2.circulation == 6
    assert v2._i == 1

def test_tangential_vel(manager):
    # See https://en.wikipedia.org/wiki/Rankine_vortex
    circulation = 10
    v1 = Vortex(0, 0, circulation, core_radius=1, manager=manager)

    radius = 0.5
    assert v1.tangential_vel(radius) == radius * circulation / (2 * np.pi)

    radius = 2
    assert v1.tangential_vel(radius) == circulation / (radius * 2 * np.pi)
    
    radii = np.arange(2,  12)
    vels = circulation / (radii * 2 * np.pi)
    assert_almost_equal(vels, v1.tangential_vel(radii))
    
    # Need to vectorize this because otherwise we just get a divide-by-zero error
    radius = np.zeros(1)
    assert v1.tangential_vel(radius)[0] == 0
    
def test_cartesian_velocity(manager):
    v1 = Vortex(0, 0, 1, 1, manager=manager)
    vel = v1.velocity_from_vortex(1, 0)
    assert_almost_equal(np.array([0, 1]), vel / np.linalg.norm(vel))

    vel = v1.velocity_from_vortex(0, 1)
    assert_almost_equal(np.array([-1, 0]), vel / np.linalg.norm(vel))

    angles = np.linspace(0, 2 * np.pi, 100)
    xs = np.cos(angles)
    ys = np.sin(angles)
    
    res = v1.velocity_from_vortex(xs, ys)
    assert res.shape == (2, 100)
    res /= np.linalg.norm(res, axis=0)
    assert_almost_equal(np.vstack((-ys, xs)), res)

# @pytest.mark.skip(reason="Other tests expand the size of the array, so when this test is run by itself, it works, but as a whole it doesn't. To fix, we need to make a VortexManager class, but that's too much work right now.")   
def test_positions_vector(manager):
    assert manager.positions.shape == (2, manager.default_size)

# @pytest.mark.skip(reason="Same reason as `test_positions_vector`")    def test_ensure_space(manager):
    lst = [Vortex(i, i, i, manager=manager) for i in range(manager.default_size + 1)]

    assert manager.positions.shape == (2, 2*manager.default_size)

