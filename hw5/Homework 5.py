import numpy as np
# import jax.numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

VORTEX_CORE_RADIUS = 1

# N = 5
# locations = np.array([[10, 10], [30, 30], [50, 50], [70, 70], [90, 90]], dtype='float64').T
# circulations = np.arange(N, dtype='float64') * 10

### Vortices
# Number of vortices
N = 2
# Vortex locations: 
locations = np.array([[20, 20], [45, 55]], dtype='float64').T
circulations = np.array([-1, 1], dtype='float64')

def u_theta(r, vorticity=1):
    C = 30
    return np.where(r <= VORTEX_CORE_RADIUS, 0.5 * r * vorticity, np.where(r != 0, C * np.sign(vorticity) / r, 0))

def velocity_from_vortex(v_x, v_y, vorticity, x, y):
    distance = np.sqrt((x - v_x)**2 + (y - v_y)**2)
    tangential_vel = u_theta(distance, vorticity)

    angle = np.pi/2 + np.arctan2(y - v_y, x - v_x)
    return tangential_vel * np.array([np.cos(angle), np.sin(angle)]) / np.sqrt(2)

velocity_from_vortex(*locations[0], circulations[0], *locations[1])

X, Y = np.meshgrid(np.linspace(0, 100, 25, dtype='float64'), np.linspace(0, 100, 25, dtype='float64'))
U = np.zeros_like(X)
V = np.zeros_like(Y)

dt = 0.1

fig, ax = plt.subplots(1,1)
Q = ax.quiver(X, Y, U, V, pivot='mid', color='gray', angles='xy', scale_units='xy', units='xy', scale=30, minshaft=5)
pts = ax.scatter(*locations.T)

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

def update_quiver(num, pts, Q):
    """updates the horizontal and vertical vector components by a
    fixed increment on each frame
    """
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i, vortex_i in enumerate(locations):
        for j, vortex_j in enumerate(locations):
            if i == j:
                continue

            locations[i] += dt * velocity_from_vortex(*vortex_j, circulations[j], *vortex_i)

        vel = velocity_from_vortex(*vortex_j, circulations[j], X, Y)
        U += vel[0]
        V += vel[1]

    Q.set_UVC(U,V)
    pts.set_offsets(locations.T)

    return pts, Q,

# you need to set blit=False, or the first set of arrows never gets
# cleared on subsequent frames
anim = animation.FuncAnimation(fig, update_quiver, fargs=(pts, Q),
                               interval=50, blit=False, cache_frame_data=False)
fig.tight_layout()
plt.show()

