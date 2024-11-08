import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

from . import Vortex

matplotlib.use('tkagg')

rng = np.random.default_rng()

fig, ax = plt.subplots(figsize=(20,12))


vortices = [Vortex(rng.random()*10, rng.random()*10, 5 * np.sign(rng.random() - 0.5)) for i in range(10)]
# vortices = [
#     Vortex(1, 1, 2),
#     Vortex(7, 7, 2),
#     # Vortex(1, 7, -2)
# ]
# vortices = [Vortex(7.7777777, 7.77777, 10)]

x = np.linspace(0, 10, 50)
y = np.linspace(0, 10, 50)

xs, ys = np.meshgrid(x, y)
vel_field = np.zeros((2, len(x), len(y)))

for v in vortices:
    vel_field += v.velocity_from_vortex(xs, ys)

ax.quiver(xs, ys, *vel_field, angles='xy', scale_units='xy', units='xy', minshaft=2, pivot='mid')
Vortex.plot_positions(ax, color='r')

print(Vortex.circulations)

plt.show()
