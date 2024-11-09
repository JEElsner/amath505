import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation, colors

from . import Vortex, VortexManager

matplotlib.use('tkagg')

rng = np.random.default_rng()

fig, ax = plt.subplots(figsize=(20,12))

manager = VortexManager()

vortices = [manager.add(rng.random()*10, rng.random()*10, 5 * np.sign(rng.random() - 0.5)) for i in range(10)]
# vortices = [
#     manager.add(1, 1, 2),
#     manager.add(7, 7, 2),
#     # manager.add(1, 7, -2)
# ]
# vortices = [manager.add(7.7777777, 7.77777, 10)]

x = np.linspace(0, 10, 50)
y = np.linspace(0, 10, 50)

xs, ys = np.meshgrid(x, y)

vel_field = manager.velocity_at(xs, ys)
norms = np.linalg.norm(vel_field, axis=0)

# vel_field /= np.exp(norms)
vel_field /= norms
# vel_field *= np.log(norms) / norms

# ax.contourf(xs, ys, norms, norm=colors.LogNorm())
ax.contourf(xs, ys, np.log(norms))
ax.quiver(xs, ys, *vel_field, angles='xy', scale_units='xy', units='xy', minshaft=2, pivot='tail')
manager.plot_positions(ax, color='r')


plt.show()
