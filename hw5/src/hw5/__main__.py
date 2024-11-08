import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

from . import Vortex

matplotlib.use('tkagg')

ax = plt.gca()

vortices = [Vortex(i, i, i) for i in range(10)]

Vortex.plot_positions(ax)

plt.show()
