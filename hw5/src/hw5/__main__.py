import numpy as np

from . import VortexManager, animate

import matplotlib.pyplot as plt

rng = np.random.default_rng()

N = 3
manager = VortexManager(default_size=N, boundaries={
    'left': 0,
    'right': 10,
    'bottom': 0,
    'top': 10
})
vortices = [manager.add(rng.random()*10, rng.random()*10, 2 * np.sign(rng.random() - 0.5)) for i in range(N)]
# manager.add(1, 1, 2)

dt = 0.1
x = np.linspace(0, 10, 50)
y = np.linspace(0, 10, 50)

animate(manager, x, y, dt)
