from math import sqrt

import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp

from matplotlib import pyplot as plt
from matplotlib import animation as animation
from matplotlib.animation import PillowWriter

### CONSTANTS

# speed of light
c = 1

# initial endpoints of Rod
A = np.array([1, 5, 0])
B = np.array([9, 5, 0])

def curve(x, t):
    # parametrized curve based on the vector parameter x at time t
    return A + x[0] * (B - A)


def v(t, x):
    # velocity at any time (can be function of x)
    if (t < 10):
        return np.array([c/8, -c, 0])
    else:
        return np.array([c/8, c, 0])


# times of simulation
t_eval = np.linspace(0, np.linalg.norm(B)/c + 30, 100)

# times of calculation of integral of v
v_t = np.linspace(t_eval[0], t_eval[-1], 3000)


### VALUES DEPENDENT ON CONSTANTS

# the integral of v evaluated at each point in v_t
vIntegral = solve_ivp(v, t_span=(v_t[0], v_t[-1]), t_eval=v_t, y0=np.array([0, 0, 0])).y.T


def realPos(x, t):
    # the real position of a particle at time t
    return curve(x, t) + vIntegral[(np.abs(v_t - t)).argmin()]


def photon(x, t):
    # the time at which a photon emmited by particle at time t will reach observer
    return t + np.linalg.norm(realPos(x, t)) / c


def photonInv(x, t):
    # the time at which photon was sent given the time t it reached
    return root_scalar(lambda t0: photon(x, t0) - t, x0=1, x1=2).root


def apparentPos(x, t):
    # the apparent position of a particle at time t
    return realPos(x, photonInv(x, t))


fig, ax = plt.subplots()

realLineAnim = [[realPos([x], t_eval[i]) for x in np.linspace(0, 1, 100)] for i in range(len(t_eval))]
apparentLineAnim = [[apparentPos([x], t_eval[i]) for x in np.linspace(0, 1, 100)] for i in range(len(t_eval))]

realPath, = ax.plot([p[0] for p in realLineAnim[0]], [p[1] for p in realLineAnim[0]])
apparentPath, = ax.plot([p[0] for p in apparentLineAnim[1]], [p[1] for p in apparentLineAnim[1]])


def simulate(frame):
    realPath.set_data([p[0] for p in realLineAnim[frame]], [p[1] for p in realLineAnim[frame]])
    apparentPath.set_data([p[0] for p in apparentLineAnim[frame]], [p[1] for p in apparentLineAnim[frame]])
    return realPath, apparentPath



# axis of the graph
axis = [0, 15, -6, 12]

ax.axis(axis)
anim = animation.FuncAnimation(fig, simulate, frames=len(t_eval), blit=True)
anim.save("wierdLight.gif", dpi=300, writer=PillowWriter(fps=25))
# plt.show()
