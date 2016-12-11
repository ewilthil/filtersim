import numpy as np
import matplotlib.pyplot as plt
from filtersim import shipmodels, navigation
T = 300
dt = 0.01
time = np.linspace(0, T, (T+dt)/dt, endpoint = True)
x0 = np.zeros(18)
x0[6] = 10
model = shipmodels.NonlinearStochasticModel()
# Set up navigation system
dt_imu = dt
dt_gps = 20*dt
nav_sys = navigation.NavigationSystem(time, x0)
# Set up ownship
ownship = shipmodels.Ownship(time, model, x0)
def heading_ref(t):
    if t < 150:
        return 0
    elif t > 170:
        return np.deg2rad(45)
    else:
        return np.deg2rad((t-150)*2.25)
def surge_ref(t):
    return 10
for t_idx, t in enumerate(time):
    ownship.step(t_idx, surge_ref(t), heading_ref(t))
    spec_force, ang_rate = ownship.imu_states(t_idx)
    nav_sys.step_strapdown(t_idx, spec_force, ang_rate)

_, ax = ownship.plot_position()
nav_sys.plot_position(ax)
plt.show()
