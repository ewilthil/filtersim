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
nav_sys = navigation.NavigationSystem(time, dt_imu, dt_gps, x0, imu=navigation.unbiased_imu)
# Set up ownship
ownship = shipmodels.Ownship(time, model, x0, nav_sys=nav_sys)
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

ang_fig, ang_ax = plt.subplots(ncols=3)
ownship.plot_angles(ang_ax)
ownship.nav_sys.plot_angles(ang_ax)

vel_fig, vel_ax = plt.subplots(ncols=3)
ownship.plot_velocity(vel_ax)
ownship.nav_sys.plot_velocity(vel_ax)

pos_fig, pos_ax = plt.subplots()
ownship.plot_position(pos_ax)
ownship.nav_sys.plot_position(pos_ax)

ownship.nav_sys.plot_errors()

ownship.nav_sys.plot_innovations()

plt.show()
