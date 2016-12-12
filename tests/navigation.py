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
M_imu = 1
M_gnss = 20
dt_imu = dt
dt_gnss = 20*dt
imu_time = np.linspace(0, T, int((T+dt_imu)/dt_imu+0.5), endpoint = True)
gnss_time = np.linspace(0, T, int((T+dt_gnss)/dt_gnss+0.5), endpoint = True)

# Set up ownship
ownship = shipmodels.Ownship(time, model, x0, imu=navigation.unbiased_imu, gnss=navigation.default_gnss)
strapdown = navigation.StrapdownSystem(imu_time, x0)
navigation_filter = navigation.NavigationFilter(gnss_time)
def heading_ref(t):
    if t < 150:
        return 0
    elif t > 170:
        return np.deg2rad(45)
    else:
        return np.deg2rad((t-150)*2.25)
def surge_ref(t):
    return 10
for time_idx, t in enumerate(time):
    imu_idx, rest_imu = int(np.floor(time_idx/M_imu)), np.mod(time_idx,M_imu)
    gnss_idx, rest_gnss = int(np.floor(time_idx/M_gnss)), np.mod(time_idx,M_gnss)
    ownship.step(time_idx, surge_ref(t), heading_ref(t))
    if rest_imu == 0:
        spec_force, ang_rate = ownship.imu_states(time_idx)
        strapdown.step(imu_idx, spec_force, ang_rate)
    if rest_gnss == 0:
        z_vec = ownship.gnss_states(time_idx)
        u_vec = strapdown.get_states(imu_idx)
        delta_ang, delta_vel, delta_pos = navigation_filter.step(gnss_idx, z_vec, u_vec)
        strapdown.update_strapdown(imu_idx, delta_ang, delta_vel, delta_pos)

pos_fig, pos_ax = ownship.plot_position()
strapdown.plot_position(pos_ax)

vel_fig, vel_ax = ownship.plot_velocity()
strapdown.plot_velocity(vel_ax)
plt.show()
