import numpy as np
import matplotlib.pyplot as plt
from filtersim import shipmodels, navigation
T = 300
dt = 0.01
time = np.linspace(0, T, (T+dt)/dt, endpoint = True)
x0 = np.zeros(18)
x0[6] = 10
# Set up navigation system
dt_imu = dt
dt_gps = 20*dt
def heading_ref(t):
    if t < 150:
        return 0
    elif t > 170:
        return np.deg2rad(45)
    else:
        return np.deg2rad((t-150)*2.25)
def surge_ref(t):
    return 10
N_MC = 100
NEES = np.zeros((1501, N_MC))
for n in range(N_MC):
    model = shipmodels.NonlinearStochasticModel()
    nav_sys = navigation.NavigationSystem(time, dt_imu, dt_gps, x0, imu=navigation.unbiased_imu)
    # Set up ownship
    ownship = shipmodels.Ownship(time, model, x0, nav_sys=nav_sys)
    for t_idx, t in enumerate(time):
        ownship.step(t_idx, surge_ref(t), heading_ref(t))
    NEES[:,n] = ownship.nav_sys.get_nees()
    print n+1

plt.plot(nav_sys.gnss_time, np.mean(NEES, axis=1))
plt.show()
