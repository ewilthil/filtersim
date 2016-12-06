import numpy as np
from filtersim import shipmodels
import matplotlib.pyplot as plt
T = 300
dt = 0.01
time = np.linspace(0, T, (T+dt)/dt, endpoint=True)
x0_target = np.array([0, 10, 0, 0])
v_ref_target = np.array([x0_target[1], x0_target[3]])
iou_model = shipmodels.IntegratedOU(dt, 2)
dwna_model = shipmodels.DiscreteWNA(dt, 2)
ownship_model = shipmodels.NonlinearStochasticModel()
x0_ownship = np.zeros(18)
x0_ownship[6] = 10
ownship = shipmodels.Ownship(time, ownship_model, x0_ownship)
target = shipmodels.TargetShip(time, iou_model, x0_target)
target_dwna = shipmodels.TargetShip(time, dwna_model, x0_target)
targets = [target, target_dwna]
def psi_ref(t):
    if t < 150:
        return 0
    elif t > 170:
        return 45
    else:
        return (t-150)*2.25
for t_idx, t in enumerate(time):
    [target.step(t_idx, v_ref_target) for target in targets]
    ownship.step(t_idx, 10, np.deg2rad(psi_ref(t)))


pos_fig, pos_ax = plt.subplots()
vel_fig, vel_ax = plt.subplots(ncols=3)
for target in targets:
    target.plot_position(pos_ax)
    target.plot_velocity(vel_ax)
ownship.plot_position(pos_ax)
ownship.plot_velocity(vel_ax)
ownship.plot_roll_pitch_heave()
ownship.plot_velocity_body()
ownship.plot_yaw()
plt.show()
