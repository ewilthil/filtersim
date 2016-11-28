import numpy as np
from filtersim import shipmodels
import matplotlib.pyplot as plt
T = 300
dt = 0.01
time = np.linspace(0, T, (T+dt)/dt, endpoint=True)
N_0 = np.array([0, 10])
E_0 = np.array([0, 0])
t_0 = 0
theta = 0.5
sigma = 0.3
iou_model_x = shipmodels.IntegratedOU(theta, N_0[1], sigma)
iou_model_y = shipmodels.IntegratedOU(theta, E_0[1], sigma)
target = shipmodels.TargetShip(time, N_0, E_0, iou_model_x, iou_model_y)
targets = [target]
for t_idx, t in enumerate(time):
    if t_idx is 0:
        pass
    else:
        [target.step(t_idx) for target in targets]
    if t >= T/2:
        target.y_model = iou_model_x
pos_fig, pos_ax = plt.subplots()
vel_fig, vel_ax = plt.subplots(ncols=2)
for target in targets:
    target.plot_position(pos_ax)
    target.plot_veloicty(vel_ax)
plt.show()
