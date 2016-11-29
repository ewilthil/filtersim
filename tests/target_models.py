import numpy as np
from filtersim import shipmodels
import matplotlib.pyplot as plt
T = 300
dt = 0.01
time = np.linspace(0, T, (T+dt)/dt, endpoint=True)
x0 = np.array([0, 10, 0, 0])
v_ref = np.array([x0[1], x0[3]])
iou_model = shipmodels.IntegratedOU(dt)
target = shipmodels.TargetShip(time, iou_model, x0)
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
    target.plot_velocity(vel_ax)
plt.show()
