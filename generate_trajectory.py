import numpy as np
from base_classes import Model
from autoseapy.sylte import dump_pkl
import matplotlib.pyplot as plt

dt, Tend = 0.01, 300
time = np.arange(0, Tend+dt, dt)
D = -np.diag((0.5, 1, 10, 10, 10, 1))
T = -np.diag((30, 1, 30, 10, 10, 60))
Q = np.diag((1e-2, 1e-2, 1e-4, 1e-4, 1e-4, 5e-4))

initial_target_heading = np.deg2rad(225)
final_target_heading = np.deg2rad(225)
target_velocity = 12
target_init = np.zeros(18)
target_init[0] = 4000
target_init[1] = 1600
target_init[6] = target_velocity
target_init[5] = initial_target_heading
maneuver_start = 150
maneuver_duration = 150
maneuver_end = maneuver_start+maneuver_duration
target_velfunc = lambda t : target_velocity
def target_turnfunc(t):
    if t < maneuver_start:
        return initial_target_heading
    elif t > maneuver_end:
        return final_target_heading
    else:
        return initial_target_heading+(final_target_heading-initial_target_heading)/maneuver_duration*(t-maneuver_start)
target_func = lambda t : np.array([target_velfunc(t), target_turnfunc(t)])

initial_ownship_heading = np.deg2rad(0)
final_ownship_heading = np.deg2rad(45)
maneuver_duration = 20
maneuver_end = maneuver_start+maneuver_duration
ownship_velocity = 10
ownship_velfunc = lambda t : ownship_velocity
def ownship_turnfunc(t):
    if t < maneuver_start:
        return initial_ownship_heading
    elif t > maneuver_end:
        return final_ownship_heading
    else:
        return initial_ownship_heading+(final_ownship_heading-initial_ownship_heading)/maneuver_duration*(t-maneuver_start)
ownship_func = lambda t : np.array([ownship_velfunc(t), ownship_turnfunc(t)])
ownship_init = np.zeros(18)
ownship_init[5] = initial_ownship_heading
ownship_init[6] = ownship_velocity
target = Model(D, T, Q, target_init, time)
ownship = Model(D, T, Q, ownship_init, time)
target_ref = np.zeros((2,len(time)))
ownship_ref = np.zeros((2,len(time)))
for k, t in enumerate(time):
    target_ref[:,k] = target_func(t)
    target.step(k, target_ref[:,k])
    ownship_ref[:,k] = ownship_func(t)
    ownship.step(k, ownship_ref[:,k])

fig, ax = plt.subplots(1,3)
ax[0].plot(target.state[1,:],target.state[0,:])
ax[0].plot(ownship.state[1,:],ownship.state[0,:])
ax[1].plot(target.time, np.rad2deg(target_ref[1,:]))
ax[1].plot(target.time, np.rad2deg(target.state[5,:]))
ax[2].plot(target.time, np.rad2deg(target.state_diff[5,:]))
vel_fig, vel_ax = plt.subplots(2,1)
vel_ax[0].plot(target.time, target.state_diff[0,:])
vel_ax[1].plot(target.time, target.state_diff[1,:])

plt.show()
#dump_pkl(target, 'target_test.pkl')
#dump_pkl(ownship, 'ownship_test.pkl')
