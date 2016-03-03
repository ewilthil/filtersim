import numpy as np
from base_classes import Model
from autopy.sylte import dump_pkl
import matplotlib.pyplot as plt

dt, Tend = 0.01, 300
time = np.arange(0, Tend+dt, dt)
D = -np.diag((0.5, 1, 10, 10, 10, 1))
T = -np.diag((30, 1, 30, 10, 10, 60))
Q = np.diag((1e-1, 1, 1e-1, 1, 1, 1e-5))

initial_target_heading = np.deg2rad(225)
final_target_heading = np.deg2rad(180)
target_velocity = 12
target_init = np.zeros(18)
target_init[0] = 4000
target_init[1] = 1600
target_init[6] = target_velocity
target_init[5] = initial_target_heading
maneuver_start = 150
maneuver_duration = 30
maneuver_end = maneuver_start+maneuver_duration
turn_func = lambda t : initial_target_heading+(final_target_heading-initial_target_heading)/maneuver_duration*(t-maneuver_start)
heading_ref = np.piecewise(time, [time < maneuver_start, time > maneuver_end], [initial_target_heading, final_target_heading, turn_func])

ownship_heading = 0
ownship_velocity = 10
ownship_init = np.zeros(18)
ownship_init[6] = ownship_velocity
target = Model(D, T, Q, target_init, time)
ownship = Model(D, T, Q, ownship_init, time)
target_ref = np.zeros((2,len(time)))
for k, t in enumerate(time):
    # Set reference
    if t < 10:
        target_ref[:,k] = np.array([target_velocity, initial_target_heading])
    else:
        target_ref[:,k] = np.array([target_velocity, final_target_heading])
    target_ref[1,k] = heading_ref[k]
    # Propagate state
    target.step(k, target_ref[:,k])
    ownship.step(k, np.array([ownship_velocity, ownship_heading]))

fig, ax = plt.subplots(1,3)
ax[0].plot(target.state[1,:],target.state[0,:])
ax[0].plot(ownship.state[1,:],ownship.state[0,:])
ax[1].plot(target.time, np.rad2deg(target_ref[1,:]))
ax[1].plot(target.time, np.rad2deg(target.state[5,:]))
ax[2].plot(target.time, np.rad2deg(target.state_diff[5,:]))
plt.show()
#dump_pkl(target, 'target_test.pkl')
#dump_pkl(ownship, 'ownship_test.pkl')
