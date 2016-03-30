import numpy as np
from base_classes import Model

dt, Tend = 0.01, 300
time = np.arange(0, Tend+dt, dt)
D0 = -np.diag((0.5, 1, 10, 10, 10, 1))
T0 = -np.diag((30, 1, 30, 10, 10, 60))
Q0 = np.diag((1e-2, 1e-2, 1e-4, 1e-4, 1e-4, 5e-4))
state_xpos = 0
state_ypos = 1
state_vel = 6
state_heading = 5

def simulate(target_x0, ownship_x0, target_refs, ownship_refs, Tend, dt=0.01, D=D0, T=T0, Q=Q0):
    target_init_state = np.zeros(18)
    ownship_init_state = np.zeros(18)
    target_init_state[[state_xpos, state_ypos, state_heading, state_vel]] = target_x0
    ownship_init_state[[state_xpos, state_ypos, state_heading, state_vel]] = ownship_x0
    time = np.arange(0, Tend+dt, dt)
    target = Model(D, T, Q, target_init_state, time)
    ownship = Model(D, T, Q, ownship_init_state, time)
    target_ref = np.zeros((2,len(time)))
    ownship_ref = np.zeros((2,len(time)))
    for k, t in enumerate(time):
        target_ref[:,k] = target_refs(t)
        target.step(k, target_ref[:,k])
        ownship_ref[:,k] = ownship_refs(t)
        ownship.step(k, ownship_ref[:,k])
    return ownship, target
