import numpy as np
import matplotlib.pyplot as plt
import filtersim.generate_trajectory as gentraj
from autopy.sylte import dump_pkl

target_vel = 10
target_heading = np.deg2rad(45)
ownship_vel = 10
ownship_heading = np.deg2rad(45)
max_ang = np.deg2rad(50)
target_x0 = np.array([500, 500, target_heading, target_vel])
ownship_x0 = np.array([0, 0, ownship_heading, ownship_vel])
def target_ref(t):
    return np.array([target_vel, target_heading])

def ownship_ref(t):
    return np.array([ownship_vel, ownship_heading+max_ang*np.sin(np.pi/25.*t)])

ownship, target = gentraj.simulate(target_x0, ownship_x0, target_ref, ownship_ref, 100)

plt.subplot(111)
plt.plot(ownship.state[0,:], ownship.state[1,:])
plt.plot(target.state[0,:], target.state[1,:])
plt.show()
dump_pkl(ownship, 'ownship_traj.pkl')
dump_pkl(target, 'target_traj.pkl')
