import filtersim.single_target_track_initiation.scenario_generation as scen_gen
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
N = 100
fig, ax = plt.subplots()
v_fig, v_ax = plt.subplots(ncols=3)
for i in range(N):
    time, traj, model = scen_gen.generate_trajectory(alpha=0.5)
    l = ax.plot(traj[2,:], traj[0,:])
    ax.plot(traj[2,0], traj[0,0], 'o',color=l[0].get_color())
    v_ax[0].plot(time, traj[1,:])
    v_ax[1].plot(time, traj[3,:])
    v_ax[2].plot(time, np.sqrt(traj[1,:]**2+traj[3,:]**2))
    ax.add_patch(patches.Rectangle((-250, -250), 500, 500, fc='none'))
plt.show()
