import filtersim.single_target_track_initiation.scenario_generation as scen_gen
from filtersim import tracking, visualization
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
time, traj, model = scen_gen.generate_trajectory(alpha=1)
measurement_model = tracking.MeasurementModel(
        target_cov=7**2*np.identity(2),
        clutter_density=10./(500**2),
        x_lims=np.array([-250, 250]),
        P_D=1)
measurements = scen_gen.generate_measurements(time, traj, measurement_model)
fig, ax = plt.subplots()
ax.plot(traj[2,:],traj[0,:],color='gray')
ax.plot(traj[2,0],traj[0,0],'o',color='gray')
visualization.plot_with_gradient(time, measurements,ax)
visualization.plot_with_gradient(time, measurements)
plt.show()
