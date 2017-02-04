import filtersim.single_target_track_initiation.scenario_generation as scen_gen
import matplotlib.pyplot as plt
from filtersim import visualization
time, true_trajectory, measurements_all, motion_model, measurement_model = scen_gen.generate_scenario(P_D=0.7)
fig, ax = plt.subplots()
ax.plot(true_trajectory[2,:], true_trajectory[0,:], color='gray')
ax.plot(true_trajectory[2,0], true_trajectory[0,0], 'o', mfc='gray')
visualization.plot_with_gradient(time, measurements_all, ax)
visualization.plot_with_gradient(time, measurements_all)
plt.show()
