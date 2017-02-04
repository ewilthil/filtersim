import filtersim.single_target_track_initiation.scenario_generation as scen_gen
from filtersim.track_initiation import IntegratedPDA
from filtersim.tracking import ProbabilisticDataAssociation
from filtersim.visualization import plot_with_gradient
import matplotlib.pyplot as plt
import numpy as np
time, true_trajectory, measurements_all, motion_model, measurement_model = scen_gen.generate_scenario()
IPDA_init = IntegratedPDA()
est_dict = mn_init.offline_processing(measurements_all, time, PDA, motion_model)
fig, ax = plt.subplots()
vel_fig, vel_ax = plt.subplots(ncols=2)
plot_with_gradient(time, measurements_all, ax)
scen_gen.visualize_mn_dict(est_dict, ax, vel_ax)
plt.show()
