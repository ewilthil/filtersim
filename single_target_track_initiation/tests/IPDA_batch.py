import filtersim.single_target_track_initiation.scenario_generation as scen_gen
from filtersim.track_initiation import IntegratedPDA
from filtersim.tracking import ProbabilisticDataAssociation
from filtersim.visualization import plot_with_gradient, plot_trajectories_from_estimates
import matplotlib.pyplot as plt
import numpy as np
time, true_trajectory, measurements_all, motion_model, measurement_model = scen_gen.generate_scenario(P_D=1)
IPDA_init = IntegratedPDA(P_D=1, P_G=0.99,p0=0.5)
est, prob = IPDA_init.offline_processing(measurements_all, time, motion_model)
fig, ax = plt.subplots()
plot_with_gradient(time, measurements_all, ax)
scen_gen.visualize_ipda_list(est, prob, ax)
plt.show()
