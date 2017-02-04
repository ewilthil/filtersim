import numpy as np
from filtersim import tracking, shipmodels
import matplotlib.pyplot as plt
from filtersim.visualization import plot_estimate_list, animate_track

#Generic parameters
T = 20
dt = 1
time = np.arange(0, T+dt, dt)
P_G = 0.99
v_ref = np.zeros(2)

# Define the measurement model
P_D_true = 0.9
radar_cov = 5**2
clutter_density_true = 5e-5
tracking_limits = np.array([-250, 250])
measurement_model = tracking.MeasurementModel(radar_cov, clutter_density_true, tracking_limits, P_D=P_D_true)

# Define the true target tracjectory
target_model = shipmodels.DiscreteWNA(dt, 2)
target_x0 = np.array([0, 5, 0, 10])
true_target = shipmodels.TargetShip(time, target_model, target_x0)
confirmed_estimates = []
preliminary_estimates = []

def append_estimate(estimate, track_dict):
    idx = estimate.track_index
    if idx not in track_dict.keys():
        track_dict[idx] = []
    track_dict[idx].append(estimate)
current_track_dict = dict()
estimates = []
# Data association / filtering method
IPDA = tracking.IntegratedPDA(P_D_true, P_G, 0.9)
existence_probability = np.zeros(len(time))
initial_estimate =tracking.Estimate(time[0], target_x0, np.diag((25, 1, 25, 1)), True, 1)
estimates = []
measurements_all = [False for _ in time]
for t_idx, timestamp in enumerate(time):
    true_target.step(t_idx, v_ref)
    if t_idx > 0:
        current_estimate = tracking.Estimate.from_estimate(timestamp, estimates[t_idx-1], target_model, v_ref)
    else:
         current_estimate = initial_estimate
    state_target = [true_target.states[:,t_idx]]
    measurements_all[t_idx] = measurement_model.generate_measurements(state_target, timestamp)
    tracking.gate_measurements(measurements_all[t_idx], current_estimate, P_G)
    post_est, prob = IPDA.calculate_posterior(current_estimate)
    print prob
    existence_probability[t_idx] = prob
    estimates.append(post_est)

pos_fig, pos_ax = plt.subplots()
true_target.plot_position(pos_ax)
plot_estimate_list(estimates, pos_ax)
#plot_with_gradient(time, measurement_all)
N_meas = [len(est.measurements) for est in estimates]
animate_track(estimates, measurements_all)
print N_meas
plt.show()
