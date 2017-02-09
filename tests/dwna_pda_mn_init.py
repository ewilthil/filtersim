import numpy as np
import matplotlib.pyplot as plt
from filtersim import tracking, shipmodels, track_initiation, visualization

#Generic parameters
T = 20
dt = 1
time = np.arange(0, T+dt, dt)
P_G = 0.99

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

# Define the initiation method
M = 3
N = 4
v_max = 20
mn_initiation = track_initiation.m_of_n(M, N, v_max)

# Define the data association / filtering method
PDA = tracking.ProbabilisticDataAssociation(measurement_model, P_G)

# Placeholder variables
measurements_all = [None for _ in range(len(time))]
for t_idx, t in enumerate(time):
    # Propagate the old estimates
    confirmed_estimates = [tracking.Estimate.from_estimate(t, est, target_model, np.zeros(2)) for est in confirmed_estimates]
    preliminary_estimates = [tracking.Estimate.from_estimate(t, est, target_model, np.zeros(2)) for est in preliminary_estimates]

    # Step the real target and generate measurements
    true_target.step(t_idx, np.zeros(2))
    measurements_targets = [true_target.cartesian_position_measurement(t_idx)]
    measurements = measurement_model.generate_measurements(measurements_targets, t)
    measurements_all[t_idx] = measurements
    measurements_associated = [False for _ in measurements]

    # Gate confirmed targets
    measurements = tracking.gate_measurements(measurements, confirmed_estimates, P_G)
    # Gate preliminary targets
    measurements = tracking.gate_measurements(measurements, preliminary_estimates, P_G)
    # Measurement update using the associated measurements
    [PDA.calculate_posterior(estimate) for estimate in confirmed_estimates+preliminary_estimates]
    # Perform track initiation and update preliminary/confirmed track list
    # preliminary estimates = old preliminary+new_preliminary-new confirmed
    preliminary_estimates += mn_initiation.form_new_tracks(measurements)
    new_confirmed, preliminary_estimates = mn_initiation.update_track_status(preliminary_estimates)
    confirmed_estimates += new_confirmed
    [append_estimate(est, current_track_dict) for est in confirmed_estimates+preliminary_estimates]


#animate_track(current_track_dict[1], measurements)
fig, ax = visualization.plot_trajectories_from_estimates(current_track_dict[1])
ax.plot(true_target.states[2,:], true_target.states[0,:],'k',lw=2)
ax.plot(true_target.states[2,0], true_target.states[0,0],'ok',lw=2)
[visualization.plot_trajectories_from_estimates(current_track_dict[i], ax) for i in current_track_dict.keys()]
visualization.animate_track(current_track_dict[1], measurements_all)
plt.show()
