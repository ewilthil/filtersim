import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from filtersim import shipmodels, tracking
from filtersim.visualization import plot_with_gradient, plot_estimate_list, animate_track
v_max = 10
v_ref = np.random.uniform(-v_max, v_max, 2)
T = 20
dt = 1
P_G = 0.99
A_max = 1.1*v_max*T
time = np.arange(0, T+dt, dt)
target_model = shipmodels.IntegratedOU(dt, 2)
target_est_model = target_model # No model mismatch
target_x0 = np.array([0, v_ref[0], 0, v_ref[1]])
target = shipmodels.TargetShip(time, target_model, target_x0)
measurement_model = tracking.MeasurementModel(8**2, 5e-3, np.array([-A_max, A_max]))
measurements_target_all = [None for _ in range(len(time))]
measurements_all = [None for _ in range(len(time))]
initial_estimate = tracking.Estimate(0, target_x0, np.diag((25, 1, 25, 1)), measurement_model, True, 1)
PDA = tracking.ProbabilisticDataAssociation(measurement_model, P_G)
estimates = []

for t_idx, t in enumerate(time):
    # Step the real target
    target.step(t_idx, v_ref)
    # Obtain the prior estimate
    if t_idx > 0:
        current_estimate = tracking.Estimate.from_estimate(t, estimates[t_idx-1], target_est_model, v_ref)
    else:
        current_estimate = initial_estimate
    measurement_target = [target.cartesian_position_measurement(t_idx)]
    measurements_all[t_idx] = measurement_model.generate_measurements(measurement_target, t)
    tracking.gate_measurements(measurements_all[t_idx], current_estimate, P_G)
    PDA.calculate_posterior(current_estimate)
    estimates.append(current_estimate)

pos_fig, pos_ax = plt.subplots()
target.plot_position(pos_ax)
plot_estimate_list(estimates, pos_ax)
#plot_with_gradient(time, measurement_all)
N_meas = [len(est.measurements) for est in estimates]
animate_track(estimates, measurements_all)
print N_meas
plt.show()
