import numpy as np

import autoseapy.tracking_common as autocommon
import autoseapy.track_initiation as autoinit
import autoseapy.tracking as autotrack
import autoseapy.track_management as automanagers
import autoseapy.simulation as autosim

t_max = 300
sample_time = 3

# Target parameters
initial_position = np.array([500, 900])
initial_velocity = np.array([0, -2.6])
target_process_noise_covariance = 0.05**2
maximum_velocity = 5

# Sensor / measurement model parameters
PD = 0.8
radar_range = 1000
clutter_density = 10/(4e6)
measurement_covariance_single_axis = 10**2
measurement_covariance = measurement_covariance_single_axis*np.identity(2)
measurement_mapping = np.array([[1, 0, 0, 0],[0, 0, 1, 0]])
gate_probability = 0.99

# Tracker parameters
survival_probability = 1.0
init_prob = 0.3
conf_threshold = 0.99
term_threshold = 0.1

# True target generation
def generate_target(time, radar):
    # Generate true state
    state_transition_model = autocommon.DWNAModel(target_process_noise_covariance)
    initial_state = np.array([initial_position[0], initial_velocity[0], initial_position[1], initial_velocity[1]])
    true_state_list = []
    for k, t in enumerate(time):
        if k == 0:
            true_state = autocommon.Estimate(t, initial_state, np.identity(4), is_posterior=True)
        else:
            true_state = state_transition_model.draw_transition(true_state_list[-1], t, is_posterior=True)
        true_state_list.append(true_state)

    # Generate measurements
    target_originated_measurements = []
    for k, t in enumerate(time):
        true_pos = true_state_list[k].est_posterior.take((0,2))
        measurement = radar.generate_target_measurements([true_pos], t)
        target_originated_measurements.append(measurement)
    return target_originated_measurements, true_state_list

def generate_scenario():
    time = np.arange(0, t_max, sample_time, dtype=float)
    radar = autosim.SquareRadar(radar_range, clutter_density, PD, measurement_covariance)
    measurements_clutter = [radar.generate_clutter_measurements(timestamp) for timestamp in time]
    target_measurements, true_target = generate_target(time, radar)
    measurements_all = []
    for k, measurement in enumerate(target_measurements):
        measurements_all.append(measurements_clutter[k].union(measurement))
    return true_target, measurements_all, time

# Setup tracker
def setup_ipda_manager():
    target_model = autocommon.DWNAModel(target_process_noise_covariance)
    track_gate = autocommon.TrackGate(gate_probability, maximum_velocity)
    clutter_model = autocommon.ConstantClutterModel(clutter_density)
    measurement_model = autocommon.CartesianMeasurementModel(measurement_mapping, measurement_covariance, PD, clutter_model=clutter_model)
    tracker = autotrack.IPDAFTracker(target_model, measurement_model, track_gate, survival_probability)
    track_initiation =autoinit.IPDAInitiator(tracker, init_prob, conf_threshold, term_threshold)
    track_termination = autotrack.IPDATerminator(term_threshold)
    ipda_track_manager = automanagers.Manager(tracker, track_initiation, track_termination)
    return ipda_track_manager

if __name__ == '__main__':
    import autoseapy.visualization as autovis
    import matplotlib.pyplot as plt
    true_targets, measurements_all, time = generate_scenario()
    manager = setup_ipda_manager()
    for measurements, timestamp in zip(measurements_all, time):
        manager.step(measurements, timestamp)

    target_fig = plt.figure(figsize=(7,7))
    target_ax = target_fig.add_axes((0.15, 0.15, 0.8, 0.8))
    autovis.plot_measurements(measurements_all, target_ax)
    autovis.plot_track_pos({1 : true_targets}, target_ax)
    autovis.plot_track_pos(manager.track_file, target_ax, color='r')
    target_ax.set_xlim(-radar_range, radar_range)
    target_ax.set_ylim(-radar_range, radar_range)
    target_ax.set_aspect('equal')
    target_ax.set_xlabel('East [m]')
    target_ax.set_ylabel('North [m]')
    plt.show()
