import numpy as np

import autoseapy.tracking_common as autocommon
import autoseapy.simulation as autosim
import autoseapy.ais as autoais
import autoseapy.tracking as autotrack
import autoseapy.track_management as automanagers
import autoseapy.track_initiation as autoinit
import autoseapy.hidden_markov_model as hmm_models

# Scenario parameters
munkholmen_mmsi = autoais.known_mmsi['MUNKHOLMEN II']
PD_high = 0.8
PD_low = 0.3
radar_range = 1000
clutter_density = 1e-5
target_process_noise_covariance = 0.05**2
measurement_covariance = 10**2
measurement_covariance_matrix = measurement_covariance*np.identity(2)
# Configure targets
initial_position = np.array([500, 900])
initial_velocity = np.array([0, -2.6])
sample_time = 3
t_max = 270
termination_time = 180
detectability_change_time = 90

# Tracker parameters
gate_probability = 0.99
maximum_velocity = 15
measurement_mapping = np.array([[1, 0, 0, 0],[0, 0, 1, 0]])
survival_probability = 0.99
new_target_probability = 1-0.999
init_prob = 0.2
markov_init_prob = np.array([init_prob/2, init_prob/2, 1-init_prob])
conf_threshold = 0.99
term_threshold = 0.1
P_low = 0.8
P_high = P_low
P_term = 1-survival_probability
hmm_transition_matrix = np.array([[P_low, 1-P_low], [1-P_high, P_high]])
hmm_emission_matrix = np.array([[1-PD_low, 1-PD_high], [PD_low, PD_high]])
hmm_initial_probability = np.array([0.5, 0.5])
ipda_transition_matrix = np.array([[survival_probability*P_low, survival_probability*(1-P_low), 1-survival_probability], [survival_probability*(1-P_high), survival_probability*P_high, 1-survival_probability], [new_target_probability/2, new_target_probability/2, 1-new_target_probability]])


def generate_target(time, radar):
    # Generate true state
    state_transition_model = autocommon.DWNAModel(target_process_noise_covariance)
    initial_state = np.array([initial_position[0], initial_velocity[0], initial_position[1], initial_velocity[1]])
    true_state_list = []
    for k, t in enumerate(time):
        if k == 0:
            true_state_list.append(autocommon.Estimate(t, initial_state, np.identity(4), is_posterior=True))
        elif t <= termination_time:
            true_state = state_transition_model.draw_transition(true_state_list[-1], t, is_posterior=True)
            true_state_list.append(true_state)

    # Generate measurements
    true_detectability = []
    true_existence = []
    target_originated_measurements = []
    for k, t in enumerate(time):
        if t >= detectability_change_time:
            radar.update_detection_probability(PD_low)
        if k < len(true_state_list):
            true_detectability.append(radar.detection_probability)
            true_pos = true_state_list[k].est_posterior.take((0,2))
            measurement = radar.generate_target_measurements([true_pos], t)
            [true_state_list[k].store_measurement(z) for z in measurement]
            target_originated_measurements.append(measurement)
            true_existence.append(1)
        else:
            true_detectability.append(None)
            target_originated_measurements.append(set())
            true_existence.append(0)
    return target_originated_measurements, true_state_list, np.array(true_detectability), np.array(true_existence)

def generate_clutter(time, radar):
    clutter_all = [radar.generate_clutter_measurements(t) for t in time]
    return clutter_all

def generate_scenario():
    time = np.arange(0, t_max, sample_time, dtype=float)
    radar = autosim.SquareRadar(radar_range, clutter_density, PD_high, measurement_covariance_matrix)
    measurements_clutter = generate_clutter(time, radar)
    target_measurements, target_state, detectability_state, existence_state = generate_target(time, radar)
    true_target = target_state
    true_detectability_mode = detectability_state
    true_existence = existence_state
    measurements_all = []
    for k, measurement in enumerate(target_measurements):
        measurements_all.append(measurements_clutter[k].union(measurement))
    return true_target, true_detectability_mode, true_existence, measurements_clutter, measurements_all, time

target_model = autocommon.DWNAModel(target_process_noise_covariance)
track_gate = autocommon.TrackGate(gate_probability, maximum_velocity)
clutter_model = autocommon.ConstantClutterModel(clutter_density)
def setup_trackers(titles):
    current_managers = dict()
    for title in titles:
        if title == 'MC1-IPDA':
            measurement_model = autocommon.CartesianMeasurementModel(measurement_mapping, measurement_covariance_matrix, PD_high, clutter_model=clutter_model)
            tracker = autotrack.IPDAFTracker(target_model, measurement_model, track_gate, survival_probability)
            init = autoinit.IPDAInitiator(tracker, init_prob, conf_threshold, term_threshold)
            term = autotrack.IPDATerminator(term_threshold)
        elif title == 'MC2-IPDA':
            measurement_model = autocommon.CartesianMeasurementModel(measurement_mapping, measurement_covariance_matrix, [0, PD_high], clutter_model=clutter_model)
            tracker = autotrack.MC2IPDAFTracker(target_model, measurement_model, track_gate, ipda_transition_matrix)
            init = autoinit.MC2IPDAInitiator(tracker, markov_init_prob, conf_threshold, term_threshold)
            term = autotrack.MC2IPDATerminator(term_threshold)
        elif title == 'HMM-IPDA':
            detection_model = hmm_models.HiddenMarkovModel(hmm_transition_matrix, hmm_emission_matrix, hmm_initial_probability, [PD_low, PD_high], clutter_density=clutter_density)
            measurement_model = autocommon.CartesianMeasurementModelMarkovDetectionProbability(measurement_mapping, measurement_covariance_matrix, detection_model, clutter_model=clutter_model)
            tracker = autotrack.IPDAFTracker(target_model, measurement_model, track_gate, survival_probability)
            init = autoinit.IPDAInitiator(tracker, init_prob, conf_threshold, term_threshold)
            term = autotrack.IPDATerminator(term_threshold)
        elif title == 'DET-IPDA':
            measurement_model = autocommon.CartesianMeasurementModel(measurement_mapping, measurement_covariance_matrix, [PD_low, PD_high], clutter_model=clutter_model)
            tracker = autotrack.MC2IPDAFTracker(target_model, measurement_model, track_gate, ipda_transition_matrix)
            init = autoinit.MC2IPDAInitiator(tracker, markov_init_prob, conf_threshold, term_threshold)
            term = autotrack.MC2IPDATerminator(term_threshold)
        elif title == 'DET1-IPDA':
            this_transition_matrix = np.array([[survival_probability, new_target_probability], [1-survival_probability, 1-new_target_probability]])
            measurement_model = autocommon.CartesianMeasurementModel(measurement_mapping, measurement_covariance_matrix, [PD_high], clutter_model=clutter_model)
            tracker = autotrack.MC2IPDAFTracker(target_model, measurement_model, track_gate, this_transition_matrix)
            init = autoinit.MC2IPDAInitiator(tracker, np.array([0.2, 0.8]), conf_threshold, term_threshold)
            term = autotrack.MC2IPDATerminator(term_threshold)
        else:
            print "Unknown tracker {}, skipping".format(title)
            continue

        track_manager = automanagers.Manager(tracker, init, term)
        current_managers[title] = track_manager
    return current_managers



if __name__ == '__main__':
    import autoseapy.visualization as autovis
    import matplotlib.pyplot as plt
    true_targets, true_detectability_mode, true_existence_mode, measurements_clutter, measurements_all, time = generate_scenario()
    target_fig = plt.figure(figsize=(7,7))
    target_ax = target_fig.add_axes((0.15, 0.15, 0.8, 0.8))
    autovis.plot_track_pos(true_targets, target_ax)
    target_ax.set_xlim(-radar_range, radar_range)
    target_ax.set_ylim(-radar_range, radar_range)
    target_ax.set_aspect('equal')
    target_ax.set_xlabel('East [m]')
    target_ax.set_ylabel('North [m]')
    ext_fig = plt.figure(figsize=(7,4))
    ext_ax = ext_fig.add_axes((0.1, 0.15, 0.85, 0.8))
    for true_ext in true_existence_mode.values():
        ext_ax.step(time, true_ext, color='k',where='mid', lw=2)
    ext_ax.set_ylim(0, 1.1)
    ext_ax.set_xlim(0, t_max)
    ext_ax.set_xlabel('Time [s]')
    ext_ax.set_ylabel('Target posterior probability')
    ext_ax.grid()
    det_fig = plt.figure(figsize=(7,4))
    det_ax = det_fig.add_axes((0.1, 0.15, 0.85, 0.8))
    for true_det in true_detectability_mode.values():
        det_ax.step(time, true_det, color='k',where='mid', lw=2)
    det_ax.set_ylim(0, 1)
    det_ax.set_xlim(0, t_max)
    det_ax.set_xlabel('Time [s]')
    det_ax.set_ylabel('Detection probability')
    det_ax.grid()
    target_fig.savefig('figs/pf_simulation_setup.pdf')
    det_fig.savefig('figs/pf_simulation_setup_detectability.pdf')
    plt.show()
