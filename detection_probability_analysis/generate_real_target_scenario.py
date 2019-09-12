import numpy as np

import autoseapy.bag_operations as autobag
import autoseapy.ais as autoais
import autoseapy.tracking_common as autocommon
import autoseapy.tracking as autotrack
import autoseapy.track_management as automanagers
import autoseapy.track_initiation as autoinit
import autoseapy.hidden_markov_model as hmm_models

import generate_single_target_scenario as gen_scen

gate_probability = 0.99
maximum_velocity = 15
measurement_mapping = np.array([[1, 0, 0, 0],[0, 0, 1, 0]])
min_cartesian_cov = 0*15**2
measurement_covariance = min_cartesian_cov*np.identity(2)
measurement_covariance_range = 20**2
simulated_measurement_covariance = 10**2*np.identity(2)
measurement_covariance_bearing = np.deg2rad(2.2997)**2
target_process_noise_covariance = 0.05**2
survival_probability = 1.0
new_target_probability = 0.0
init_prob = 0.2
markov_init_prob = np.array([init_prob/2, init_prob/2, 1-init_prob])
conf_threshold = 0.99
term_threshold = 0.1
P_low = 0.8
P_high = P_low
P_birth = 0.0
P_term = 1-survival_probability
low_PD = 0.3
high_PD = 0.8
single_PD = 0.8
hmm_transition_matrix = np.array([[P_low, 1-P_low], [1-P_high, P_high]])
hmm_emission_matrix = np.array([[1-low_PD, 1-high_PD], [low_PD, high_PD]])
hmm_initial_probability = np.array([0.5, 0.5])

ipda_transition_matrix = np.array([[survival_probability*P_low, survival_probability*(1-P_low), 1-survival_probability], [survival_probability*(1-P_high), survival_probability*P_high, 1-survival_probability], [new_target_probability/2, new_target_probability/2, 1-new_target_probability]])

selected_rosbag = '/Users/ewilthil/Documents/autosea_testdata/25-09-2018/filtered_bags/filtered_scenario_6_2018-09-25-11-28-47.bag'
munkholmen_mmsi = autoais.known_mmsi['MUNKHOLMEN II']
drone_mmsi = autoais.known_mmsi['KSX_OSD1']
ownship_mmsi = autoais.known_mmsi['TELEMETRON']
landmark_mmsi = 123456789
chosen_targets = [munkholmen_mmsi, drone_mmsi, landmark_mmsi]

def add_landmark(measurements_all, landmark_mmsi, ais_data):
    landmark_measurements = []
    landmark_time = []
    for measurements in measurements_all:
        t_added = False 
        for measurement in measurements:
            z_pos = measurement.value
            if z_pos[0] < 1500 and z_pos[0] > 1200 and z_pos[1] > 300 and z_pos[1] < 700: # Manually found
                landmark_measurements.append(z_pos)
                t_added = True
                landmark_time.append(measurement.timestamp)
    landmark_measurements = np.array(landmark_measurements)
    landmark_mean = np.mean(landmark_measurements, axis=0)
    landmark_data = []
    ais_data[landmark_mmsi] = []
    for t in landmark_time:
        mark_est = autocommon.Estimate(t, np.array([landmark_mean[0], 0, landmark_mean[1], 0]), np.identity(4), True, landmark_mmsi)
        ais_data[landmark_mmsi].append(mark_est)
    return ais_data

def generate_scenario():
    ais_data, measurements_all, measurement_timestamps = autobag.bag2raw_data(selected_rosbag, return_timestamps=True)
    ais_data = add_landmark(measurements_all, landmark_mmsi, ais_data)
    ais_data = autobag.synchronize_track_file_to_timestamps(ais_data, measurement_timestamps, target_process_noise_covariance)
    true_targets = {mmsi : ais_data[mmsi] for mmsi in chosen_targets if mmsi in ais_data.keys()}
    ownship_state = ais_data[ownship_mmsi]
    return true_targets, ownship_state, measurements_all, measurement_timestamps

target_model = autocommon.DWNAModel(target_process_noise_covariance)
track_gate = autocommon.TrackGate(gate_probability, maximum_velocity)
clutter_model = autocommon.NonparametricClutterModel()

def setup_trackers(titles):
    current_managers = dict()
    for title in titles:
        if title == 'MC1-IPDA':
            measurement_model = autocommon.ConvertedMeasurementModel(measurement_mapping, measurement_covariance_range, measurement_covariance_bearing, single_PD, min_cartesian_cov, clutter_model=clutter_model)
            tracker = autotrack.IPDAFTracker(target_model, measurement_model, track_gate, survival_probability)
            init = autoinit.IPDAInitiator(tracker, init_prob, conf_threshold, term_threshold)
            term = autotrack.IPDATerminator(term_threshold)
        elif title == 'MC2-IPDA':
            measurement_model = autocommon.ConvertedMeasurementModel(measurement_mapping, measurement_covariance_range, measurement_covariance_bearing, [0, single_PD], min_cartesian_cov,clutter_model=clutter_model)
            tracker = autotrack.MC2IPDAFTracker(target_model, measurement_model, track_gate, ipda_transition_matrix)
            init = autoinit.MC2IPDAInitiator(tracker, markov_init_prob, conf_threshold, term_threshold)
            term = autotrack.MC2IPDATerminator(term_threshold)
        elif title == 'HMM-IPDA':
            detection_model = hmm_models.HiddenMarkovModel(hmm_transition_matrix, hmm_emission_matrix, hmm_initial_probability, [low_PD, high_PD])
            measurement_model = autocommon.ConvertedMeasurementModelMarkovDetectionProbability(measurement_mapping, measurement_covariance_range, measurement_covariance_bearing, detection_model, min_cartesian_cov, clutter_model=clutter_model)
            tracker = autotrack.IPDAFTracker(target_model, measurement_model, track_gate, survival_probability)
            init = autoinit.IPDAInitiator(tracker, init_prob, conf_threshold, term_threshold)
            term = autotrack.IPDATerminator(term_threshold)
        elif title == 'DET-IPDA':
            measurement_model = autocommon.ConvertedMeasurementModel(measurement_mapping, measurement_covariance_range, measurement_covariance_bearing, [low_PD, high_PD], min_cartesian_cov, clutter_model=clutter_model)
            tracker = autotrack.MC2IPDAFTracker(target_model, measurement_model, track_gate, ipda_transition_matrix)
            init = autoinit.MC2IPDAInitiator(tracker, markov_init_prob, conf_threshold, term_threshold)
            term = autotrack.MC2IPDATerminator(term_threshold)
        track_manager = automanagers.Manager(tracker, init, term)
        current_managers[title] = track_manager
    return current_managers

def plot_data(ax, measurements, true_tracks):
    pass

if __name__ == '__main__':
    import autoseapy.visualization as autovis
    import matplotlib.pyplot as plt

    true_targets, ownship_state, measurements_all, measurement_timestamps = generate_scenario()
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_axes((0.15, 0.15, 0.8, 0.8))
    autovis.plot_measurements_dots(measurements_all, ax, color='grey', label='Measurements')
    autovis.plot_track_pos(true_targets, ax, color='k', title='Targets', end_title='End position', lw=1)
    autovis.plot_track_pos({-1 : ownship_state}, ax, color='k', ls='--', title='Ownship', lw=1)
    ax.legend(numpoints=1)
    ax.set_ylim(-1000, 5000)
    ax.set_xlim(-1000, 5000)
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.set_aspect('equal')
    target_strings = {munkholmen_mmsi : 'MH2', drone_mmsi : 'OSD', landmark_mmsi : 'Mark'}
    for mmsi, target_list in true_targets.items():
        ax.text(target_list[-1].est_posterior[2], target_list[-1].est_posterior[0]-200, target_strings[mmsi])
    ax.text(ownship_state[-1].est_posterior[2], ownship_state[-1].est_posterior[0]-200, 'Ownship')
    fig.savefig('figs/real_data_overview.pdf')
    plt.show()
