from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import autoseapy.bag_operations as autobag
import autoseapy.ais as autoais
import autoseapy.tracking_common as tracking_common
import autoseapy.tracking as autotrack
import autoseapy.track_initiation as autoinit
import autoseapy.track_management as automanagers
import autoseapy.visualization as autovis
import autoseapy.hidden_markov_model as hmm_models
import analyse_tracks

def add_zero_sets(measurements_all, measurement_timestamps):
    insert_indices = []
    insert_timestamp = []
    previous_timestamp = measurement_timestamps[0]
    for k, timestamp in enumerate(measurement_timestamps):
        if timestamp-previous_timestamp > 3:
            insert_indices.append(k)
            insert_timestamp.append(previous_timestamp+2.88)
        previous_timestamp = timestamp
    for new_timestamp, index in zip(insert_timestamp[::-1], insert_indices[::-1]):
        measurements_all.insert(index, set())
        measurement_timestamps.insert(index, new_timestamp)
    return measurements_all, measurement_timestamps

selected_bag = '/Users/ewilthil/Documents/autosea_testdata/25-09-2018/filtered_bags/filtered_scenario_6_2018-09-25-11-28-47.bag'
chosen_targets = [autoais.known_mmsi['TELEMETRON'], autoais.known_mmsi['MUNKHOLMEN II'], autoais.known_mmsi['KSX_OSD1']]
ais_data, measurements, measurement_timestamps = autobag.bag2raw_data(selected_bag, return_timestamps=True)
ais_data_filtered = {mmsi : ais_data[mmsi] for mmsi in chosen_targets if mmsi in ais_data.keys()}
ais_data_targets = {mmsi : ais_data[mmsi] for mmsi in chosen_targets if mmsi in ais_data.keys() and mmsi is not autoais.known_mmsi['TELEMETRON']}
measurements, measurement_timestamps = add_zero_sets(measurements, measurement_timestamps)
ownship_pose, ownship_twist = autobag.bag2navigation_data(selected_bag, timestamps=measurement_timestamps)


target_process_noise_covariance = 0.05**2
gate_probability = 0.99
maximum_velocity = 15
measurement_mapping = np.array([[1, 0, 0, 0],[0, 0, 1, 0]])
measurement_covariance = 15**2*np.identity(2)
measurement_covariance_range = 20**2
measurement_covariance_bearing = np.deg2rad(2.2997)**2
survival_probability = 1.0
init_prob = 0.3
conf_threshold = 0.95
term_threshold = 0.1

low_PD = 0.3
high_PD = 0.8
hmm_transition_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
hmm_emission_matrix = np.array([[1-low_PD, 1-high_PD], [low_PD, high_PD]])
hmm_initial_probability = np.array([0.5, 0.5])
hmm_pd = hmm_models.HiddenMarkovModel(hmm_transition_matrix, hmm_emission_matrix, hmm_initial_probability, [low_PD, high_PD])
# Markov chain 2 parameters
P_low = 0.7
P_high = 0.7
P_birth = 0.0
P_term = 1-survival_probability

target_model = tracking_common.DWNAModel(target_process_noise_covariance)
track_gate = tracking_common.TrackGate(gate_probability, maximum_velocity)
measurement_model = tracking_common.ConvertedMeasurementModel(measurement_mapping, measurement_covariance_range, measurement_covariance_bearing, high_PD, 15**2)
mc2_measurement_model = tracking_common.ConvertedMeasurementModel(measurement_mapping, measurement_covariance_range, measurement_covariance_bearing, [low_PD, high_PD], 15**2)
van_mc2_measurement_model = tracking_common.ConvertedMeasurementModel(measurement_mapping, measurement_covariance_range, measurement_covariance_bearing, [0, high_PD], 15**2)
markov_measurement_model = tracking_common.ConvertedMeasurementModelMarkovDetectionProbability(measurement_mapping, measurement_covariance_range, measurement_covariance_bearing, hmm_pd, 15**2)

# Vanilla IPDAs
ipda_tracker = autotrack.IPDAFTracker(target_model, measurement_model, track_gate, survival_probability)
ipda_init = autoinit.IPDAInitiator(ipda_tracker, init_prob, conf_threshold, term_threshold)
ipda_term = autotrack.IPDATerminator(term_threshold)
track_manager = automanagers.Manager(ipda_tracker, ipda_init, ipda_term)


# HMM-based IPDA
hmm_ipda_tracker = autotrack.IPDAFTracker(target_model, markov_measurement_model, track_gate, survival_probability)
hmm_ipda_init = autoinit.IPDAInitiator(hmm_ipda_tracker, init_prob, conf_threshold, term_threshold)
hmm_ipda_term = autotrack.IPDATerminator(term_threshold)
hmm_track_manager = automanagers.Manager(hmm_ipda_tracker, hmm_ipda_init, hmm_ipda_term)

markov_probabilities = np.array([[P_low, 1-(P_low+P_term), P_term], [1-(P_high+P_term), P_high, P_term], [0.5*P_birth, 0.5*P_birth, 1-P_birth]]).T
# Vanilla MC2 IPDA

vanilla_mc2_ipda_tracker = autotrack.MC2IPDAFTracker(target_model, van_mc2_measurement_model, track_gate, markov_probabilities)
vanilla_mc2_ipda_tracker_init = autotrack.IPDAFTracker(target_model, measurement_model, track_gate, survival_probability)
vanilla_mc2_ipda_init = autoinit.IPDAInitiator(vanilla_mc2_ipda_tracker_init, init_prob, conf_threshold, term_threshold)
vanilla_mc2_ipda_term = autotrack.MC2IPDATerminator(term_threshold)
vanilla_ipda_mc2_manager = automanagers.Manager(vanilla_mc2_ipda_tracker, vanilla_mc2_ipda_init, vanilla_mc2_ipda_term)

# Extra MC2 IPDA

#markov_probabilities = np.identity(3)
mc2_ipda_tracker = autotrack.MC2IPDAFTracker(target_model, mc2_measurement_model, track_gate, markov_probabilities)
mc2_ipda_tracker_init = autotrack.IPDAFTracker(target_model, measurement_model, track_gate, survival_probability)
mc2_ipda_init = autoinit.IPDAInitiator(mc2_ipda_tracker_init, init_prob, conf_threshold, term_threshold)
mc2_ipda_term = autotrack.MC2IPDATerminator(term_threshold)
ipda_mc2_manager = automanagers.Manager(mc2_ipda_tracker, mc2_ipda_init, mc2_ipda_term)

# Experiment with different settings for 1D 
titles = ['reg. mc1', 'hmm mc1', 'reg. mc2', 'hmm mc2']
current_managers = [track_manager, hmm_track_manager, vanilla_ipda_mc2_manager, ipda_mc2_manager]
#current_managers = [track_manager, hmm_track_manager, ipda_mc2_manager]
#current_managers = [track_manager, ipda_mc2_manager]
#current_managers = [track_manager]
[manager.reset for manager in current_managers]
for k, measurement in enumerate(measurements):
    timestamp = measurement_timestamps[k]
    current_position = ownship_pose[1:3,k]
    for manager in current_managers:
        manager.tracking_method.measurement_model.update_ownship(current_position)
        manager.step(measurement, timestamp)
    print "stepped {}/{}".format(k+1, len(measurements))
fig, ax = plt.subplots()
autovis.plot_track_pos(ais_data_filtered, ax, add_index=True)
autovis.plot_measurements(measurements, ax)
autovis.plot_track_pos(track_manager.track_file, ax, color='r', add_index=True)
autovis.plot_track_pos(hmm_track_manager.track_file, ax, color='g', add_index=True)
autovis.plot_track_pos(ipda_mc2_manager.track_file, ax, color='b', add_index=True)
autovis.plot_track_pos(vanilla_ipda_mc2_manager.track_file, ax, color='c', add_index=True)
ax.set_aspect('equal')

def plot_normalized_distance(track_file, ais_file, title, ax):
    correspondence_dict = analyse_tracks.get_normalized_distance(track_file, ais_file)
    for id_pair, data in correspondence_dict.items():
        time, dist = data
        ax.semilogy(time, dist)
        ais_str = autoais.known_mmsi_rev[id_pair[1]] if id_pair[1] in autoais.known_mmsi_rev.keys() else 'N/A'
        ax.text(time[-1], dist[-1], "({}-{})".format(str(id_pair[0]), ais_str))
    ax.grid(True, which='both')
    ax.set_title("Normalized distance - {}".format(title))

def plot_true_distance(track_file, ais_file, title, ax):
    distance_dict = analyse_tracks.get_absolute_distance(track_file, ais_file)
    for id_pair, data in distance_dict.items():
        time, dist = data
        ax.semilogy(time, dist)
        ais_str = autoais.known_mmsi_rev[id_pair[1]] if id_pair[1] in autoais.known_mmsi_rev.keys() else 'N/A'
        ax.text(time[-1], dist[-1], "({}-{})".format(str(id_pair[0]), ais_str))
    ax.grid(True, which='both')
    ax.set_title("True distance - {}".format(title))

color_cycle = plt.rcParams['axes.color_cycle']
def plot_true_tracks(track_file, ais_file,title, ax):
    true_tracks, _ = analyse_tracks.get_true_tracks(track_file, ais_file)
    false_tracks = {radar_id : radar_ests for radar_id, radar_ests in track_file.items() if radar_id not in true_tracks.keys()}
    print "Number of true/false tracks: {}/{} for {}".format(len(true_tracks), len(false_tracks), title)
    autovis.plot_track_pos(ais_file, ax, add_index=False)
    autovis.plot_measurements(measurements, ax)
    autovis.plot_track_pos(true_tracks, ax, color=color_cycle[0], add_index=True)
    autovis.plot_track_pos(false_tracks, ax, color=color_cycle[1], add_index=True)
    ax.set_aspect('equal')
    ax.set_title('True and false tracks - {}'.format(title))
    return true_tracks, false_tracks

def plot_existence_probability(track_file, title, ax):
    existence_probabilities = analyse_tracks.get_existence_probabilities(track_file)
    for track_id, data in existence_probabilities.items():
        time, existence = data
        ax.plot(time, existence)
        ax.text(time[-1], existence[-1], str(track_id))
        ax.set_title('Existence probability - {}'.format(title))
def plot_false_track_duration(track_file, title, ax):
    false_track_duration = [len(est_list) for est_list in track_file.values()]
    print np.mean(false_track_duration)

for manager, title in zip(current_managers, titles):
    fig, ax = plt.subplots(nrows=2,ncols=2)
    true_tracks, false_tracks = plot_true_tracks(manager.track_file, ais_data_targets, title, ax[0,0])
    plot_false_track_duration(false_tracks, title, ax[1,0])
    plot_existence_probability(true_tracks, title, ax[0,1])
    plot_existence_probability(false_tracks, title, ax[1,1])

# Create publication-quality figs
plt.show()
