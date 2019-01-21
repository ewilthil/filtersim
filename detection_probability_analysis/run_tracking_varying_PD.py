from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import autoseapy.bag_operations as autobag
import autoseapy.ais as autoais
import autoseapy.tracking_common as tracking_common
import autoseapy.tracking as autotrack
import autoseapy.visualization as autovis
import analyse_tracks
import setup_trackers
color_cycle = plt.rcParams['axes.color_cycle']

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
        mark_est = tracking_common.Estimate(t, np.array([landmark_mean[0], 0, landmark_mean[1], 0]), np.identity(4), True, landmark_mmsi)
        ais_data[landmark_mmsi].append(mark_est)


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
landmark_mmsi = 123456789
munkholmen_mmsi = autoais.known_mmsi['MUNKHOLMEN II']
drone_mmsi = autoais.known_mmsi['KSX_OSD1']
chosen_targets = [autoais.known_mmsi['TELEMETRON'], munkholmen_mmsi, drone_mmsi]
ais_data, measurements, measurement_timestamps = autobag.bag2raw_data(selected_bag, return_timestamps=True)
ais_data_filtered = {mmsi : ais_data[mmsi] for mmsi in chosen_targets if mmsi in ais_data.keys()}
ais_data_targets = {mmsi : ais_data[mmsi] for mmsi in chosen_targets if mmsi in ais_data.keys() and mmsi is not autoais.known_mmsi['TELEMETRON']}
measurements, measurement_timestamps = add_zero_sets(measurements, measurement_timestamps)
ownship_pose, ownship_twist = autobag.bag2navigation_data(selected_bag, timestamps=measurement_timestamps)

ipda_track_manager = setup_trackers.setup_ipda_manager()
hmm_track_manager = setup_trackers.setup_hmm_ipda_manager()
vanilla_ipda_mc2_manager = setup_trackers.setup_mc2_ipda_manager()
ipda_mc2_manager = setup_trackers.setup_mcn_ipda_manager()

# Experiment with different settings for 1D 
current_managers = [ipda_track_manager, hmm_track_manager, vanilla_ipda_mc2_manager, ipda_mc2_manager]
#current_managers = [hmm_track_manager]
[manager.reset for manager in current_managers]
for k, measurement in enumerate(measurements):
    timestamp = measurement_timestamps[k]
    current_position = ownship_pose[1:3,k]
    for manager in current_managers:
        manager.tracking_method.measurement_model.update_ownship(current_position)
        manager.step(measurement, timestamp)

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
    print "Average length of false tracks: {} for {}".format(np.mean([len(estimates) for estimates in false_tracks.values()]), title)
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
    durations = [len(track_list) for track_list in track_file.values()]
    durations.sort()
    cumulative_prob_duration = [float(n+1)/(len(durations)) for n in range(len(durations))]
    # Hack it to look nice with a starting vertical line
    durations.insert(0, 0.9*durations[0])
    cumulative_prob_duration.insert(0, 0)
    ax.step(durations, cumulative_prob_duration, where='post',label=title, lw=2)

def find_track_confirm_time(track_file):
    existence_probabilities = analyse_tracks.get_existence_probabilities(track_file)
    conf_times = []
    for track_id, data in existence_probabilities.items():
        time, existence = data
        is_preliminary = True
        len_preliminary = 0
        for ext_val in existence:
            if ext_val < setup_trackers.conf_threshold and is_preliminary:
                len_preliminary += 1
            elif ext_val >= setup_trackers.conf_threshold:
                is_preliminary = False
        conf_times.append(len_preliminary)
    return conf_times

def plot_detection_probability(track_file, PD_values, title, ax):
    PD_all = analyse_tracks.get_detection_probability(track_file, PD_values)
    for track_id, data in PD_all.items():
        time, PD = data
        ax.plot(time, PD)
        ax.set_title('Existence probability - {}'.format(title))

def plot_existence_prob(radar_id_list, track_file, ax, track_label, t0=False, color='k', style='-'):
    existence_probabilities = analyse_tracks.get_existence_probabilities(track_file)
    for radar_id in radar_id_list:
        time, ext_prob = existence_probabilities[radar_id]
        if t0 != False:
            time = time-t0
        ax.plot(time, ext_prob, label=track_label, color=color, ls=style)

def plot_detection_probability(radar_id_list, track_manager, ax, track_label, t0=0, color='k', style='-'):
    for radar_id in radar_id_list:
        time = np.array([est.timestamp for est in track_manager.track_file[radar_id]])-t0
        detection_probabilities = np.zeros_like(time)
        if track_label == 'IPDA' or track_label == 'MC2-IPDA':
            detection_probabilities = track_manager.tracking_method.measurement_model.detection_probability*np.ones_like(time)
        elif track_label == 'HMM-IPDA':
            mode_probabilities = hmm_track_manager.tracking_method.measurement_model.detection_probability.state_probabilities[radar_id]
            mode_values = hmm_track_manager.tracking_method.measurement_model.detection_probability.state_values
            detection_probabilities = np.array([np.sum(elem*mode_values) for elem in mode_probabilities])
        elif track_label == 'MCN-IPDA':
            pass
        ax.plot(time, detection_probabilities, color=color, ls=style)



add_landmark(measurements, landmark_mmsi, ais_data_targets)

#TODO the PD for the HMM isnt properly evaluated
PD_values_all = [0.5, [0.3, 0.8], [0, 0.5], [0.3, 0.8]]
dur_fig = plt.figure(figsize=(7,4))
dur_ax = dur_fig.add_axes((0.15, 0.15, 0.8, 0.8))
dur_fig, dur_ax = plt.subplots()
ext_fig, ext_ax = plt.subplots(nrows=3)

pub_fig = plt.figure(figsize=(7,4))
pub_ax = pub_fig.add_axes((0.15, 0.15, 0.8, 0.8))
track_labels = ['OSD', 'MH2', 'LM']
manager_labels = ['IPDA', 'HMM-IPDA', 'MC2-IPDA', 'MCN-IPDA']
for k, manager in enumerate(current_managers):
    manager_label = manager_labels[k]
    title = repr(manager.tracking_method)
    fig, ax = plt.subplots(nrows=2,ncols=2)
    true_tracks, mmsi_dict = analyse_tracks.get_true_tracks(manager.track_file, ais_data_targets)
    false_tracks = {radar_id : radar_ests for radar_id, radar_ests in manager.track_file.items() if radar_id not in true_tracks.keys()}
    for mmsi, e_ax, track_label in zip([drone_mmsi, munkholmen_mmsi, landmark_mmsi], ext_ax, track_labels):
        if mmsi in mmsi_dict.keys():
            plot_existence_prob(mmsi_dict[mmsi], true_tracks, e_ax, track_label)
            e_ax.set_title(track_label)
            if mmsi == drone_mmsi:
                plot_existence_prob(mmsi_dict[mmsi], true_tracks, pub_ax, manager_label, measurement_timestamps[0], color_cycle[k], style='-')
                #plot_detection_probability(mmsi_dict[mmsi], manager, pub_ax[1], manager_label, measurement_timestamps[0], color_cycle[k], style='--')
    plot_true_tracks(manager.track_file, ais_data_targets, title, ax[0,0])
    conf_times = find_track_confirm_time(true_tracks)
    print "True track confirmation times: {}".format(conf_times)
    plot_false_track_duration(false_tracks, title, ax[1,0])
    plot_false_track_duration(false_tracks, manager_labels[k], dur_ax)
    plot_existence_probability(true_tracks, title, ax[0,1])
    plot_existence_probability(false_tracks, title, ax[1,1])
[e_ax.legend() for e_ax in ext_ax]
pub_ax.grid('on')
dur_ax.legend(loc='lower left')
dur_ax.grid()
dur_ax.set_xlabel('Time [s]')
pub_ax.set_xlabel('Time [s]')
pub_ax.set_ylabel('Existence probability')
pub_ax.legend()
pub_ax.set_yticks(np.arange(0, 1.1, 0.2))
# Create publication-quality figs
dur_fig.savefig('duration_cumulative.pdf')
pub_fig.savefig('ext_prob_all.pdf')

xy_fig = plt.figure(figsize=(7,7))
xy_ax = xy_fig.add_axes((0.15, 0.15, 0.8, 0.8))
for measurement_set in measurements:
    for measurement in measurement_set:
        z_line, = xy_ax.plot(measurement.value[1], measurement.value[0], '.', color='#aaaaaa')
ownship_line, = xy_ax.plot(ownship_pose[2,:], ownship_pose[1,:], 'k--')
end_position, = xy_ax.plot(ownship_pose[2,-1], ownship_pose[1,-1], 'ko')
map_labels = ['OSD', 'MH II', 'Mark']
for mmsi, label in zip([drone_mmsi, munkholmen_mmsi, landmark_mmsi], map_labels):
    position = np.array([estimate.est_posterior[[0,2]] for estimate in ais_data_targets[mmsi]])
    target_line, = xy_ax.plot(position[:,1], position[:,0], 'k-')
    xy_ax.plot(position[-1,1], position[-1,0], 'ko')
    xy_ax.text(position[-1,1], position[-1,0]-250, label)
for track_id, track_list in true_tracks.items():
    position = np.array([estimate.est_posterior[[0,2]] for estimate in track_list])
    true_target_line, = xy_ax.plot(position[:,1], position[:,0], '-', color=color_cycle[0])
    xy_ax.plot(position[-1,1], position[-1,0], 'o', color=color_cycle[0])

for track_id, track_list in false_tracks.items():
    position = np.array([estimate.est_posterior[[0,2]] for estimate in track_list])
    false_target_line, = xy_ax.plot(position[:,1], position[:,0], '-', color=color_cycle[2])
    xy_ax.plot(position[-1,1], position[-1,0], 'o', color=color_cycle[2])

#xy_ax.set_xlim(-1000, 5000)
#xy_ax.set_ylim(-1000, 5000)
xy_ax.set_xlim(1000, 3000)
xy_ax.set_ylim(1000, 2400)
xy_ax.set_aspect('equal')
#xy_ax.legend([ownship_line, target_line, end_position, z_line, true_target_line, false_target_line], ['Ownship', 'Targets', 'End position', 'Measurements', 'True tracks', 'False tracks'])
xy_ax.set_ylabel('North [m]')
xy_ax.set_xlabel('East [m]')

xy_fig.savefig('vanilla_results.pdf')

xy_fig = plt.figure(figsize=(7,7))
xy_ax = xy_fig.add_axes((0.15, 0.15, 0.8, 0.8))
for measurement_set in measurements:
    for measurement in measurement_set:
        z_line, = xy_ax.plot(measurement.value[1], measurement.value[0], '.', color='#aaaaaa')
ownship_line, = xy_ax.plot(ownship_pose[2,:], ownship_pose[1,:], 'k--')
end_position, = xy_ax.plot(ownship_pose[2,-1], ownship_pose[1,-1], 'ko')
map_labels = ['OSD', 'MH II', 'Mark']
for mmsi, label in zip([drone_mmsi, munkholmen_mmsi, landmark_mmsi], map_labels):
    position = np.array([estimate.est_posterior[[0,2]] for estimate in ais_data_targets[mmsi]])
    target_line, = xy_ax.plot(position[:,1], position[:,0], 'k-')
    xy_ax.plot(position[-1,1], position[-1,0], 'ko')
    xy_ax.text(position[-1,1], position[-1,0]-250, label)
for track_id, track_list in true_tracks.items():
    position = np.array([estimate.est_posterior[[0,2]] for estimate in track_list])
    true_target_line, = xy_ax.plot(position[:,1], position[:,0], '-', color=color_cycle[0])
    xy_ax.plot(position[-1,1], position[-1,0], 'o', color=color_cycle[0])

for track_id, track_list in false_tracks.items():
    position = np.array([estimate.est_posterior[[0,2]] for estimate in track_list])
    false_target_line, = xy_ax.plot(position[:,1], position[:,0], '-', color=color_cycle[2])
    xy_ax.plot(position[-1,1], position[-1,0], 'o', color=color_cycle[2])

xy_ax.set_xlim(-1000, 5000)
xy_ax.set_ylim(-1000, 5000)
xy_ax.set_xlim(400, 2400)
xy_ax.set_ylim(1000, 3000)
xy_ax.set_aspect('equal')
#xy_ax.legend([ownship_line, target_line, end_position, z_line, true_target_line, false_target_line], ['Ownship', 'Targets', 'End position', 'Measurements', 'True tracks', 'False tracks'])
xy_ax.set_ylabel('North [m]')
xy_ax.set_xlabel('East [m]')

xy_fig.savefig('vanilla_results.pdf')

drone_synced = autotrack.sync_track_list(measurement_timestamps, ais_data_targets[drone_mmsi])

#autovis.make_ais_movie(drone_synced, measurements, hmm_track_manager.track_file, 'movie_out')
#autovis.make_ais_movie(drone_synced, measurements, {}, 'movie_out')
#autovis.make_single_track_movie(est_list, measurements_all, measurement_model, fname, chosen_fps=10, spacing=300):
#autovis.make_track_movie(ipda_track_manager.track_file, measurements, 'movie_out', ais_track_file=ais_data_targets)



plt.show()
