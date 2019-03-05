import numpy as np
import matplotlib.pyplot as plt
np.random.seed(12345)

import autoseapy.visualization as autovis
import generate_single_target_scenario as setup
import setup_trackers
import analyse_tracks

def get_active_tracks(track_file, time):
    num_active_tracks = np.zeros_like(time)
    track_conf_times = np.array([track[0].timestamp for track in track_file.values()])
    track_term_times = np.array([track[-1].timestamp for track in track_file.values()])
    for k, t in enumerate(time):
        n_tracks = sum(track_conf_times <= t)-sum(track_term_times < t)
        num_active_tracks[k] = n_tracks
    return num_active_tracks, track_conf_times, track_term_times

def plot_tracks(ax, true_tracks, false_tracks, true_states):
    autovis.plot_track_pos(true_states, ax, color='k')
    autovis.plot_track_pos(true_tracks, ax, color='g')
    autovis.plot_track_pos(false_tracks, ax, color='r')
    ax.set_xlim(-setup.radar_range, setup.radar_range)
    ax.set_ylim(-setup.radar_range, setup.radar_range)
    ax.set_aspect('equal')

def plot_tracks_status(ax, track_file, track_status, true_states):
    true_tracks = {track_id : [] for track_id in track_file.keys()}
    false_tracks = {track_id : [] for track_id in track_file.keys()}
    inactive_tracks ={track_id : [] for track_id in track_file.keys()}
    for radar_track_index, est_list in track_file.items():
        for estimate in est_list:
            current_track_status = track_status[estimate.timestamp][radar_track_index]
            if current_track_status == 'true':
                true_tracks[radar_track_index].append(estimate)
            elif current_track_status == 'false':
                false_tracks[radar_track_index].append(estimate)
            elif current_track_status == 'inactive':
                inactive_tracks[radar_track_status].append(estimate)
    autovis.plot_track_pos(true_states, ax, color='k')
    autovis.plot_track_pos(true_tracks, ax, color='g')
    autovis.plot_track_pos(false_tracks, ax, color='r')
    autovis.plot_track_pos(inactive_tracks, ax, color='gray')
    ax.set_xlim(-setup.radar_range, setup.radar_range)
    ax.set_ylim(-setup.radar_range, setup.radar_range)
    ax.set_aspect('equal')

def find_true_track(track_file, true_target):
    true_target_index = -1
    for track_index, track in track_file.items():
        if True:
            true_target_index = track_index
    return true_target_index

if __name__ == '__main__':
    _, _, _, _, _, time = setup.generate_scenario()
    N_MC = 100
    ipda_track_manager = setup_trackers.setup_ipda_manager(measurement_type=setup_trackers.SIMULATED)
    hmm_track_manager = setup_trackers.setup_hmm_ipda_manager(measurement_type=setup_trackers.SIMULATED)
    vanilla_ipda_mc2_manager = setup_trackers.setup_mc2_ipda_manager(measurement_type=setup_trackers.SIMULATED)
    ipda_mc2_manager = setup_trackers.setup_mcn_ipda_manager(measurement_type=setup_trackers.SIMULATED)
    titles = ['IPDA', 'HMM-IPDA', 'MC2-IPDA', 'MCN-IPDA']
    current_managers_list = [ipda_track_manager, hmm_track_manager, vanilla_ipda_mc2_manager, ipda_mc2_manager]
    current_managers = {title : manager for title, manager in zip(titles, current_managers_list)}

    num_true_tracks = {title : np.zeros((len(time), N_MC)) for title in titles}
    num_false_tracks = {title : np.zeros((len(time), N_MC)) for title in titles}
    duration_false_tracks = {title : [] for title in titles}
    existence_prob = {title: np.zeros((len(time), N_MC)) for title in titles}
    detection_prob = {title: np.zeros((len(time), N_MC)) for title in titles}

    ext_prob_fig, ext_prob_ax = plt.subplots()
    term_pos_fig, term_pos_ax = plt.subplots()
    area_fig, area_ax  = plt.subplots()
    for n_mc in range(N_MC):
        true_target, true_detectability, true_existence, measurements_clutter, measurements_all, measurement_timestamps = setup.generate_scenario()
        # Run tracking with clutter-only
        [manager.reset() for manager in current_managers.values()]
        for n_time, timestamp in enumerate(measurement_timestamps):
            measurements = measurements_clutter[n_time]
            for title, manager in current_managers.items():
                current_estimates, _ = manager.step(measurements, timestamp)
                num_false_tracks[title][n_time][n_mc] = len(current_estimates)
        for title, manager in current_managers.items():
            [duration_false_tracks[title].append(len(est_list)) for est_list in manager.track_file.values()]
        # Run tracking with target
        [manager.reset() for manager in current_managers.values()]
        for n_time, timestamp in enumerate(measurement_timestamps):
            measurements = measurements_all[n_time]
            for title, manager in current_managers.items():
                current_estimates, _ = manager.step(measurements, timestamp)
                true_track_index = find_true_track(manager.track_file, true_target)
                if true_track_index > 0: # True track is found. Extract the latest and greatest existence probability, detection probability and error
                    t_ext, p_ext = analyse_tracks.get_existence_probability([manager.track_file[true_track_index][-1]])
                    existence_prob[title][n_time][n_mc] = p_ext[0]
                    t_det, p_det = analyse_tracks.get_detection_probability([manager.track_file[true_track_index][-1]], manager.tracking_method.measurement_model.get_detection_probability(true_track_index))
                    detection_prob[title][n_time][n_mc] = p_det[0]
                else:
                    existence_prob[title][n_time][n_mc] = np.nan
                    detection_prob[title][n_time][n_mc] = np.nan
        print "completed run {}/{}".format(n_mc+1, N_MC)

    markers = {title : marker for title, marker in zip(titles, ['s', 'o', 'd', 'v'])}
    
    def setup_figure(axis_type='regular'):
        if axis_type == 'regular':
            fig = plt.figure(figsize=(7,4))
            ax = fig.add_axes((0.1, 0.15, 0.85, 0.8))
        elif axis_type == 'xy':
            fig = plt.figure(figsize=(7,7))
            ax = fig.add_axes((0.15, 0.15, 0.8, 0.8))
        return fig, ax

    detection_fig = plt.figure(figsize=(7,4))
    detection_ax  = detection_fig.add_axes((0.1, 0.15, 0.85, 0.8))
    detection_ax.plot(measurement_timestamps, true_detectability, 'k')
    for title, detection_values in detection_prob.items():
        det_val = np.nanmean(detection_values, axis=1)
        detection_ax.plot(measurement_timestamps, det_val, label=title, marker=markers[title], markevery=20, lw=2)
    detection_ax.legend(loc='best')
    detection_ax.set_xlabel('time')
    detection_ax.set_ylim(0, 1)


    #num_true_fig, num_true_ax = plt.subplots(figsize=(7,4))
    num_true_fig = plt.figure(figsize=(7,4))
    num_true_ax = num_true_fig.add_axes((0.1, 0.15, 0.85, 0.8))
    num_true_ax.step(measurement_timestamps, true_existence, where='mid', color='k', lw=2)
    for title in titles:
        average_num_true_tracks = np.mean(num_true_tracks[title], axis=1)
        num_true_ax.plot(time, average_num_true_tracks, label=title, marker=markers[title], markevery=20, lw=2)
    num_true_ax.legend(loc='best')
    num_true_ax.set_ylabel('Average number of true tracks')
    num_true_ax.set_xlabel('Time')
    num_true_ax.set_ylim(0, 2)
    num_true_ax.grid()

    num_false_fig, num_false_ax = setup_figure()
    for title in titles:
        average_num_false_tracks = np.mean(num_false_tracks[title], axis=1)
        num_false_ax.plot(time, average_num_false_tracks, label=title, lw=2, marker=markers[title], markevery=20)
    num_false_ax.legend()
    num_false_ax.set_ylabel('Average number of false tracks')
    num_false_ax.set_xlabel('Time')
    num_false_ax.grid()
    
    duration_fig, duration_ax = setup_figure()
    durations = [duration_false_tracks[title] for title in titles]
    duration_ax.hist(durations, label=titles)
    duration_ax.set_xlabel('False track duration (scans)')
    duration_ax.legend(loc='best')
    duration_ax.grid()

    num_true_fig.savefig('figs/average_number_true_targets.pdf')
    num_false_fig.savefig('figs/average_number_false_targets.pdf')
    duration_fig.savefig('figs/false_track_duration.pdf')
    plt.show()
