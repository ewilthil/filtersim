import numpy as np
import matplotlib.pyplot as plt
np.random.seed(12345)

import autoseapy.visualization as autovis
from autoseapy.sylte import dump_pkl

import generate_single_target_scenario as setup
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

def find_true_track(track_file, true_target_measurements):
    true_target_index = -1
    for track_index, track in track_file.items():
        N_total = 0
        for estimate in track:
            for measurement in estimate.measurements:
                if measurement in true_target_measurements:
                    N_total += 1
        if N_total > 0.5:
            true_target_index = track_index
    return true_target_index

def is_true_estimate(estimate, true_state):
    true_pos = true_state.est_posterior.take((0,2))
    est_pos = estimate.est_posterior.take((0,2))
    return np.linalg.norm(est_pos-true_pos) < 100


if __name__ == '__main__':
    _, _, _, _, _, time = setup.generate_scenario()
    N_MC = 100
    titles = ['MC1-IPDA', 'MC2-IPDA', 'HMM-IPDA', 'DET-IPDA']
    current_managers = setup.setup_trackers(titles)

    num_true_tracks = {title : np.zeros((len(time), N_MC)) for title in titles}
    num_lost_tracks = {title : np.zeros((len(time), N_MC)) for title in titles}
    num_false_tracks = {title : np.zeros((len(time), N_MC)) for title in titles}
    duration_false_tracks = {title : [] for title in titles}
    existence_prob = {title: np.zeros((len(time), N_MC)) for title in titles}
    detection_prob = {title: np.zeros((len(time), N_MC)) for title in titles}
    detection_prob_mode = {title: np.zeros((len(time), N_MC)) for title in titles}

    confirmed_tracker = 'MC2-IPDA'
    compared_tracker = 'DET-IPDA'

    for n_mc in range(N_MC):
        true_target, true_detectability, true_existence, measurements_clutter, measurements_all, measurement_timestamps = setup.generate_scenario()
        true_target_measurements = set.union(*[z_all-z_clutter for z_all, z_clutter in zip(measurements_all, measurements_clutter)])
        false_confirmed_track_measurements = {title: [] for title in titles}
        # Run tracking with clutter-only
        [manager.reset() for manager in current_managers.values()]
        for n_time, timestamp in enumerate(measurement_timestamps):
            measurements = measurements_clutter[n_time]
            for title, manager in current_managers.items():
                current_estimates, new_tracks, debug_data = manager.step(measurements, timestamp)
                for track in new_tracks:
                    track_measurements = [estimate.measurements for estimate in track]
                    false_confirmed_track_measurements[title].append(track_measurements)
                num_false_tracks[title][n_time][n_mc] = len(current_estimates)
        for title, manager in current_managers.items():
            print "{} manager has {} false tracks".format(title, len(manager.track_file))
        if len(current_managers[confirmed_tracker].track_file) > 0:# and len(current_managers[compared_tracker].track_file) == 0:
            for track in current_managers[confirmed_tracker].track_file.values():
                confirmed_timestamps = [estimate.timestamp for estimate in track]
                confirmed_measurements = [estimate.measurements for estimate in track]
                fname = 'data/conf_{}_{}_{}.pkl'.format(confirmed_tracker, n_mc, confirmed_timestamps[0])
                dump_pkl(confirmed_measurements, fname)
                print "saved {}".format(fname)
        for title, manager in current_managers.items():
            [duration_false_tracks[title].append(len(est_list)) for est_list in manager.track_file.values()]
        # Run tracking with target
        [manager.reset() for manager in current_managers.values()]
        true_tracks = {title : set() for title in titles}
        for n_time, timestamp in enumerate(measurement_timestamps):
            measurements = measurements_all[n_time]
            for title, manager in current_managers.items():
                current_estimates, new_tracks, _ = manager.step(measurements, timestamp)
                active_tracks = {track_id : manager.track_file[track_id] for track_id in manager.active_tracks}
                true_track_index = -1
                for estimate in current_estimates:
                    if n_time < len(true_target):
                        if is_true_estimate(estimate, true_target[n_time]):
                            true_track_index = estimate.track_index
                            true_tracks[title].add(true_track_index)
                        elif estimate.track_index in true_tracks[title]:
                            num_lost_tracks[title][n_time][n_mc] = 1
                    else:
                        if estimate.track_index in true_tracks[title]:
                            num_lost_tracks[title][n_time][n_mc] = 1

                if true_track_index > 0: # True track is found. Extract the latest and greatest existence probability, detection probability and error
                    num_true_tracks[title][n_time][n_mc] = 1
                    t_ext, p_ext = analyse_tracks.get_existence_probability([manager.track_file[true_track_index][-1]])
                    existence_prob[title][n_time][n_mc] = p_ext[0]
                    t_det, p_det = analyse_tracks.get_detection_probability_mean([manager.track_file[true_track_index][-1]], manager.tracking_method.measurement_model.get_detection_probability(true_track_index))
                    detection_prob[title][n_time][n_mc] = p_det[0]
                    t_det, p_det = analyse_tracks.get_detection_probability_mode([manager.track_file[true_track_index][-1]], manager.tracking_method.measurement_model.get_detection_probability(true_track_index))
                    detection_prob_mode[title][n_time][n_mc] = p_det[0]
                else:
                    existence_prob[title][n_time][n_mc] = np.nan
                    detection_prob[title][n_time][n_mc] = np.nan
                    detection_prob_mode[title][n_time][n_mc] = np.nan
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

    detection_fig, detection_ax = setup_figure()
    detection_ax.plot(measurement_timestamps, true_detectability, 'k')
    for title, detection_values in detection_prob.items():
        det_val = np.nanmean(detection_values, axis=1)
        det_val_mode = np.nanmean(detection_prob_mode[title], axis=1)
        l = detection_ax.plot(measurement_timestamps, det_val_mode, label=title, marker=markers[title], markevery=10, lw=2)
        #detection_ax.plot(measurement_timestamps, det_val_mode, label=title, marker=markers[title], markevery=10, lw=2, color=l[0].get_color(),ls='--')
    detection_ax.legend(loc='best')
    detection_ax.set_xlabel('Time')
    detection_ax.set_ylabel('Average detectability')
    detection_ax.set_ylim(0, 1)


    num_true_fig, num_true_ax = setup_figure()
    num_true_ax.step(measurement_timestamps, true_existence, where='mid', color='k', lw=2)
    for title in titles:
        average_num_true_tracks = np.mean(num_true_tracks[title], axis=1)
        num_true_ax.plot(time, average_num_true_tracks, label=title, marker=markers[title], markevery=10, lw=2)
    num_true_ax.legend(loc='best')
    num_true_ax.set_ylabel('Average number of true tracks')
    num_true_ax.set_xlabel('Time')
    num_true_ax.set_ylim(0, 2)
    num_true_ax.grid()

    num_false_fig, num_false_ax = setup_figure()
    for title in titles:
        average_num_false_tracks = np.mean(num_false_tracks[title], axis=1)
        num_false_ax.plot(time, average_num_false_tracks, label=title, lw=2, marker=markers[title], markevery=10)
    num_false_ax.legend(loc='best')
    num_false_ax.set_ylabel('Average number of false tracks')
    num_false_ax.set_xlabel('Time')
    num_false_ax.grid()

    num_lost_fig, num_lost_ax = setup_figure()
    for title in titles:
        average_num_lost_tracks = np.mean(num_lost_tracks[title], axis=1)
        num_lost_ax.plot(time, average_num_lost_tracks, label=title, lw=2, marker=markers[title], markevery=10)
    num_lost_ax.legend(loc='best')
    num_lost_ax.set_ylabel('Average number of lost tracks')
    num_lost_ax.set_xlabel('Time')
    num_lost_ax.grid()
    
    #duration_fig, duration_ax = setup_figure()
    #durations = [duration_false_tracks[title] for title in titles]
    #duration_ax.hist(durations, label=titles)
    #duration_ax.set_xlabel('False track duration (scans)')
    #duration_ax.legend(loc='best')
    #duration_ax.grid()

    detection_fig.savefig('figs/average_detectability_mode.pdf')
    num_true_fig.savefig('figs/average_number_true_targets.pdf')
    num_false_fig.savefig('figs/average_number_false_targets.pdf')
    num_lost_fig.savefig('figs/average_number_lost_tracks.pdf')
    #duration_fig.savefig('figs/false_track_duration.pdf')
    plt.show()
