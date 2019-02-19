import numpy as np
import matplotlib.pyplot as plt
np.random.seed(12345)

import autoseapy.visualization as autovis
import generate_scenario as setup
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

if __name__ == '__main__':
    _, _, _, time = setup.generate_scenario()
    N_MC = 10
    ipda_track_manager = setup_trackers.setup_ipda_manager(measurement_type=setup_trackers.SIMULATED)
    hmm_track_manager = setup_trackers.setup_hmm_ipda_manager(measurement_type=setup_trackers.SIMULATED)
    vanilla_ipda_mc2_manager = setup_trackers.setup_mc2_ipda_manager(measurement_type=setup_trackers.SIMULATED)
    ipda_mc2_manager = setup_trackers.setup_mcn_ipda_manager(measurement_type=setup_trackers.SIMULATED)
    current_managers = [ipda_track_manager, hmm_track_manager, vanilla_ipda_mc2_manager, ipda_mc2_manager]
    titles = ['IPDA', 'HMM-IPDA', 'MC2-IPDA', 'MCN-IPDA']
    num_true_tracks = {title : np.zeros((len(time), N_MC)) for title in titles}
    num_false_tracks = {title : np.zeros((len(time), N_MC)) for title in titles}
    duration_false_tracks = {title : [] for title in titles}

    ext_prob_fig, ext_prob_ax = plt.subplots()
    term_pos_fig, term_pos_ax = plt.subplots()
    area_fig, area_ax  = plt.subplots()
    for n_mc in range(N_MC):
        true_targets, true_detectability, measurements_all, measurement_timestamps = setup.generate_scenario()
        [manager.reset() for manager in current_managers]
        for n_time, timestamp in enumerate(measurement_timestamps):
            measurements = measurements_all[n_time]
            for manager, title in zip(current_managers, titles):
                manager.step(measurements, timestamp)
        for manager, title in zip(current_managers, titles):
            true_tracks, mmsi_dict = analyse_tracks.get_true_tracks(manager.track_file, true_targets)
            false_idx = set(manager.track_file.keys()).difference(set(true_tracks.keys()))
            false_tracks = {track_id : manager.track_file[track_id] for track_id in false_idx}
            num_true_tracks[title][:,n_mc], true_conf_time, true_term_time = get_active_tracks(true_tracks, time)
            num_false_tracks[title][:,n_mc], false_conf_time, false_term_time = get_active_tracks(false_tracks, time)
            [duration_false_tracks[title].append(len(track)) for track in false_tracks.values()]
        print n_mc
    markers = {title : marker for title, marker in zip(titles, ['s', 'o', 'd', 'v'])}
    #num_true_fig, num_true_ax = plt.subplots(figsize=(7,4))
    num_true_fig = plt.figure(figsize=(7,4))
    num_true_ax = num_true_fig.add_axes((0.1, 0.15, 0.85, 0.8))
    for title in titles:
        average_num_true_tracks = np.mean(num_true_tracks[title], axis=1)
        num_true_ax.plot(time, average_num_true_tracks, label=title, marker=markers[title], markevery=20, lw=2)
    num_true_ax.legend(loc='best')
    num_true_ax.set_ylabel('Average number of true tracks')
    num_true_ax.set_xlabel('Time')
    num_true_ax.set_ylim(0, 4)
    num_true_ax.grid()

    num_false_fig = plt.figure(figsize=(7,4))
    num_false_ax = num_false_fig.add_axes((0.1, 0.15, 0.85, 0.8))
    for title in titles:
        average_num_false_tracks = np.mean(num_false_tracks[title], axis=1)
        num_false_ax.plot(time, average_num_false_tracks, label=title, lw=2, marker=markers[title], markevery=20)
    num_false_ax.legend()
    num_false_ax.set_ylabel('Average number of false tracks')
    num_false_ax.set_xlabel('Time')
    num_false_ax.grid()
    
    duration_fig = plt.figure(figsize=(7,4))
    duration_ax = duration_fig.add_axes((0.1, 0.15, 0.85, 0.8))
    durations = [duration_false_tracks[title] for title in titles]
    duration_ax.hist(durations, label=titles)
    duration_ax.set_xlabel('False track duration (scans)')
    duration_ax.legend(loc='best')
    duration_ax.grid()

    num_true_fig.savefig('average_number_true_targets.pdf')
    num_false_fig.savefig('average_number_false_targets.pdf')
    duration_fig.savefig('false_track_duration.pdf')
    plt.show()
