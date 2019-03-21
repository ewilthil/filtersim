import numpy as np
import matplotlib.pyplot as plt

from autoseapy.sylte import load_pkl
import autoseapy.visualization as autovis

import generate_single_target_scenario as setup

if __name__ == '__main__':
    #chosen_dataset = 'data/conf_MC1-IPDA_29_171.0.pkl' # This illustrates the case when only the MC1 confirms a track (i.e. the one that should be most conservative, intuitively).
    chosen_dataset = 'data/conf_MC1-IPDA_88_87.0.pkl'
    chosen_dataset = 'data/conf_MC2-IPDA_88_0.0.pkl'
    measurements_all = load_pkl(chosen_dataset)
    first_measurement = measurements_all[0].pop()
    t0 = first_measurement.timestamp
    measurements_all[0].add(first_measurement)
    timestamps = np.arange(t0, t0+3*(len(measurements_all)-1)+1, 3)
    titles = ['MC1-IPDA', 'MC2-IPDA', 'HMM-IPDA', 'DET-IPDA']
    colors = {title : color for title, color in zip(titles, plt.rcParams['axes.color_cycle'])}
    current_managers = setup.setup_trackers(titles)
    preliminary_tracks = {title : dict() for title in titles}
    confirmed_tracks = {title : dict() for title in titles}
    for n_time, timestamp in enumerate(timestamps):
        print "\n"
        for title, manager in current_managers.items():
            print "stepping {}".format(title)
            current_estimates, new_tracks, debug_data = manager.step(measurements_all[n_time], timestamp)
            for track_id, track_list in debug_data['preliminary_tracks'].items():
                if track_id not in preliminary_tracks[title].keys():
                    preliminary_tracks[title][track_id] = []
                preliminary_tracks[title][track_id].append(track_list[-1]) # The rest have been appended before
            for track in new_tracks:
                if track[-1].track_index in preliminary_tracks[title].keys():
                    preliminary_tracks[title][track[-1].track_index].append(track[-1])
            for estimate in current_estimates:
                if estimate.track_index not in confirmed_tracks[title].keys():
                    confirmed_tracks[title][estimate.track_index] = []
                confirmed_tracks[title][estimate.track_index].append(estimate)
            if len(new_tracks) > 0:
                print "Track confirmed in {}".format(title)

    pos_fig, pos_ax = plt.subplots()
    autovis.plot_measurements(measurements_all, pos_ax)
    for title, manager in current_managers.items():
        autovis.plot_track_pos(confirmed_tracks[title], pos_ax, color=colors[title], title=title)
        autovis.plot_track_pos(preliminary_tracks[title], pos_ax, color=colors[title], title=title, ls='--')
    pos_ax.set_aspect('equal')
    pos_ax.legend()
    ext_fig, ext_ax = plt.subplots()
    for title, manager in current_managers.items():
        for track_id, est_list in confirmed_tracks[title].items():
            t = np.array([est.timestamp for est in est_list])
            p = np.array([est.get_existence_probability() for est in est_list])
            ext_ax.plot(t, p, color=colors[title], ls='-', label=title)
            print "max existence probability={} for {} confirmed track".format(np.max(p), title)
        for track_id, est_list in preliminary_tracks[title].items():
            t = np.array([est.timestamp for est in est_list])
            p = np.array([est.get_existence_probability() for est in est_list])
            print "max existence probability={} for {} preliminary track".format(np.max(p), title)
            ext_ax.plot(t, p, color=colors[title], ls='--', label=title)
    plt.show()
