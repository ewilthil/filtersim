from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import autoseapy.tracking as autotrack
import autoseapy.visualization as autovis
import autoseapy.ais as autoais
import analyse_tracks
import generate_real_target_scenario as setup
color_cycle = plt.rcParams['axes.color_cycle']

def plot_distance(track_file, ais_file, title):
    distance_dict = analyse_tracks.get_absolute_distance(track_file, ais_file)
    fig, ax = plt.subplots()
    for id_pair, data in distance_dict.items():
        time, dist = data
        ax.semilogy(time, dist)
        ais_str = autoais.known_mmsi_rev[id_pair[1]] if id_pair[1] in autoais.known_mmsi_rev.keys() else 'N/A'
        ax.text(time[-1], dist[-1], "({}-{})".format(str(id_pair[0]), ais_str))
        ax.grid(True, which='both')
        ax.set_title("True distance - {}".format(title))

if __name__ == '__main__':
    titles = ['MC1-IPDA', 'MC2-IPDA', 'HMM-IPDA', 'DET-IPDA']
    colors = {title : color for title, color in zip(titles, color_cycle)}
    markers = {title : marker for title, marker in zip(titles, ['s', 'o', 'd', 'v'])}
    current_managers = setup.setup_trackers(titles)
    true_targets, ownship_state, measurements_all, measurement_timestamps = setup.generate_scenario()
    num_true_tracks = {title : np.zeros(len(measurement_timestamps)) for title in titles}
    num_false_tracks = {title : np.zeros(len(measurement_timestamps)) for title in titles}
    for n_time, timestamp in enumerate(measurement_timestamps):
        measurements = measurements_all[n_time]
        ownship_position = ownship_state[n_time].est_posterior.take((0, 2))
        for title, manager in current_managers.items():
            manager.tracking_method.measurement_model.update_ownship(ownship_position)
            current_estimates, new_tracks, _, = manager.step(measurements, timestamp)
            current_track_dict = {estimate.track_index : [estimate] for estimate in current_estimates}
            current_ais_dict = {mmsi : [true_targets[mmsi][n_time]] for mmsi in true_targets.keys()}
            true_tracks, _ = analyse_tracks.get_true_tracks(current_track_dict, current_ais_dict)
            num_true_tracks[title][n_time] = len(true_tracks)
            num_false_tracks[title][n_time] = len(current_estimates) - len(true_tracks)

    def setup_figure(axis_type='regular'):
        if axis_type == 'regular':
            fig = plt.figure(figsize=(7,4))
            ax = fig.add_axes((0.1, 0.15, 0.85, 0.8))
        elif axis_type == 'xy':
            fig = plt.figure(figsize=(7,7))
            ax = fig.add_axes((0.15, 0.15, 0.8, 0.8))
        return fig, ax
    markers = {title : marker for title, marker in zip(titles, ['s', 'o', 'd', 'v'])}

    rel_time = np.array(measurement_timestamps)-measurement_timestamps[0]
    num_true_fig, num_true_ax = setup_figure()
    for title in titles:
        num_true_ax.plot(rel_time, num_true_tracks[title], label=title, marker=markers[title], markevery=10, lw=2)
    num_true_ax.legend(loc='best')
    num_true_ax.set_ylabel('Number of true tracks')
    num_true_ax.set_xlabel('Time')
    num_true_ax.set_ylim(0, 3.5)
    num_true_ax.grid()
    num_true_fig.savefig('real_data_num_true.pdf')

    num_false_fig, num_false_ax = setup_figure()
    for title in titles:
        num_false_ax.plot(rel_time, num_false_tracks[title], label=title, lw=2, marker=markers[title], markevery=10)
    num_false_ax.legend(loc='best')
    num_false_ax.set_ylabel('Number of false tracks')
    num_false_ax.set_xlabel('Time')
    num_false_ax.set_ylim(0, 3.5)
    num_false_ax.grid()
    num_false_fig.savefig('real_data_num_false.pdf')

    # Plot the tracking results
    for title, manager in current_managers.items():
        fig, ax = plt.subplots()
        autovis.plot_measurements_dots(measurements_all, ax, color='gray',label='Measurements')
        autovis.plot_track_pos(true_targets, ax, lw=2, title='Targets')
        autovis.plot_track_pos(manager.track_file, ax, color=colors[title], lw=2,title='Tracks', end_title='Track end pos.')
        small_ax = fig.add_axes([0.22, 0.62, 0.25, 0.25])
        autovis.plot_measurements_dots(measurements_all, small_ax, color='gray')
        autovis.plot_track_pos(true_targets, small_ax, lw=2)
        autovis.plot_track_pos(manager.track_file, small_ax, color=colors[title], lw=2)
        #ax.set_title('Confirmed tracks for {}'.format(title))
        ax.set_ylim(-1000, 5000)
        ax.set_xlim(-1000, 5000)
        ax.set_xlabel('East [m]')
        ax.set_ylabel('North [m]')
        ax.set_aspect('equal')
        E0 = 1400
        N0 = 1600
        width = 200
        box_N = [N0, N0, N0+width, N0+width, N0]
        box_E = [E0, E0+width, E0+width, E0, E0]
        ax.plot(box_E, box_N, 'k', lw=0.8)
        small_ax.set_xticks([])
        small_ax.set_yticks([])
        small_ax.tick_params(axis='x', direction='in', pad=-15, labelsize='small')
        small_ax.tick_params(axis='y', direction='in', pad=-25, labelsize='small')
        small_ax.set_xlim(E0, E0+width)
        small_ax.set_ylim(N0, N0+width)
        small_ax.set_aspect('equal')
        ax.legend(numpoints=1)
        fig.savefig('figs/tracks_real_data_{}.pdf'.format(title))
    plt.show()
