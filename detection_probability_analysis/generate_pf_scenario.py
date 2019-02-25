import numpy as np

import autoseapy.tracking_common as autocommon
import autoseapy.simulation as autosim
import autoseapy.ais as autoais

landmark_mmsi = 123456789
drone_mmsi = autoais.known_mmsi['KSX_OSD1']
munkholmen_mmsi = autoais.known_mmsi['MUNKHOLMEN II']
PD_high = 0.8
PD_low = 0.3
radar_range = 1000
clutter_density = 10/(4e6)
target_process_noise_covariance = 0.05**2
measurement_covariance_single_axis = 10**2
measurement_covariance = measurement_covariance_single_axis*np.identity(2)
# Configure targets
initial_position = np.array([500, 900])
initial_velocity = np.array([0, -2.6])
t_max = 270
termination_time = 180
detectability_change_time = 90

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

target_mmsi = (drone_mmsi, munkholmen_mmsi, landmark_mmsi)

def generate_scenario():
    time = np.arange(0, t_max, 3, dtype=float)
    radar = autosim.SquareRadar(radar_range, clutter_density, PD_high, measurement_covariance)
    true_targets = dict()
    true_detectability_mode = dict()
    true_existence = dict()
    measurements_all = generate_clutter(time, radar)
    target_measurements, target_state, detectability_state, existence_state = generate_target(time, radar)
    true_targets[munkholmen_mmsi] = target_state
    true_detectability_mode[munkholmen_mmsi] = detectability_state
    true_existence[munkholmen_mmsi] = existence_state
    for k, measurement in enumerate(target_measurements):
        measurements_all[k] = measurements_all[k].union(measurement)
    return true_targets, true_detectability_mode, true_existence, measurements_all, time

if __name__ == '__main__':
    import autoseapy.visualization as autovis
    import matplotlib.pyplot as plt
    true_targets, true_detectability_mode, true_existence_mode, measurements, time = generate_scenario()
    target_fig = plt.figure(figsize=(7,7))
    target_ax = target_fig.add_axes((0.15, 0.15, 0.8, 0.8))
    autovis.plot_track_pos(true_targets, target_ax)
    #autovis.plot_measurements(measurements, ax)
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
    #autovis.make_track_movie(true_targets, measurements, 'sim_movie')
    plt.show()
