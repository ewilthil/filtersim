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
clutter_density = 5-5
#clutter_density = 2e-6
measurement_covariance_single_axis = 10**2
measurement_covariance = measurement_covariance_single_axis*np.identity(2)
# Configure targets
initial_positions = {
        landmark_mmsi : np.array([-800, -800]),
        drone_mmsi : np.array([900, 200]),
        munkholmen_mmsi : np.array([500, 900]),
        }
velocities = {
        landmark_mmsi : np.array([0, 0]),
        drone_mmsi : np.array([-3.3, 0]),
        munkholmen_mmsi : np.array([0, -2.6]),
        }
detectability_change_times = {
        landmark_mmsi : np.array([]), 
        drone_mmsi : np.array([200, 300, 400, 450]),
        munkholmen_mmsi :np.array([]),
        }

def generate_target(time, mmsi, radar):
    initial_position = initial_positions[mmsi]
    velocity = velocities[mmsi]
    
    # Generate true state
    initial_state = np.array([initial_position[0], velocity[0], initial_position[1], velocity[1]])
    true_state_list = []
    for t in time:
        current_state = initial_state+np.array([velocity[0]*t, 0, velocity[1]*t, 0])
        true_state = autocommon.Estimate(t, current_state, np.identity(4), is_posterior=True)
        true_state_list.append(true_state)

    # Generate detectability state and measurements
    
    # Generate measurements
    detectability_state = []
    target_originated_measurements = []
    for t in time:
        if np.mod(np.sum(detectability_change_times[mmsi] <= t), 2) == 1:
            detectability_state.append(PD_low)
        else:
            detectability_state.append(PD_high)
    for k, t in enumerate(time):
        radar.update_detection_probability(detectability_state[k])
        true_pos = true_state_list[k].est_posterior.take((0,2))
        measurement = radar.generate_target_measurements([true_pos], t)
        [true_state_list[k].store_measurement(z) for z in measurement]
        target_originated_measurements.append(measurement)
    return target_originated_measurements, true_state_list, detectability_state

def generate_clutter(time, radar):
    clutter_all = [radar.generate_clutter_measurements(t) for t in time]
    return clutter_all

target_mmsi = (drone_mmsi, munkholmen_mmsi, landmark_mmsi)

def generate_scenario(targets=target_mmsi, tmax=540):
    time = np.arange(0, tmax, 3, dtype=float)
    radar = autosim.SquareRadar(radar_range, clutter_density, PD_high, measurement_covariance)
    true_targets = dict()
    true_detectability_mode = dict()
    measurements_all = generate_clutter(time, radar)
    for mmsi in targets:
        target_measurements, target_state, detectability_state = generate_target(time, mmsi, radar)
        true_targets[mmsi] = target_state
        true_detectability_mode[mmsi] = detectability_state
        for k, measurement in enumerate(target_measurements):
            measurements_all[k] = measurements_all[k].union(measurement)
    return true_targets, true_detectability_mode, measurements_all, time

if __name__ == '__main__':
    import autoseapy.visualization as autovis
    import matplotlib.pyplot as plt
    true_targets, true_detectability_mode, measurements, time = generate_scenario()
    target_fig = plt.figure(figsize=(7,7))
    target_ax = target_fig.add_axes((0.15, 0.15, 0.8, 0.8))
    autovis.plot_track_pos(true_targets, target_ax)
    #autovis.plot_measurements(measurements, ax)
    target_ax.set_xlim(-1000, 1000)
    target_ax.set_ylim(-1000, 1000)
    target_ax.set_aspect('equal')
    target_ax.set_xlabel('East [m]')
    target_ax.set_ylabel('North [m]')
    det_fig = plt.figure(figsize=(7,4))
    det_ax = det_fig.add_axes((0.1, 0.15, 0.85, 0.8))
    det_ax.plot(time, true_detectability_mode[drone_mmsi], lw=2)
    det_ax.set_ylim(0, 1)
    det_ax.set_xlabel('Time [s]')
    det_ax.set_ylabel('Detection probability')
    det_ax.grid()
    target_fig.savefig('simulation_setup.pdf')
    det_fig.savefig('simulation_setup_detectability.pdf')
    #autovis.make_track_movie(true_targets, measurements, 'sim_movie')
    plt.show()
