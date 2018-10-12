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
measurements, measurement_timestamps = add_zero_sets(measurements, measurement_timestamps)
ownship_pose, ownship_twist = autobag.bag2navigation_data(selected_bag, timestamps=measurement_timestamps)


target_process_noise_covariance = 0.05**2
gate_probability = 0.99
maximum_velocity = 15
measurement_mapping = np.array([[1, 0, 0, 0],[0, 0, 1, 0]])
measurement_covariance = 15**2*np.identity(2)
measurement_covariance_range = 400.0
measurement_covariance_bearing = 0.001611
detection_probability = 0.8
survival_probability = 1.0
init_prob = 0.3
conf_threshold = 0.95
term_threshold = 0.1

target_model = tracking_common.DWNAModel(target_process_noise_covariance)
track_gate = tracking_common.TrackGate(gate_probability, maximum_velocity)
measurement_model = tracking_common.ConvertedMeasurementModel(measurement_mapping, measurement_covariance_range, measurement_covariance_bearing, detection_probability, 15**2)
# Trackers
ipda_tracker = autotrack.IPDAFTracker(target_model, measurement_model, track_gate, survival_probability)
# Manager - IPDA with hysteresis
ipda_init = autoinit.IPDAInitiator(ipda_tracker, init_prob, conf_threshold, term_threshold)
ipda_term = autotrack.IPDATerminator(term_threshold)
track_manager = automanagers.Manager(ipda_tracker, ipda_init, ipda_term)
for k, measurement in enumerate(measurements):
    z = measurement.pop()
    timestamp = z.timestamp
    measurement.add(z)
    current_position = ownship_pose[1:3,k]
    track_manager.tracking_method.measurement_model.update_ownship(current_position)
    track_manager.step(measurement, timestamp)
    print "stepped {}/{}".format(k+1, len(measurements))
fig, ax = plt.subplots()
autovis.plot_measurements(measurements, ax)
autovis.plot_track_pos(ais_data_filtered, ax)
autovis.plot_track_pos(track_manager.track_file, ax, color='r')
#autovis.plot_track_pos(recorded_radar_data, ax, color='g')
ax.set_aspect('equal')
plt.show()
