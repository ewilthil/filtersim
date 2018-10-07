from __future__ import division
import glob
import numpy as np
import matplotlib.pyplot as plt
import autoseapy.bag_operations as autobag
import autoseapy.tracking_common as autocommon
import autoseapy.ais as autoais

def get_sample_times(measurements_all):
    sample_times = []
    current_time = None
    for measurements in measurements_all:
        z = measurements.pop()
        if current_time == None:
            current_time = z.timestamp
        else:
            sample_times.append(z.timestamp-current_time)
            current_time = z.timestamp
    return sample_times

def plot_cdf(data, ax):
    data.sort()
    cdf = np.array([k for k, _ in enumerate(data)])/(len(data)-1)
    ax.step(data, cdf, where='pre')

def plot_sample_time_cdf(filenames):
    fig, ax = plt.subplots()
    sample_times = []
    for rosbag in filenames:
        if ".bag" not in rosbag:
            continue
        _, measurements_all = autobag.bag2raw_data(rosbag)
        sample_times = sample_times+get_sample_times(measurements_all)
        print "processed {}".format(rosbag)
    plot_cdf(sample_times, ax)

def is_detected(est_list, measurements_all, ownship_pose, measurement_model, gate):
    # Assumes measurements have been synced with est_list
    detection_indicator = []
    aspect_angle = []
    previous_timestamp = est_list[0].timestamp
    for k, measurements in enumerate(measurements_all):
        current_position = ownship_pose[1:3,k]
        measurement_model.update_ownship(current_position)
        current_ais_estimate = est_list[k]
        current_ais_velocity = np.array([current_ais_estimate.est_posterior[1], current_ais_estimate.est_posterior[3]])
        if np.linalg.norm(current_ais_velocity) < 0.1:
            continue
        gated_measurements = gate.gate_estimate(current_ais_estimate, measurements, measurement_model)
        detection_indicator.append(1) if len(gated_measurements) > 0 else detection_indicator.append(0)
        previous_timestamp = current_ais_estimate.timestamp
        aspect_angle.append(calculate_aspect_angle(current_ais_estimate, current_position))
    return detection_indicator, aspect_angle

def detect_based_on_ais(chosen_rosbag, measurement_model, gate, chosen_mmsi):
    ais_data, measurements_all, measurement_timestamps = autobag.bag2raw_data(chosen_rosbag, True)
    measurements_all, measurement_timestamps = add_zero_sets(measurements_all, measurement_timestamps)
    ais_data = autobag.synchronize_track_file_to_timestamps(ais_data, measurement_timestamps, q=0)
    ownship_pose, ownship_twist = autobag.bag2navigation_data(chosen_rosbag, measurement_timestamps)
    detection_indicator, aspect_angle = is_detected(ais_data[chosen_mmsi], measurements_all, ownship_pose, measurement_model, gate)
    return detection_indicator, aspect_angle

def calculate_aspect_angle(estimate, ownship_position):
    est_pos = np.array([estimate.est_posterior[0], estimate.est_posterior[2]])
    est_vel = np.array([estimate.est_posterior[1], estimate.est_posterior[3]])
    pos_vec = ownship_position-est_pos
    cos_angle = pos_vec.dot(est_vel)/(np.linalg.norm(pos_vec)*np.linalg.norm(est_vel))
    perpendicular_est_vel = np.array([-est_vel[1], est_vel[0]])
    cos_perp_angle = pos_vec.dot(perpendicular_est_vel)/(np.linalg.norm(pos_vec)*np.linalg.norm(perpendicular_est_vel))
    angle = np.arccos(cos_angle)
    if cos_perp_angle < 0:
        angle = -angle
        print "target view from port side"
    else:
        print "target view from starboard side"
    if np.isnan(angle):
        print "ERROR: est_pos={}, est_vel={}, own_pos={}".format(est_pos, est_vel, ownship_position)
    return angle

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

def cluster_detections(detections, angles, N_bins=18):
    angles_half = np.linspace(-np.pi, np.pi, N_bins)
    ang_index = np.digitize(angles, angles_half)
    detections_half = np.zeros_like(angles_half)
    tries = np.zeros_like(angles_half)
    for detection, idx in zip(detections, ang_index):
        tries[idx-1] += 1
        detections_half[idx-1] += detection
    P_D = np.zeros_like(angles_half)
    for k, ang in enumerate(angles_half):
        if tries[k] > 0:
            P_D[k] = detections_half[k]/tries[k]
        else:
            P_D[k] = 0
    P_D[-1] = P_D[-2]
    return angles_half, P_D, tries

if __name__ == '__main__':
    chosen_rosbag = '/Users/ewilthil/Documents/autosea_testdata/25-09-2018/filtered_bags/filtered_scenario_11_2018-09-25-14-06-12.bag'
    all_files = glob.glob('/Users/ewilthil/Documents/autosea_testdata/25-09-2018/filtered_bags/filtered_scenario_2*.bag')
    #plot_sample_time_cdf(all_files)
    gate_probability = 0.99
    maximum_velocity = 15
    measurement_mapping = np.array([[1, 0, 0, 0],[0, 0, 1, 0]])
    measurement_covariance_range = 400.0
    measurement_covariance_bearing = 0.001611
    detection_probability = 0.1
    track_gate = autocommon.TrackGate(gate_probability, maximum_velocity)
    measurement_model = autocommon.ConvertedMeasurementModel(measurement_mapping, measurement_covariance_range, measurement_covariance_bearing, detection_probability, 15**2)
    detections_all = []
    aspect_angle_all = []
    for chosen_rosbag in all_files:
        detections, angle = detect_based_on_ais(chosen_rosbag, measurement_model, track_gate, autoais.known_mmsi['KSX_OSD1'])
        print "processed {}".format(chosen_rosbag)
        detections_all += detections
        aspect_angle_all += angle
    N_bins=36
    angles, P_D, num_measurements = cluster_detections(detections_all, aspect_angle_all, N_bins=N_bins)
    ax = plt.subplot(projection='polar')
    ax.set_theta_zero_location("N")
    ax.plot(angles, P_D, 'r')
    #ax.plot(angles+np.pi, P_D[::-1], 'r')
    plt.figure()
    ax = plt.subplot(projection='polar')
    ax.set_theta_zero_location("N")
    ax.bar(angles, num_measurements, width=np.pi/(N_bins-1))
    #ax.bar(angles+np.pi-np.pi/(N_bins-1), num_measurements[::-1], width=np.pi/(N_bins-1))
    plt.show()
