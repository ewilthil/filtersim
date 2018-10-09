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

def is_detected(est_list, measurements_all, measurement_timestamps, ownship_pose, measurement_model, gate):
    # Assumes measurements have been synced with est_list
    detection_indicator = []
    aspect_angle = []
    estimate_timestamps = [est.timestamp for est in est_list]
    est_alive_indexes = np.digitize(estimate_timestamps, measurement_timestamps, right=True)
    if np.any(np.diff(est_alive_indexes) != 1):
        print "Warning: Diff"
    for k_est, k in enumerate(est_alive_indexes):#, measurements in enumerate(measurements_all):
        measurements = measurements_all[k]
        current_position = ownship_pose[1:3,k]
        measurement_model.update_ownship(current_position)
        current_ais_estimate = est_list[k_est]
        current_ais_velocity = np.array([current_ais_estimate.est_posterior[1], current_ais_estimate.est_posterior[3]])
        current_ais_position = np.array([current_ais_estimate.est_posterior[0], current_ais_estimate.est_posterior[2]])
        if np.linalg.norm(current_ais_velocity) < 1 or np.linalg.norm(current_ais_position-current_position) > 1800:
            continue
        gated_measurements = gate.gate_estimate(current_ais_estimate, measurements, measurement_model)
        detection_indicator.append(1) if len(gated_measurements) > 0 else detection_indicator.append(0)
        aspect_angle.append(calculate_aspect_angle(current_ais_estimate, current_position))
    return detection_indicator, aspect_angle

def detect_based_on_ais(chosen_rosbag, measurement_model, gate, chosen_mmsi):
    ais_data, measurements_all, measurement_timestamps = autobag.bag2raw_data(chosen_rosbag, True)
    measurements_all, measurement_timestamps = add_zero_sets(measurements_all, measurement_timestamps)
    ais_data = autobag.synchronize_track_file_to_timestamps(ais_data, measurement_timestamps, q=0)
    ownship_pose, ownship_twist = autobag.bag2navigation_data(chosen_rosbag, measurement_timestamps)
    detection_indicator_all, aspect_angle_all = [], []
    for mmsi in chosen_mmsi:
        if mmsi not in ais_data.keys():
            continue
        est_list = ais_data[mmsi]
        detection_indicator, aspect_angle = is_detected(est_list, measurements_all, measurement_timestamps, ownship_pose, measurement_model, gate)
        detection_indicator_all += detection_indicator
        aspect_angle_all += aspect_angle
    return detection_indicator_all, aspect_angle_all

def detect_based_on_radar(chosen_rosbag, measurement_model, gate):
    radar_data, ais_data, measurements_all, measurement_timestamps = autobag.bag2tracking_data(chosen_rosbag, True)
    measurements_all, measurement_timestamps = add_zero_sets(measurements_all, measurement_timestamps)
    ownship_pose, ownship_twist = autobag.bag2navigation_data(chosen_rosbag, measurement_timestamps)
    detection_indicator_all, aspect_angle_all = [], []
    for track_id, est_list in radar_data.items():
        detection_indicator, aspect_angle = is_detected(est_list, measurements_all, measurement_timestamps, ownship_pose, measurement_model, gate)
        detection_indicator_all += detection_indicator
        aspect_angle_all += aspect_angle
    return detection_indicator_all, aspect_angle_all

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

def cluster_detections(detections, angles, N_bins, symmetrize=False):
    new_angles = np.linspace(-np.pi, np.pi, N_bins)
    ang_index = np.digitize(angles, new_angles)
    new_detections = np.zeros_like(new_angles)
    tries = np.zeros_like(new_angles)
    for detection, idx in zip(detections, ang_index):
        tries[idx-1] += 1
        new_detections[idx-1] += detection
    P_D = np.zeros_like(new_angles)
    for k, ang in enumerate(new_angles):
        if tries[k] > 0:
            P_D[k] = new_detections[k]/tries[k]
        else:
            P_D[k] = 0
    if symmetrize:
        for lower_idx in range(int(np.floor(len(new_angles)/2.))):
            upper_idx = -(lower_idx+1)
            total_tries = tries[lower_idx]+tries[upper_idx]
            new_PD = (P_D[lower_idx]*tries[lower_idx]+P_D[upper_idx]*tries[upper_idx])/(1.*total_tries)
            P_D[upper_idx] = new_PD
            P_D[lower_idx] = new_PD
            tries[upper_idx] = total_tries
            tries[lower_idx] = total_tries
    else:
        P_D[-1] = P_D[0]
    return new_angles, P_D, tries

if __name__ == '__main__':
    chosen_rosbag = '/Users/ewilthil/Documents/autosea_testdata/25-09-2018/filtered_bags/filtered_scenario_11_2018-09-25-14-06-12.bag'
    all_files = glob.glob('/Users/ewilthil/Documents/autosea_testdata/2[57]-09-2018/filtered_bags/filtered_scenario*.bag')
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
        detections, angle = detect_based_on_ais(chosen_rosbag, measurement_model, track_gate, [autoais.known_mmsi['MUNKHOLMEN II']])
        #detections, angle = detect_based_on_radar(chosen_rosbag, measurement_model, track_gate)
        print "processed {}".format(chosen_rosbag)
        detections_all += detections
        aspect_angle_all += angle
    N_bins=36
    sym_fname = {True : "symmetric", False : "non_symmetric"}
    for symmetrize in [True, False]:
        plt.figure()
        angles, P_D, num_measurements = cluster_detections(detections_all, aspect_angle_all, N_bins=N_bins, symmetrize=symmetrize)
        ax = plt.subplot(projection='polar')
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.plot(angles, P_D, 'r')
        ax.set_ylim(0, 1)
        ax.set_title('Detection probability')
        plt.savefig("munkholmen_{}_PD.pdf".format(sym_fname[symmetrize]))
        plt.figure()
        ax = plt.subplot(projection='polar')
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        bar_width = (2*np.pi)/N_bins
        ax.bar(angles-bar_width/2., num_measurements, width=bar_width)
        ax.set_title('Number of samples')
        plt.savefig("munkholmen_{}_num.pdf".format(sym_fname[symmetrize]))
    plt.show()
