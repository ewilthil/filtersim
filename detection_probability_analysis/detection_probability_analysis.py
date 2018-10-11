from __future__ import division
import glob
import numpy as np
import matplotlib.pyplot as plt
import autoseapy.bag_operations as autobag
import autoseapy.tracking_common as autocommon
import autoseapy.ais as autoais

class DataStructure(object):
    def __init__(self, radar, ais, measurements, measurement_timestamps, ownship_pose, ownship_vel):
        self.radar = radar
        self.ais = ais
        self.measurements = measurements
        self.measurement_timestamps = measurement_timestamps
        self.ownship_pose = ownship_pose
        self.ownship_vel = ownship_vel

def load_data(rosbags):
    data_out = []
    for rosbag in rosbags:
        radar, ais, measurements, measurement_timestamps = autobag.bag2tracking_data(rosbag, return_timestamps=True)
        measurements, measurement_timestamps = add_zero_sets(measurements, measurement_timestamps)
        ownship_pose, ownship_twist = autobag.bag2navigation_data(rosbag, measurement_timestamps)
        data_out.append(DataStructure(radar, ais, measurements, measurement_timestamps, ownship_pose, ownship_twist))
    return data_out

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

def filter_data(datasets, chosen_mmsi):
    for dataset in datasets:
        filtered_ais = {mmsi : dataset.ais[mmsi] for mmsi in chosen_mmsi if mmsi in dataset.ais.keys()}
        dataset.ais = filtered_ais

def calculate_detection_statistics(datasets, measurement_model, track_gate, velocity_threshold=1):
    ais_statistics = dict()
    radar_statistics = dict(detections=[], aspect_angle=[], target_range=[])
    for dataset in datasets:
        for mmsi, ais_track in dataset.ais.items():
            detection_indicator, aspect_angle, target_range = is_detected(
                    ais_track,
                    dataset.measurements,
                    dataset.measurement_timestamps,
                    dataset.ownship_pose,
                    measurement_model,
                    track_gate,
                    velocity_threshold)
            if mmsi not in ais_statistics.keys():
                ais_statistics[mmsi] = dict(detections=[], aspect_angle=[], target_range=[])
            ais_statistics[mmsi]['detections'] += [detection_indicator]
            ais_statistics[mmsi]['aspect_angle'] += [aspect_angle]
            ais_statistics[mmsi]['target_range'] += [target_range]
        for track_id, radar_track in dataset.radar.items():
            detection_indicator, aspect_angle, target_range = is_detected(
                    radar_track,
                    dataset.measurements,
                    dataset.measurement_timestamps,
                    dataset.ownship_pose,
                    measurement_model,
                    track_gate,
                    velocity_threshold)
            radar_statistics['detections'] += [detection_indicator]
            radar_statistics['aspect_angle'] += [aspect_angle]
            radar_statistics['target_range'] += [target_range]
    return radar_statistics, ais_statistics

def is_detected(est_list, measurements_all, measurement_timestamps, ownship_pose, measurement_model, gate, velocity_threshold):
    # Assumes measurements have been synced with est_list
    detection_indicator = []
    aspect_angle = []
    target_range = []
    estimate_timestamps = [est.timestamp for est in est_list if est.timestamp >= np.min(measurement_timestamps) and est.timestamp <= np.max(measurement_timestamps)]
    est_alive_indexes = np.digitize(estimate_timestamps, measurement_timestamps, right=True)
    for k_est, k in enumerate(est_alive_indexes):#, measurements in enumerate(measurements_all):
        measurements = measurements_all[k]
        current_position = ownship_pose[1:3,k]
        measurement_model.update_ownship(current_position)
        current_ais_estimate = est_list[k_est]
        current_ais_velocity = np.array([current_ais_estimate.est_posterior[1], current_ais_estimate.est_posterior[3]])
        current_ais_position = np.array([current_ais_estimate.est_posterior[0], current_ais_estimate.est_posterior[2]])
        current_target_range = np.linalg.norm(current_ais_position-current_position)
        if np.linalg.norm(current_ais_velocity) < velocity_threshold or current_target_range > 1800:
            continue
        gated_measurements = gate.gate_estimate(current_ais_estimate, measurements, measurement_model)
        detection_indicator.append(1) if len(gated_measurements) > 0 else detection_indicator.append(0)
        aspect_angle.append(calculate_aspect_angle(current_ais_estimate, current_position))
        target_range.append(current_target_range)
    return detection_indicator, aspect_angle, target_range

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

def cluster_detections_angle(detections_all, angles_all, N_bins, symmetrize=False):
    detections = [detection for det_list in detections_all for detection in det_list]
    angles = [angle for ang_list in angles_all for angle in ang_list]
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

def cluster_detections_range(detections_all, target_range_all, N_bins):
    detections = [detection for det_list in detections_all for detection in det_list]
    target_range = [t_range for range_list in target_range_all for t_range in range_list]
    new_target_range = np.linspace(np.min(target_range), np.max(target_range), N_bins)
    range_index = np.digitize(target_range, new_target_range, right=True)
    new_detections = np.zeros_like(new_target_range)
    tries = np.zeros_like(new_target_range)
    for detection, idx in zip(detections, range_index):
        tries[idx-1] += 1
        new_detections[idx-1] += detection
    P_D = np.zeros_like(new_target_range)
    for k, ang in enumerate(new_target_range):
        if tries[k] > 0:
            P_D[k] = new_detections[k]/tries[k]
        else:
            P_D[k] = 0
    return new_target_range, P_D, tries

def cluster_detections_temporal(detections_all, N_one_sided):
    P_D_all = []
    for detections in detections_all:
        P_D = np.zeros_like(detections, dtype=float)
        for k in range(len(detections)):
            k_lower = k-N_one_sided
            k_upper = k+N_one_sided+1
            if k_lower < 0:
                k_lower = 0
            if k_upper > len(detections):
                k_upper = len(detections)
            P_D[k] = np.mean(detections[k_lower:k_upper], dtype=float)
        P_D_all.append(P_D)
    return P_D_all

def plot_angular_detections(ax, angles, P_D, label):
    ax.plot(angles, P_D, label=label, lw=2)
    ax.set_ylim(0, 1)
    ax.set_title('Detection probability')

def plot_range_detections(ax, target_range, P_D):
    ax.plot(target_range, P_D)

def plot_aspect_angle_vs_range(ax, angles, target_range, title, color):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.scatter(angles, target_range)
    ax.set_ylim(0, np.max(target_range))
    ax.set_xticks((0, np.pi/2, np.pi, 3*np.pi/2))
    ax.set_xticklabels(('Bow', 'Starboard', 'Stern', 'Port'))
    ax.set_title('Detection probability')

def plot_angular_num_scans(ax, angles, num_scans):
    pass

def plot_range_num_scans(ax, target_range, num_scans):
    ax.bar(target_range, num_scans, target_range[1]-target_range[0])
    ax.set_xlim(target_range[0], target_range[-1])

def setup_polar_fig():
    fig, ax = plt.subplots(subplot_kw={'projection' : 'polar'})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.grid(ls='-', color='#999999')
    ax.set_xticks((0, np.pi/2, np.pi, 3*np.pi/2))
    return fig, ax

    


if __name__ == '__main__':
    chosen_rosbag = '/Users/ewilthil/Documents/autosea_testdata/25-09-2018/filtered_bags/filtered_scenario_11_2018-09-25-14-06-12.bag'
    all_files = glob.glob('/Users/ewilthil/Documents/autosea_testdata/2[57]-09-2018/filtered_bags/filtered_scenario_*.bag')
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
    range_all = []


    chosen_mmsi = (autoais.known_mmsi['MUNKHOLMEN II'], autoais.known_mmsi['KSX_OSD1'])
    datasets = load_data(all_files)
    filter_data(datasets, chosen_mmsi)
    radar_detection_data, ais_detection_data = calculate_detection_statistics(datasets, measurement_model, track_gate, velocity_threshold=1)

    munkholmen_data = ais_detection_data[autoais.known_mmsi['MUNKHOLMEN II']]
    drone_data = ais_detection_data[autoais.known_mmsi['KSX_OSD1']]
    titles = ['Munkholmen', 'OSD', 'Radar']
    
    polar_ais_fig, polar_ais_ax = setup_polar_fig()
    polar_radar_fig, polar_radar_ax = setup_polar_fig()
    polar_axes = [polar_ais_ax, polar_ais_ax, polar_radar_ax]
    range_fig, range_ax = plt.subplots(nrows=2)
    for k, data in enumerate([munkholmen_data, drone_data, radar_detection_data]):
        angles, P_D, tries = cluster_detections_angle(data['detections'], data['aspect_angle'], 36, True)
        plot_angular_detections(polar_axes[k], angles, P_D, titles[k])
        print "average P_D={} for {}".format(np.mean(P_D), titles[k])
        target_range, P_D, tries = cluster_detections_range(data['detections'], data['target_range'], 10)
        plot_range_detections(range_ax[0], target_range, P_D)
        plot_range_num_scans(range_ax[1], target_range, tries)
    polar_ais_ax.legend(bbox_to_anchor=(1.3, 1))

    P_D_all = cluster_detections_temporal(drone_data['detections'], 15)
    for P_D, aspect_angle in zip(P_D_all, data['aspect_angle']):
        temporal_fig, temporal_ax = plt.subplots(nrows=2)
        temporal_ax[0].plot(P_D)
        temporal_ax[0].set_ylim(0,1)
        temporal_ax[1].plot(np.rad2deg(np.array(aspect_angle)))
    plt.show()




    plt.tight_layout()
    polar_ais_ax.set_title('AIS-based probability of detection')
    polar_radar_ax.set_title('Radar-based probability of detection')
    polar_ais_fig.savefig('detection_probability_ais.pdf')
    polar_radar_fig.savefig('detection_probability_radar.pdf')
    plt.show()
