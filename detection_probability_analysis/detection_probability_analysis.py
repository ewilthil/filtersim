from __future__ import division
import glob
import numpy as np
import matplotlib.pyplot as plt
import autoseapy.bag_operations as autobag
import autoseapy.tracking_common as autocommon
import autoseapy.ais as autoais

RADAR_KEY = 'radar'
munkholmen_mmsi = autoais.known_mmsi['MUNKHOLMEN II']
drone_mmsi = autoais.known_mmsi['KSX_OSD1']

class DataStructure(object):
    def __init__(self, radar, ais, measurements, measurement_timestamps, ownship_pose, ownship_vel, fname):
        self.radar = radar
        self.ais = ais
        self.measurements = measurements
        self.measurement_timestamps = measurement_timestamps
        self.ownship_pose = ownship_pose
        self.ownship_vel = ownship_vel
        self.fname = fname

def load_data(rosbags):
    data_out = []
    for rosbag in rosbags:
        radar, ais, measurements, measurement_timestamps = autobag.bag2tracking_data(rosbag, return_timestamps=True)
        measurements, measurement_timestamps = add_zero_sets(measurements, measurement_timestamps)
        ownship_pose, ownship_vel = autobag.bag2navigation_data(rosbag, measurement_timestamps)
        data_out.append(DataStructure(radar, ais, measurements, measurement_timestamps, ownship_pose, ownship_vel, rosbag))
        print "loaded bag {}".format(rosbag)
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

def calculate_detection_statistics(datasets, measurement_model, track_gate, velocity_threshold):
    for dataset in datasets:
        dataset.detections = dict()
        dataset.aspect_angle = dict()
        dataset.target_range = dict()
        dataset.valid_timestamp_idx = dict()
        for mmsi, ais_track in dataset.ais.items():
            detection_indicator, aspect_angle, target_range, timestamp_idx = is_detected(
                    ais_track,
                    dataset.measurements,
                    dataset.measurement_timestamps,
                    dataset.ownship_pose,
                    measurement_model,
                    track_gate,
                    velocity_threshold)
            dataset.detections[mmsi] = detection_indicator
            dataset.aspect_angle[mmsi] = aspect_angle
            dataset.target_range[mmsi] = target_range
            dataset.valid_timestamp_idx[mmsi] = timestamp_idx
        dataset.detections[RADAR_KEY] = []
        dataset.aspect_angle[RADAR_KEY] = []
        dataset.target_range[RADAR_KEY] = []
        for track_id, radar_track in dataset.radar.items():
            detection_indicator, aspect_angle, target_range, _ = is_detected(
                    radar_track,
                    dataset.measurements,
                    dataset.measurement_timestamps,
                    dataset.ownship_pose,
                    measurement_model,
                    track_gate,
                    velocity_threshold)
            dataset.detections[RADAR_KEY] += detection_indicator
            dataset.aspect_angle[RADAR_KEY] += aspect_angle
            dataset.target_range[RADAR_KEY] += target_range

def is_detected(est_list, measurements_all, measurement_timestamps, ownship_pose, measurement_model, gate, velocity_threshold):
    # Assumes measurements have been synced with est_list
    detection_indicator = []
    aspect_angle = []
    target_range = []
    timestamp_idx = []
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
        timestamp_idx.append(k)
        gated_measurements = gate.gate_estimate(current_ais_estimate, measurements, measurement_model)
        detection_indicator.append(1) if len(gated_measurements) > 0 else detection_indicator.append(0)
        aspect_angle.append(calculate_aspect_angle(current_ais_estimate, current_position))
        target_range.append(current_target_range)
    return detection_indicator, aspect_angle, target_range, timestamp_idx

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

def cluster_detections_angle(datasets, mmsi, N_bins, symmetrize=False):
    detections, angles = [], []
    for dataset in datasets:
        detections += dataset.detections[mmsi]
        angles += dataset.aspect_angle[mmsi]
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

def calculate_moving_average_detection_probability(dataset, mmsi, N_one_sided):
    detections = dataset.detections[mmsi]
    P_D = np.zeros_like(detections, dtype=float)
    for k in range(len(detections)):
        k_lower = k-N_one_sided
        k_upper = k+N_one_sided+1
        if k_lower < 0:
            k_lower = 0
        if k_upper > len(detections):
            k_upper = len(detections)
        P_D[k] = np.mean(detections[k_lower:k_upper], dtype=float)
    return P_D

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
    all_files = glob.glob('/Users/ewilthil/Documents/autosea_testdata/27-09-2018/filtered_bags/*.bag')
    gate_probability = 0.99
    maximum_velocity = 15
    measurement_mapping = np.array([[1, 0, 0, 0],[0, 0, 1, 0]])
    measurement_covariance_range = 400.0
    measurement_covariance_bearing = 0.001611
    detection_probability = 0.1
    track_gate = autocommon.TrackGate(gate_probability, maximum_velocity)
    measurement_model = autocommon.ConvertedMeasurementModel(measurement_mapping, measurement_covariance_range, measurement_covariance_bearing, detection_probability, 15**2)


    chosen_mmsi = (munkholmen_mmsi, drone_mmsi)
    datasets = load_data(all_files)
    filter_data(datasets, chosen_mmsi)
    calculate_detection_statistics(
            datasets,
            measurement_model,
            track_gate,
            1)

    titles = ['Munkholmen', 'OSD', 'Radar']
    polar_ais_fig, polar_ais_ax = setup_polar_fig()
    polar_radar_fig, polar_radar_ax = setup_polar_fig()
    polar_axes = [polar_ais_ax, polar_ais_ax, polar_radar_ax]
    for k, mmsi in enumerate([munkholmen_mmsi, drone_mmsi, RADAR_KEY]):
        angles, P_D, tries = cluster_detections_angle(datasets, mmsi, 36, True)
        plot_angular_detections(polar_axes[k], angles, P_D, titles[k])
        print "average P_D={} for {}".format(np.mean(P_D), titles[k])
    polar_ais_ax.legend(bbox_to_anchor=(1.3, 1))
    polar_ais_ax.set_title('AIS-based probability of detection')
    polar_radar_ax.set_title('Radar-based probability of detection')
    polar_ais_fig.savefig('detection_probability_ais.pdf')
    polar_radar_fig.savefig('detection_probability_radar.pdf')

    pd_fig, pd_ax = plt.subplots(nrows=3, ncols=2)
    pd_row, pd_col = 0, 0
    scenario_numbers = ['1', '2', '4', '6', '7', '15']
    for dataset in datasets:
        plotted = False
        for mmsi in [munkholmen_mmsi, drone_mmsi]:
            P_D = calculate_moving_average_detection_probability(dataset, mmsi, 10)
            if np.any(np.array([dataset.fname.find("_"+scen_num+"_") for scen_num in scenario_numbers]) > 0):
                time = dataset.ownship_vel[0,dataset.valid_timestamp_idx[mmsi]]
                if len(time) > 0:
                    pd_ax[pd_row, pd_col].plot(time-time[0],P_D, lw=2)
                    pd_ax[pd_row, pd_col].set_ylim(0, 1)
                    pd_ax[pd_row, pd_col].set_xlabel('time [s]')
                    pd_ax[pd_row, pd_col].set_ylabel('$P_D$')
                    plotted = True
        if plotted:
            pd_row += 1
            if pd_row > 2:
                pd_row = 0
                pd_col += 1
        #temporal_fig, temporal_ax = plt.subplots(nrows=4)
        #temporal_ax[0].plot(P_D)
        #temporal_ax[0].set_ylim(0,1)
        #temporal_ax[1].plot(np.rad2deg(np.array(dataset.aspect_angle[drone_mmsi])))
        #temporal_ax[2].plot(dataset.target_range[drone_mmsi])
        #temporal_ax[3].plot(np.rad2deg(dataset.ownship_vel[-1,dataset.valid_timestamp_idx[drone_mmsi]]))
        #temporal_ax[0].set_title(dataset.fname)
    plt.tight_layout()
    pd_fig.savefig('varying_pd.pdf')
    plt.show()
