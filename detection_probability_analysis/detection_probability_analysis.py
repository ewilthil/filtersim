from __future__ import division
import glob
import numpy as np
import matplotlib.pyplot as plt
import autoseapy.bag_operations as autobag
import autoseapy.tracking_common as autocommon
import autoseapy.ais as autoais
from scipy.stats import poisson
import autoseapy.visualization as autovis

RADAR_KEY = 'radar'
munkholmen_mmsi = autoais.known_mmsi['MUNKHOLMEN II']
drone_mmsi = autoais.known_mmsi['KSX_OSD1']
telemetron_mmsi = autoais.known_mmsi['TELEMETRON']

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

def cluster_detections_range(datasets, mmsi, N_bins):
    detections, target_range = [], []
    for dataset in datasets:
        detections += dataset.detections[mmsi]
        target_range += dataset.target_range[mmsi]
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

def add_navigation_data(data_bag, navigation_bag):
    import rosbag
    with rosbag.Bag('output_bag.bag', 'w') as outbag:
        for topic, message, timestamp in rosbag.Bag(data_bag).read_messages():
            outbag.write(topic, message, timestamp)
        for topic, message, timestamp in rosbag.Bag(navigation_bag).read_messages():
            if topic in ['/seapath/pose', '/seapath/twist']:
                outbag.write(topic, message, timestamp)

def add_ais_data(data_bag, ais_bag):
    import rosbag
    with rosbag.Bag('output_bag.bag', 'w') as outbag:
        for topic, message, timestamp in rosbag.Bag(data_bag).read_messages():
            outbag.write(topic, message, timestamp)
        for topic, message, timestamp in rosbag.Bag(ais_bag).read_messages():
            if topic in ['/ais/ais_timestamped']:
                outbag.write(topic, message, timestamp)

if __name__ == '__main__':
    all_files = glob.glob('/Users/ewilthil/Documents/autosea_testdata/25-09-2018/filtered_bags/filtered_scenario_6_2018-09-25-11-28-47.bag')
    #all_files = glob.glob('*.bag')
    gate_probability = 0.99
    maximum_velocity = 15
    measurement_mapping = np.array([[1, 0, 0, 0],[0, 0, 1, 0]])
    measurement_covariance_range = 400.0
    measurement_covariance_bearing = 0.001611
    detection_probability = 0.1
    track_gate = autocommon.TrackGate(gate_probability, maximum_velocity)
    measurement_model = autocommon.ConvertedMeasurementModel(measurement_mapping, measurement_covariance_range, measurement_covariance_bearing, detection_probability, 15**2)


    chosen_mmsi = (munkholmen_mmsi, drone_mmsi, telemetron_mmsi)
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
    polar_ais_ax.set_title('Probability of detection vs AIS aspect angle')
    polar_radar_ax.set_title('Probability of detection vs radar aspect angle')
    polar_ais_fig.savefig('detection_probability_ais.pdf')
    polar_radar_fig.savefig('detection_probability_radar.pdf')

    range_fig, range_ax = plt.subplots(nrows=2)
    labels = ['Tugboat', 'Glass fiber boat', 'Radar tracks']
    n_range_bins = 20
    for k, mmsi in enumerate([munkholmen_mmsi, drone_mmsi, RADAR_KEY]):
        ranges, PD, num_measurements = cluster_detections_range(datasets, mmsi, n_range_bins)
        range_ax[0].plot(ranges, PD, label=labels[k])
    range_bins = dict()
    for dataset in datasets:
        for k, mmsi in enumerate([munkholmen_mmsi, drone_mmsi, RADAR_KEY]):
            if mmsi not in range_bins.keys():
                range_bins[mmsi] = []
            range_bins[mmsi] += dataset.target_range[mmsi]
    range_bin_dat = [range_bins[munkholmen_mmsi], range_bins[drone_mmsi], range_bins[RADAR_KEY]]
    range_ax[1].hist(range_bin_dat, bins=n_range_bins, label=labels)
    [ax.legend() for ax in range_ax]
    range_ax[0].set_ylim(0, 1)
    range_ax[0].set_title('$P_D$ based on range')
    range_ax[0].set_ylabel('$P_D$')
    [ax.set_xlabel('range [m]') for ax in range_ax]
    range_ax[1].set_ylabel('Number of samples')
    range_ax[1].set_ylabel('Total number of samples')
    range_fig.savefig('PD_vs_range.pdf')

    pd_fig, pd_ax = plt.subplots(nrows=2)
    mh_fig, mh_ax = plt.subplots(nrows=2)
    f_fig = plt.figure(figsize=(7,4))
    f_ax = f_fig.add_axes((0.15, 0.15, 0.8, 0.75))
    min_pd = {drone_mmsi : [], munkholmen_mmsi : []}
    min_range = {drone_mmsi : [], munkholmen_mmsi : []}
    for dataset in datasets:
        fig, ax = plt.subplots(nrows=2)
        for mmsi, target in zip([drone_mmsi, munkholmen_mmsi], ['OSD', 'MH2']):
            P_D = calculate_moving_average_detection_probability(dataset, mmsi, 10)
            current_range = dataset.target_range[mmsi]
            time = dataset.ownship_vel[0,dataset.valid_timestamp_idx[mmsi]]
            if len(time) > 0:
                min_range[mmsi].append(np.min(current_range))
                min_pd[mmsi].append(np.min(P_D))
                ax[0].plot(time-time[0],P_D, lw=2, label=target)
                ax[0].set_ylim(0, 1)
                ax[1].set_xlabel('time [s]')
                ax[0].set_ylabel('$P_D$')
                ax[1].plot(time-time[0], current_range, lw=2)
            if dataset.fname == '/Users/ewilthil/Documents/autosea_testdata/25-09-2018/filtered_bags/filtered_scenario_6_2018-09-25-11-28-47.bag':
                f_ax.plot(time-time[0], P_D, lw=2, label=target)
                f_ax.grid('on')
                f_ax.set_ylim(0, 1)
                f_ax.set_xlabel('time [s]')
                f_ax.set_ylabel('$P_D$')
                f_ax.set_title('Detection probability based on AIS-centered gate')
        ax[0].legend()
        ax[1].grid()
        ax[1].set_ylabel('Range')
        ax[1].set_ylim(0, 1800)
        ax[1].set_yticks(np.arange(0, 2000, 200))
    scat_fig, scat_ax = plt.subplots()
    for mmsi, color in zip([drone_mmsi, munkholmen_mmsi], plt.rcParams['axes.color_cycle']):
        scat_ax.scatter(min_range[mmsi], min_pd[mmsi],c=color)
    f_ax.legend()

    pd_ax[0].set_title('Detection probability')

    # moving average number of measurements (i.e. false alarms)
    hist_fig, hist_ax = plt.subplots()
    num_avg = 10
    num_all = []
    means_all = []
    for dataset in datasets:
        means = []
        num_measurements = []
        for k, _ in enumerate(dataset.measurements):
            k_min = np.max([0, k-num_avg])
            k_max = np.min([len(dataset.measurements), k+num_avg])
            means.append(np.mean([len(z) for z in dataset.measurements[k_min:k_max]]))
            num_measurements.append(len(dataset.measurements[k]))
        num_all.append(num_measurements)
        time = np.array(dataset.measurement_timestamps)
        pd_ax[1].plot(time-time[0], means, lw=2)
        means_all.append(means)
    x_vals = np.arange(np.max(np.max(num_all))+1)
    hist_ax.hist(num_all, bins=x_vals,normed=True, edgecolor='none')
    for num in num_all:
        pdfs = poisson(mu=np.mean(num)).pmf(x_vals)
        hist_ax.plot(x_vals, pdfs, 'k', lw=2)
    pd_ax[1].set_title('Number of measurements')

    hist_ax.set_title('Measurements per scan')
    hist_fig.savefig('hist_num_measurements.pdf')
    
    chosen_dataset = datasets[0]
    chosen_target = drone_mmsi
    P_D_drone_1 = calculate_moving_average_detection_probability(chosen_dataset, drone_mmsi, 10)
    P_D_munkholmen = calculate_moving_average_detection_probability(chosen_dataset, munkholmen_mmsi, 10)
    P_D_drone_2 = calculate_moving_average_detection_probability(chosen_dataset, drone_mmsi, 5)
    #time = np.array(dataset.measurement_timestamps)
    drone_time = dataset.ownship_vel[0,dataset.valid_timestamp_idx[drone_mmsi]]
    drone_time = drone_time-drone_time[0]
    munkholmen_time = dataset.ownship_vel[0,dataset.valid_timestamp_idx[munkholmen_mmsi]]
    munkholmen_time = munkholmen_time-munkholmen_time[0]

    
    det_fig = plt.figure(figsize=(7,4))
    det_ax = det_fig.add_axes((0.1, 0.15, 0.85, 0.75))
    det_ax.grid('on')
    l = det_ax.plot(drone_time, P_D_drone_1, label='OSD, $N=10$', lw=2)
    det_ax.plot(drone_time, P_D_drone_2, label='OSD, $N=5$', ls='--', color=l[0].get_color(), lw=2)
    det_ax.plot(munkholmen_time, P_D_munkholmen, label='MH II', lw=2)
    loc = det_ax.legend(loc=(4.2/6, 2.9/5))
    det_ax.set_ylim(0, 1.1)
    det_ax.set_yticks(np.arange(0,1.1, 0.2))
    det_ax.set_title('Detection probability based on AIS-centered validation gate')
    det_ax.set_ylabel('$P_D$')
    det_ax.set_xlabel('Time [s]')
    det_ax.set_axisbelow(True)
    det_fig.savefig('detection_probability.pdf')
    plt.show()

    landmark_measurements = []
    landmark_time = []
    for measurements in chosen_dataset.measurements:
        t_added = False 
        for measurement in measurements:
            z_pos = measurement.value
            if z_pos[0] < 1500 and z_pos[0] > 1200 and z_pos[1] > 300 and z_pos[1] < 700:
                landmark_measurements.append(z_pos)
                t_added = True
                landmark_time.append(measurement.timestamp)
    landmark_measurements = np.array(landmark_measurements)
    landmark_mean = np.mean(landmark_measurements, axis=0)
    landmark_data = []
    landmark_mmsi = 123456789
    chosen_dataset.ais[landmark_mmsi] = []
    for t in landmark_time:
        mark_est = autocommon.Estimate(t, np.array([landmark_mean[0], 0, landmark_mean[1], 0]), np.identity(4), True, landmark_mmsi)
        chosen_dataset.ais[landmark_mmsi].append(mark_est)



    xy_fig = plt.figure(figsize=(7,7))
    xy_ax = xy_fig.add_axes((0.15, 0.15, 0.8, 0.8))
    for measurements in chosen_dataset.measurements:
        for measurement in measurements:
            z_line, = xy_ax.plot(measurement.value[1], measurement.value[0], '.', color='#aaaaaa')
    ownship_line, = xy_ax.plot(ownship_pose[2,:], ownship_pose[1,:], 'k--')
    end_position, = xy_ax.plot(ownship_pose[2,-1], ownship_pose[1,-1], 'ko')
    map_labels = ['OSD', 'MH II', 'LM']
    for mmsi, label in zip([drone_mmsi, munkholmen_mmsi, landmark_mmsi], map_labels):
        position = np.array([estimate.est_posterior[[0,2]] for estimate in chosen_dataset.ais[mmsi]])
        target_line, = xy_ax.plot(position[:,1], position[:,0], 'k-')
        xy_ax.plot(position[-1,1], position[-1,0], 'ko')
        xy_ax.text(position[-1,1], position[-1,0]-250, label)

    xy_ax.legend([ownship_line, target_line, end_position, z_line], ['Ownship', 'Targets', 'End position', 'Measurements'])
    xy_ax.set_aspect('equal')
    xy_ax.set_ylabel('North [m]')
    xy_ax.set_xlabel('East [m]')

    xy_fig.savefig('scenario_overview.pdf')
    xy_ax.set_xlim(0, 3000)
    xy_ax.set_ylim(1000, 3000)
    xy_fig.savefig('scenario_overview_closeup.pdf')
    plt.show()

    # Total number of measurements (histogram)
    plt.tight_layout()
    pd_fig.savefig('varying_pd.pdf')
    plt.show()
