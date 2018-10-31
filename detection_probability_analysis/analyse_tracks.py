from __future__ import division
import numpy as np
import ipdb

import autoseapy.tracking as autotrack

def iterate_pair_of_track_files(radar_track_file, ais_track_file):
    return [(radar_id, radar_track, ais_id, ais_track) for ais_id, ais_track in ais_track_file.items() for radar_id, radar_track in radar_track_file.items()]

def get_normalized_distance(radar_track_file, ais_track_file):
    normalized_distance = dict()
    for radar_id, radar_track, ais_id, ais_track in iterate_pair_of_track_files(radar_track_file, ais_track_file):
        radar_timestamps = np.array([estimate.timestamp for estimate in radar_track])
        filtered_ais_track = autotrack.sync_track_list(radar_timestamps, ais_track)
        normalized_distance_all = []
        for k, estimates in enumerate(zip(radar_track, filtered_ais_track)):
            radar_est, ais_est = estimates
            normalized_dist = autotrack.normalized_error_distance(ais_est, radar_est, first_is_true_state=True)
            normalized_distance_all.append(normalized_dist)
        normalized_distance[(radar_id, ais_id)] = (radar_timestamps, np.array(normalized_distance_all))
    return normalized_distance

def get_absolute_distance(radar_track_file, ais_track_file):
    distance_data = dict()
    for radar_id, radar_track, ais_id, ais_track in iterate_pair_of_track_files(radar_track_file, ais_track_file):
        radar_timestamps = np.array([estimate.timestamp for estimate in radar_track])
        filtered_ais_track = autotrack.sync_track_list(radar_timestamps, ais_track)
        distance_all = []
        for k, estimates in enumerate(zip(radar_track, filtered_ais_track)):
            radar_est, ais_est = estimates
            distance = np.linalg.norm(radar_est.est_posterior[[0,2]]-ais_est.est_posterior[[0,2]])
            distance_all.append(distance)
        distance_data[(radar_id, ais_id)] = (radar_timestamps, np.array(distance_all))
    return distance_data

def get_true_tracks(radar_track_file, ais_track_file, distance_threshold=200):
    true_tracks = dict()
    corresponding_mmsi = dict()
    absolute_distance = get_absolute_distance(radar_track_file, ais_track_file)
    for id_pair, data in absolute_distance.items():
        radar_id, mmsi = id_pair
        timestamps, absolute_distance = data
        if np.all(absolute_distance < distance_threshold):
            true_tracks[radar_id] = radar_track_file[radar_id]
            corresponding_mmsi[radar_id] = mmsi
    return true_tracks, corresponding_mmsi

def get_coherence_measure(track_file, average_data=False):
    coherence_data = dict()
    for track_id, track_list in track_file.items():
        coherence_vals = []
        coherence_time = []
        for old_estimate, current_estimate in zip(track_list, track_list[1:]):
            coherence_time.append(current_estimate.timestamp)
            v = np.array([current_estimate.est_posterior[1], current_estimate.est_posterior[3]])
            u = np.array([  current_estimate.est_posterior[0]-old_estimate.est_posterior[0],\
                            current_estimate.est_posterior[2]-old_estimate.est_posterior[2]])
            coherence_vals.append(np.dot(v, u)/(np.linalg.norm(u)*np.linalg.norm(v)))
        average_coherence_vals = np.array([np.mean(coherence_vals[np.max([0,k-5]):k+1]) for k in range(len(coherence_vals))])
        if average_data:
            coherence_data[track_id] = (coherence_time, average_coherence_vals)
        else:
            coherence_data[track_id] = (coherence_time, np.array(coherence_vals))
    return coherence_data

def get_coherent_tracks(track_file, min_coherence):
    coherent_tracks = dict()
    coherence_data = get_coherence_measure(track_file)
    for track_id, data in coherence_data.items():
        _, coherence = data
        if np.all(coherence > min_coherence):
            coherent_tracks[track_id] = track_file[track_id]

    return coherent_tracks

def get_track_existence(track_file):
    existence_probs = dict()
    for track_id, track_list in track_file.items():
        ext_out = []
        for estimate in track_list:
            if isinstance(estimate.existence_probability, float):
                ext_out.append(estimate.existence_probability)
            else:
                ext_out.append(np.sum(estimate.existence_probability[:-1]))
        existence_probs[track_id] = np.array(ext_out)
    return existence_probs

def get_existence_probabilities(track_file):
    existence_probs = dict()
    for track_id, track_list in track_file.items():
        ext_out = []
        time_out = []
        for estimate in track_list:
            time_out.append(estimate.timestamp)
            if isinstance(estimate.existence_probability, float):
                ext_out.append(estimate.existence_probability)
            else:
                ext_out.append(1-estimate.existence_probability[2])
        existence_probs[track_id] = (np.array(time_out), np.array(ext_out))
    return existence_probs
