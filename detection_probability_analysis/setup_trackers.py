import numpy as np

import autoseapy.tracking_common as autocommon
import autoseapy.tracking as autotrack
import autoseapy.track_management as automanagers
import autoseapy.track_initiation as autoinit
import autoseapy.hidden_markov_model as hmm_models

target_process_noise_covariance = 0.05**2
gate_probability = 0.99
maximum_velocity = 15
measurement_mapping = np.array([[1, 0, 0, 0],[0, 0, 1, 0]])
min_cartesian_cov = 0*15**2
measurement_covariance = min_cartesian_cov*np.identity(2)
measurement_covariance_range = 20**2
simulated_measurement_covariance = 10**2*np.identity(2)
measurement_covariance_bearing = np.deg2rad(2.2997)**2
survival_probability = 1.0
new_target_probability = 0.0
init_prob = 0.2
markov_init_prob = np.array([0.15, 0.15, 0.7])
conf_threshold = 0.99
term_threshold = 0.1
P_low = 0.8
P_high = P_low
P_birth = 0.0
P_term = 1-survival_probability
low_PD = 0.3
high_PD = 0.8
single_PD = 0.8
hmm_transition_matrix = np.array([[P_low, 1-P_low], [1-P_high, P_high]])
hmm_emission_matrix = np.array([[1-low_PD, 1-high_PD], [low_PD, high_PD]])
hmm_initial_probability = np.array([0.5, 0.5])

REAL = 'real'
SIMULATED = 'simulated'



ipda_transition_matrix = np.array([[survival_probability*P_low, survival_probability*(1-P_low), 1-survival_probability], [survival_probability*(1-P_high), survival_probability*P_high, 1-survival_probability], [new_target_probability/2, new_target_probability/2, 1-new_target_probability]])

# Common stuff
def setup_target_model(covariance):
    return autocommon.DWNAModel(covariance)
target_model = setup_target_model(target_process_noise_covariance)

def setup_track_gate(gate_prob, max_vel):
    return autocommon.TrackGate(gate_prob, max_vel)
track_gate = setup_track_gate(gate_probability, maximum_velocity)
    
# Vanilla IPDAs
def setup_ipda_manager(measurement_type=REAL):
    if measurement_type == REAL:
        measurement_model = autocommon.ConvertedMeasurementModel(measurement_mapping, measurement_covariance_range, measurement_covariance_bearing, single_PD, min_cartesian_cov)
    else:
        measurement_model = autocommon.CartesianMeasurementModel(measurement_mapping, simulated_measurement_covariance, single_PD)
    tracker = autotrack.IPDAFTracker(target_model, measurement_model, track_gate, survival_probability)
    init = autoinit.IPDAInitiator(tracker, init_prob, conf_threshold, term_threshold)
    term = autotrack.IPDATerminator(term_threshold)
    ipda_track_manager = automanagers.Manager(tracker, init, term)
    return ipda_track_manager


# HMM-based IPDA
def setup_hmm_ipda_manager(measurement_type=REAL):
    detection_model = hmm_models.HiddenMarkovModel(hmm_transition_matrix, hmm_emission_matrix, hmm_initial_probability, [low_PD, high_PD])
    if measurement_type == REAL:
        measurement_model = autocommon.ConvertedMeasurementModelMarkovDetectionProbability(measurement_mapping, measurement_covariance_range, measurement_covariance_bearing, detection_model, min_cartesian_cov)
    else:
        measurement_model = autocommon.CartesianMeasurementModelMarkovDetectionProbability(measurement_mapping, simulated_measurement_covariance, detection_model)
    tracker = autotrack.IPDAFTracker(target_model, measurement_model, track_gate, survival_probability)
    init = autoinit.IPDAInitiator(tracker, init_prob, conf_threshold, term_threshold)
    term = autotrack.IPDATerminator(term_threshold)
    track_manager = automanagers.Manager(tracker, init, term)
    return track_manager

# "Vanilla" MC2 IPDA
def setup_mc2_ipda_manager(measurement_type=REAL):
    if measurement_type == REAL:
        van_mc2_measurement_model = autocommon.ConvertedMeasurementModel(measurement_mapping, measurement_covariance_range, measurement_covariance_bearing, [0, single_PD], min_cartesian_cov)
    else:
        van_mc2_measurement_model = autocommon.CartesianMeasurementModel(measurement_mapping, simulated_measurement_covariance, [0, single_PD])
    vanilla_mc2_ipda_tracker = autotrack.MC2IPDAFTracker(target_model, van_mc2_measurement_model, track_gate, ipda_transition_matrix)
    vanilla_mc2_ipda_init = autoinit.MC2IPDAInitiator(vanilla_mc2_ipda_tracker, markov_init_prob, conf_threshold, term_threshold)
    vanilla_mc2_ipda_term = autotrack.MC2IPDATerminator(term_threshold)
    vanilla_ipda_mc2_manager = automanagers.Manager(vanilla_mc2_ipda_tracker, vanilla_mc2_ipda_init, vanilla_mc2_ipda_term)
    return vanilla_ipda_mc2_manager

# Extra MC2 IPDA
def setup_mcn_ipda_manager(measurement_type=REAL):
    if measurement_type == REAL:
        mc2_measurement_model = autocommon.ConvertedMeasurementModel(measurement_mapping, measurement_covariance_range, measurement_covariance_bearing, [low_PD, high_PD], min_cartesian_cov)
    else:
        mc2_measurement_model = autocommon.CartesianMeasurementModel(measurement_mapping, simulated_measurement_covariance, [low_PD, high_PD])
    mc2_ipda_tracker = autotrack.MC2IPDAFTracker(target_model, mc2_measurement_model, track_gate, ipda_transition_matrix)
    mc2_ipda_init = autoinit.MC2IPDAInitiator(mc2_ipda_tracker, markov_init_prob, conf_threshold, term_threshold)
    mc2_ipda_term = autotrack.MC2IPDATerminator(term_threshold)
    ipda_mc2_manager = automanagers.Manager(mc2_ipda_tracker, mc2_ipda_init, mc2_ipda_term)
    return ipda_mc2_manager
