import numpy as np
import filtersim.tracking as tracking
import autoseapy.tracking as autotrack
from scipy.stats import chi2, multivariate_normal
from ipdb import set_trace

class m_of_n(object):
    def __init__(self, M, N, max_vel):
        self.initiators = []
        self.min_available_track_idx = 1 # 0 is for ownship (in future stuff)
        self.preliminary_indices = dict()
        self.max_vel = max_vel
        self.M = M
        self.N = N

    def __repr__(self):
        pass

    def form_new_tracks(self, measurements):
        new_initiators = []
        new_estimates = []
        logged_estimates = [] # The first time step
        for measurement in measurements:
            associated = False
            for init in self.initiators:
                dt = measurement.timestamp-init.timestamp
                vel = (measurement.value-init.value)/(1.*dt)
                if np.linalg.norm(vel) < self.max_vel:
                    associated = True
                    est_1, est_2 = tracking.Estimate.from_measurement(init, measurement)
                    est_1.track_index = self.min_available_track_idx
                    est_2.track_index = self.min_available_track_idx
                    self.min_available_track_idx += 1
                    new_estimates.append(est_2)
                    logged_estimates.append(est_1)
            if not associated:
                new_initiators.append(measurement)
        self.initiators = new_initiators
        return new_estimates, logged_estimates
                    

    def update_track_status(self, current_preliminary_estimates):
        confirmed_estimates = []
        preliminary_estimates = []
        for est in current_preliminary_estimates:
            if est.track_index not in self.preliminary_indices.keys():
                self.preliminary_indices[est.track_index] = (0,0)
                preliminary_estimates.append(est)
            else:
                m, n = self.preliminary_indices[est.track_index]
                n += 1
                if len(est.measurements) > 0:
                    m += 1
                if m < self.M and n < self.N:
                    preliminary_estimates.append(est)
                elif m >= self.M and n <= self.N:
                    confirmed_estimates.append(est)
                self.preliminary_indices[est.track_index] = (m, n)
        return confirmed_estimates, preliminary_estimates

    def fuse_tracks(self, estimate_list):
        new_estimates = []
        while len(estimate_list) > 0:
            est_under_test = estimate_list.pop(0)
            for est_idx in range(1, len(estimate_list))[::-1]: # Reverse to not mess with iterator
                if est_under_test.merge_test(estimate_list[est_idx], 0.99):
                    est_under_test = tracking.merge_estimates(est_under_test, estimate_list[est_idx])
                    estimate_list.pop(est_idx)
            new_estimates.append(est_under_test)
        return new_estimates

    def offline_processing(self, measurements_all, timestamps, posterior_method, target_model, termination_steps=np.inf):
        confirmed_estimates = []
        preliminary_estimates = []
        all_estimates = dict()
        termination_dict = dict()
        for measurements, t in zip(measurements_all, timestamps):
            # Step old estimates
            confirmed_estimates = [tracking.Estimate.from_estimate(t, est, target_model, np.zeros(2))for est in confirmed_estimates]
            preliminary_estimates = [tracking.Estimate.from_estimate(t, est, target_model, np.zeros(2)) for est in preliminary_estimates]
            used_measurements = [False for _ in measurements]
            # Gate confirmed targets
            for est in confirmed_estimates:
                is_gated = posterior_method.gate_measurements(measurements, est)
                used_measurements = [new or old for (new, old) in zip(used_measurements, is_gated)]
            measurements = [m for (m, v) in zip(measurements, used_measurements) if not v]
            # Gate preliminary targets
            for est in preliminary_estimates:
                is_gated = posterior_method.gate_measurements(measurements, est)
                used_measurements = [new or old for (new, old) in zip(used_measurements, is_gated)]
            measurements = [m for (m, v) in zip(measurements, used_measurements) if not v]
            # Measurement update using the associated measurements
            [posterior_method.calculate_posterior(estimate) for estimate in confirmed_estimates+preliminary_estimates]
            # Perform track initiation and update pre/conf list
            new_preliminary_estimates, logged_estimates = self.form_new_tracks(measurements)
            preliminary_estimates += new_preliminary_estimates
            new_confirmed_estimates, preliminary_estimates = self.update_track_status(preliminary_estimates)
            confirmed_estimates += new_confirmed_estimates
            for est in confirmed_estimates:
                track_idx = est.track_index
                if track_idx not in all_estimates.keys():
                    all_estimates[track_idx] = []
                all_estimates[track_idx].append((est, 'CONFIRMED'))
            for est in logged_estimates+preliminary_estimates:
                track_idx = est.track_index
                if track_idx not in all_estimates.keys():
                    all_estimates[track_idx] = []
                all_estimates[track_idx].append((est, 'PRELIMINARY'))
            # Termination
            surviving_confirmed_estimates = []
            for est in confirmed_estimates:
                track_idx = est.track_index
                if track_idx not in termination_dict.keys():
                    termination_dict[track_idx] = 0
                if len(est.measurements) == 0:
                    termination_dict[track_idx] += 1
                else:
                    termination_dict[track_idx] = 0
                if termination_dict[track_idx] > termination_steps:
                    pass
                else:
                    surviving_confirmed_estimates.append(est)
            confirmed_estimates = surviving_confirmed_estimates
            # Fusion
            confirmed_estimates = self.fuse_tracks(confirmed_estimates)
            preliminary_estimates = self.fuse_tracks(preliminary_estimates)
        return all_estimates


class IntegratedPDA(object):
    def __init__(self, P_D, P_G, p0, max_vel=20):
        self.p0 = p0
        self.P_D = P_D
        self.P_G = P_G
        self.gamma = chi2(df=2).ppf(self.P_G)
        self.p_c = 1
        self.p_b = 0.1
        self.max_vel = max_vel

    def __repr__(self):
        pass

    def step(self, estimate):
        pass

    def offline_processing(self, measurements_all, timestamps, target_model):
        estimates = []
        probabilities = []
        current_estimate = None
        existence_probability = 0
        for measurements, t in zip(measurements_all, timestamps):
            m_k = len(measurements)
            # First handle no measurements
            if m_k == 0:
                if current_estimate is not None: # Do a step with no measurements
                    current_estimate = tracking.Estimate.from_estimate(t, current_estimate, target_model, np.zeros(2))
                    existence_probability = self.p_c*existence_probability+p_b*(1-existence_probability)
                    current_estimate.est_posterior = current_estimate.est_prior
                    current_estimate.cov_posterior = current_estimate.cov_posterior
                    delta = self.P_D*self.P_G
                    existence_probability = (1-delta)*existence_probability/float(1-delta*existence_probability)
                else: # No measurements, no prior estimate: Do nothing
                    pass
            else: # There are measurements
                if current_estimate is None:
                    current_estimate, existence_probability = self.setup_estimate(measurements, t)
                else: #This is the normal IPDA step
                    is_gated = tracking.gate_measurements(measurements, current_estimate, self.P_G)
                    measurements = [measurement for (measurement, inside) in zip(measurements, is_gated) if inside]
                    current_estimate, existence_probability = self.calculate_posterior(current_estimate, measurements, existence_probability, target_model, t)
            if current_estimate is not None:
                estimates.append(current_estimate)
                probabilities.append(existence_probability)
        return estimates, probabilities

    def calculate_posterior(self, estimate, measurements, existence_probability, target_model, timestamp):
        # Time step
        estimate = tracking.Estimate.from_estimate(timestamp, estimate, target_model, np.zeros(2))
        existence_probability = self.p_c*existence_probability+self.p_b*(1-existence_probability)
        m_k = len(measurements)
        if m_k > 0:
            H = measurements[0].measurement_matrix
            R = measurements[0].covariance
            z_hat = H.dot(estimate.est_prior)
            S = H.dot(estimate.cov_prior).dot(H.T)+R
            V_k = np.pi*np.sqrt(self.gamma*np.linalg.det(S))
            gain = np.dot(estimate.cov_prior, np.dot(H.T, np.linalg.inv(S)))
            delta = self.P_G*self.P_D
            likelihoods = np.zeros(m_k)
            innovations = np.zeros((2, m_k))
            for idx, z in enumerate(measurements):
                innovations[:, idx] = z.value-z_hat
                likelihoods[idx] = multivariate_normal.pdf(innovations[:, idx], np.zeros(2), S)
                delta -= self.P_D*self.P_G*likelihoods[idx]*V_k/float(m_k)
            existence_probability = (1-delta)*existence_probability/(1-delta*existence_probability)
            betas = np.zeros(m_k+1)
            for idx, z in enumerate(measurements):
                betas[idx] = self.P_D*self.P_G*V_k*likelihoods[idx]/(m_k*(1-delta))
            betas[-1] = (1-self.P_D*self.P_G)/(1-delta)
            betas /= np.sum(betas)
            tracking.PDA_update(estimate, measurements, betas)
        else:
            tracking.DR_update(estimate)

        return estimate, existence_probability
            
    def setup_estimate(self, measurements, timestamp):
        n_z = len(measurements)
        z = np.zeros((2, n_z))
        for idx, meas in enumerate(measurements):
            z[:, idx] = meas.value
        sample_mean = np.mean(z, axis=1)
        sample_cov = np.zeros((2,2))
        for idx in range(n_z):
            diff = z[:, idx]-sample_mean
            diff_vec = diff.reshape((2,1))
            sample_cov += diff_vec.dot(diff_vec.T)
        sample_cov = 1./(n_z-1)*sample_cov
        mean = np.array([sample_mean[0], 0, sample_mean[1], 0])
        cov = np.zeros((4,4))
        cov[0, 0] = sample_cov[0, 0]
        cov[0, 2] = sample_cov[0, 1]
        cov[2, 0] = sample_cov[0, 1]
        cov[2, 2] = sample_cov[1, 1]
        cov[1, 1] = self.max_vel**2
        cov[3, 3] = self.max_vel**2
        estimate = tracking.Estimate(timestamp, mean, cov, is_posterior=True, track_index=1)
        existence_probability = self.p0
        return estimate, existence_probability

class SequentialTrackExtraction(object):
    def __init__(self, P0=0.01, P1=0.99):
        # P1: P(Accept H1 | H1)
        # P0: P(Accept H1 | H0)
        self.lower_bound = (1-P1)/(1-P0)
        self.upper_bound = P1/P0

    def __repr__(self):
        pass

    def offline_processing(self, measurements_all, timestamps, target_model, windowsize=None):
        from filtersim.common_math import Node
        if windowsize is None:
            windowsize=len(measurements_all)
        measurements_all = measurements_all[:windowsize]
        # Construct the tree
        Tree = Node('root')
        for measurements in measurements_all:
            new_leafnodes = []
            for leafnode in leafnodes:
                for measurement in measurements:
                    new_node = Node(measurement, parent=leafnode)
                    leafnode.add_child(new_node)
                    new_leafnodes.append(new_node)

    def extract_measurement_set(self, measurements_all, indices):
        selected_measurements = []
        for measurements, index in zip(measurements_all, indices):
            selected_measurements.append(measurements[index])
        return selected_measurements

class Initiator(object):
    def __init__(self, v_max=np.inf):
        self.tentative_tracks = []
        self.v_max = v_max

    def __repr__(self):
        pass

    def step(self, measurements):
        new_estimates = []
        is_taken = [False for _ in measurements]
        for initiator in self.tentative_tracks:
            for z_idx, new_measurement in enumerate(measurements):
                estimates = tracking.Estimate.from_measurement(initiator, measurement)
                current_state = estimates[1].est_posterior
                if np.sqrt(current_state[1]**2+current_state[3]**2) < self.v_max:
                    new_estimates.append(estimates)
                    is_taken[z_idx] = True
        self.tentative_tracks = [z for z, idx in enumerate(measurements) if not is_taken[idx]]
        return new_estimates

def run_track_manager(track_manager, measurements_all, time):
    for measurements, timestamp in zip(measurements_all, time):
        old_estimates, new_tracks = track_manager.step(measurements, timestamp)
