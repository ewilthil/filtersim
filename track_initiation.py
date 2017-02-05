import numpy as np
import filtersim.tracking as tracking
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

    def offline_processing(self, measurements_all, timestamps, posterior_method, target_model):
        confirmed_estimates = []
        preliminary_estimates = []
        all_estimates = dict()
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
        return all_estimates


class IntegratedPDA(object):
    def __init__(self, P_D, P_G, p0, max_vel=20):
        self.p0 = p0
        self.P_D = P_D
        self.P_G = P_G
        self.gamma = chi2(df=2).ppf(self.P_G)
        self.p_c = 1
        self.p_b = 0
        self.max_vel = max_vel

    def __repr__(self):
        pass

    def step(self, estimate):
        pass

    def offline_processing(self, measurements_all, timestamps, target_model):
        estimates = []
        probabilities = []
        current_estimate = None
        for measurements, t in zip(measurements_all, timestamps):
            m_k = len(measurements)
            # First handle no measurements
            if m_k == 0:
                if current_estimate is not None: # Do a step with no measurements
                    current_estimate = tracking.Estimate.from_estimate(t, current_estimate, target_model, np.zeros(2))
                    self.existence_probability = self.p_c*self.existence_probability+p_b*(1-self.existence_probability)
                    current_estimate.est_posterior = current_estimate.est_prior
                    current_estimate.cov_posterior = current_estimate.cov_posterior
                    delta = self.P_D*self.P_G
                    self.existence_probability = (1-delta)*self.existence_probability/(1-delta*self.existence_probability)
                else: # No measurements, no prior estimate: Do nothing
                    pass
            else: # There are measurements
                H = measurements[0].measurement_matrix
                R = measurements[0].covariance
                if current_estimate is None: #Find some initializing method
                    current_estimate = self.setup_estimate(measurements, t)
                else: #This is the normal IPDA step
                    # First, gate the measurements
                    is_gated = tracking.gate_measurements(measurements, current_estimate, self.P_G)
                    measurements = [measurement for (measurement, inside) in zip(measurements, is_gated) if inside]
                    m_k = len(measurements)
                    current_estimate = tracking.Estimate.from_estimate(t, current_estimate, target_model, np.zeros(2))
                    z_hat = H.dot(current_estimate.est_prior)
                    S = H.dot(current_estimate.cov_prior).dot(H.T)+R
                    V_k = np.pi*np.sqrt(self.gamma*np.linalg.det(S))
                    gain = np.dot(current_estimate.cov_prior, np.dot(H.T, np.linalg.inv(S)))
                    delta = self.P_G*self.P_D
                    likelihoods = np.zeros(m_k)
                    innovations = np.zeros((2, m_k))
                    for idx, z in enumerate(measurements):
                        innovations[:, idx] = z.value-z_hat
                        likelihoods[idx] = multivariate_normal.pdf(innovations[:, idx], np.zeros(2), S)
                        delta -= self.P_D*self.P_G*likelihoods[idx]*V_k/float(m_k)
                    self.existence_probability = (1-delta)*self.existence_probability/(1-delta*self.existence_probability)
                    betas = np.zeros(m_k+1)
                    for idx, z in enumerate(measurements):
                        betas[idx] = self.P_D*self.P_G*V_k*likelihoods[idx]/(m_k*(1-delta))
                    betas[-1] = (1-self.P_D*self.P_G)/(1-delta)
                    betas /= np.sum(betas)
                    tracking.PDA_update(current_estimate, innovations, betas, gain, S)
            if current_estimate is not None:
                estimates.append(current_estimate)
                probabilities.append(self.existence_probability)
        return estimates, probabilities
            
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
        self.existence_probability = self.p0
        return estimate
