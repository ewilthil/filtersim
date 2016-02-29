import numpy as np
from base_classes import Sensor
from scipy.linalg import block_diag
from estimators import KF
class IMM:
    def __init__(self, P_in):
        self.markov_probabilites = P_in
        self.r = P_in.shape[0]
        pass

    def mix(self, prev_est, prev_cov, prev_prob):
        x_out = np.zeros_like(prev_est)
        P_out = np.zeros_like(prev_cov)
        pass

    def update_mode_probabilities(self, c_j, lambda_j):
        pass

    def output_combination(self):
        pass

    def update_filters(self, z):
        pass

    def step(self):
        pass

class DWNA_filter:
    def __init__(self, time, sigma_v, R_polar, state_init, cov_init):
        self.dt = time[1]-time[0]
        self.time = time
        self.N = len(time)
        self.state = np.zeros((4, N))
        Fsub = np.array([[1, self.dt],[0, 1]])
        self.F = block_diag(Fsub, Fsub)
        G = np.array([[T**2/2., 0],[T, 0],[0,T**2/2.],[0, T]])
        Q = np.dot(G, np.dot(sigma_v, G.T))
        H = np.array([[1, 0, 0, 0],[0, 0, 1, 0]])
        self.R_polar = R_polar
        self.filter = KF(F, H, Q, np.zeros((2,2)), state_init, cov_init)
    
    def measurement_noise_covariance(self, measurement):
        r = measurement[0]
        alpha = measurement[1]
        Rot = np.array([[np.cos(alpha), -np.sin(alpha)],[np.sin(alpha), np.cos(alpha)]])
        R_marked = self.R_polar
        R_marked[1,1] = R_marked[1,1]*r**2
        R_z = np.dot(Rot, np.dot(R_marked, Rot.T))
        return R_z

    def convert_measurement(self, measurement):
        R = self.measurement_noise_covariance(measurement)
        x = measurement[0]*np.cos(measurement[1])
        y = measurement[0]*np.sin(measurement[1])
        return np.array([x, y]), R

    def step(self, measurement, k):
        pos_meas, cov_meas = self.convert_measurement(measurement)
        if k > 0:
            est_prior, cov_prior = self.filter.step_markov()
        else:
            # The initial values for the filter is se
            est_prior, cov_prior = self.filter.est_posterior, self.filter.cov_posterior
        self.filter.R = cov_meas
        est_posterior, cov_posterior = self.filter.step_filter(measurement)
        return est_prior, est_posterior, cov_prior, cov_posterior
        
