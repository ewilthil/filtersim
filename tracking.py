import numpy as np
from base_classes import Sensor
from scipy.linalg import block_diag

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
    # Assumes converted measurements. This means the measurement covariance will be range- and bearing-dependent
    def __init__(self, T, sigma_w, R_polar):
        self.dt = T
        Fsub = np.array([[1, self.dt],[0, 1]])
        self.F = block_diag(Fsub, Fsub)
        G = np.array([[T**2/2., 0],[T, 0],[0,T**2/2.],[0, T]])
        self.Q = np.dot(G, np.dot(sigma_w, G.T))
        self.H = np.array([[1, 0, 0, 0],[0, 0, 1, 0]])
        self.R_polar = R_polar
    
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

    def step(self, measurement):
        pos, cov = self.convert_measurement(measurement)
