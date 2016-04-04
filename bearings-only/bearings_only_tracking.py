import numpy as np
from filtersim.estimators import EKF
import autopy.conversion as conv
from filtersim.base_classes import numerical_jacobian, dwna_transition
import ipdb

class BearingsOnly():
    def __init__(self, time, sigma_a, R, x0, P0, current_state):
        self.time = time
        self.N = len(time)
        self.dt = time[1]-time[0]
        self.cartesian_transition = dwna_transition(self.dt)
        self.cartesian_noise_transition = np.array([[self.dt**2/2, 0],[self.dt, 0],[0, self.dt**2/2],[0, self.dt]])
        self.initialize_filter(x0, P0, sigma_a, R)
        self.prev_state = current_state
        self.local_est_posterior = np.zeros((4,self.N))
        self.local_cov_posterior = np.zeros((4,4,self.N))
        self.global_est_posterior = np.zeros((4,self.N))
        self.global_cov_posterior = np.zeros((4,4,self.N))

    def local_to_global_estimate(self, ownship_pose, k):
        pos = ownship_pose[:2]+self.local_est_posterior[[0,2],k]
        vel = self.local_est_posterior[[1,3],k]
        return np.hstack((pos[0], vel[0], pos[1], vel[1])), np.identity(4)

class BearingsOnlyEKF(BearingsOnly):
    def initialize_filter(self, x0, P0, sigma_a, R):
        Q = sigma_a*np.dot(self.cartesian_noise_transition, self.cartesian_noise_transition.T)
        self.filter = EKF(self.state_transition, self.measurement_model, Q, R, x0, P0)
        self.local_est_posterior = np.zeros((4,self.N))
        self.local_cov_posterior = np.zeros((4,4,self.N))
        self.global_est_posterior = np.zeros((4,self.N))
        self.global_cov_posterior = np.zeros((4,4,self.N))

    def step(self, measurement, current_state, current_pose, k):
        u = np.dot(self.cartesian_transition, self.prev_state)-current_state
        self.prev_state = current_state
        if k > 0:
            self.filter.step_markov(u)
        self.local_est_posterior[:,k], self.local_cov_posterior[:,:,k], _ = self.filter.step_filter(measurement, [0])
        self.global_est_posterior[:,k], self.global_cov_posterior[:,:,k] = self.local_to_global_estimate(current_pose, k)
        
    def state_transition(self, x, u, v):
        x_next = np.dot(self.cartesian_transition, x)+u+v
        return x_next

    def measurement_model(self, x, w):
        z = np.arctan2(x[2], x[0])+w
        return z

class BearingsOnlyMP(BearingsOnly):
    def initialize_filter(self, x0, P0, sigma_a, R):
        y0 = self.cartesian_to_polar(x0)
        G0 = numerical_jacobian(x0, self.cartesian_to_polar)
        P0_y = np.dot(G0, np.dot(P0, G0.T))
        Q = sigma_a*np.dot(self.cartesian_noise_transition, self.cartesian_noise_transition.T)
        self.filter = EKF(self.state_transition, self.measurement_model, Q, R, y0, P0_y)
        self.polar_est = np.zeros((4,self.N))

    def step(self, measurement, current_state, current_pose, k):
        u = np.dot(self.cartesian_transition, self.prev_state)-current_state
        self.prev_state = current_state
        if k > 0:
            self.filter.step_markov(u)
        y, cov_y, _ = self.filter.step_filter(measurement, [0])
        self.polar_est[:,k] = y
        G_y = numerical_jacobian(y, self.polar_to_cartesian)
        self.local_est_posterior[:,k] = self.polar_to_cartesian(y)
        self.local_cov_posterior[:,:,k] = np.dot(G_y, np.dot(cov_y, G_y.T))
        self.global_est_posterior[:,k], self.global_cov_posterior[:,:,k] = self.local_to_global_estimate(current_pose, k)

    def state_transition(self, y, u, v):
        # The support functions should be replaced with the true stuff
        x = self.polar_to_cartesian(y)
        x_next = np.dot(self.cartesian_transition, x)+u+v
        return self.cartesian_to_polar(x_next)

    def measurement_model(self, y, w):
        return y[0]+w

    def cartesian_to_polar(self, x):
        N = x[0]
        E = x[2]
        VN = x[1]
        VE = x[3]
        R = np.sqrt(N**2+E**2)
        y = np.array([np.arctan2(E, N), (N*VE-E*VN)/R**2, 1./R, (N*VN+E*VE)/(1.*R**2)])
        return y

    def polar_to_cartesian(self, y):
        beta = y[0]
        beta_dot = y[1]
        R = 1/y[2]
        R_dot = y[3]*R
        sin_beta = np.sin(beta)
        cos_beta = np.cos(beta)
        x = np.array([R*cos_beta, R_dot*cos_beta-R*sin_beta*beta_dot, R*sin_beta, R_dot*sin_beta+R*cos_beta*beta_dot])
        return x
