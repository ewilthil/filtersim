import numpy as np
from base_classes import Sensor
from scipy.linalg import block_diag
from estimators import KF, EKF
from autopy.conversion import heading_to_matrix_2D
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

class TrackingFilter:
    def __init__(self, pose=np.zeros(3)):
        self.pose = pose

    def step(self, measurement, k):
        pos_meas, cov_meas = self.convert_measurement(measurement)
        self.R[:,:,k] = cov_meas
        self.filter.R = cov_meas
        if k > 0:
            self.est_prior[:,k], self.cov_prior[:,:,k] = self.filter.step_markov()
        self.est_posterior[:,k], self.cov_posterior[:,:,k] = self.filter.step_filter(pos_meas)

    def convert_measurement(self, measurement):
        r = measurement[0]
        alpha = measurement[1]
        Rot = heading_to_matrix_2D(alpha)
        R_marked = np.zeros((2,2))
        R_marked += self.R_polar
        R_marked[1,1] = R_marked[1,1]*r**2
        R_z = np.dot(Rot, np.dot(R_marked, Rot.T))
        x = r*np.cos(alpha)
        y = r*np.sin(alpha)
        pos_local = np.array([x,y])
        pos_global = self.pose[0:2]+np.dot(heading_to_matrix_2D(self.pose[2]), pos_local)
        return pos_global, R_z
    
    def update_sensor_pose(self, pose):
        self.pose = pose

class DWNA_filter(TrackingFilter):
    def __init__(self, time, sigma_v, R_polar, state_init, cov_init):
        TrackingFilter.__init__(self)
        self.dt = time[1]-time[0]
        self.time = time
        self.N = len(time)
        self.est_prior = np.zeros((4, self.N))
        self.est_prior[:,0] = state_init
        self.est_posterior = np.zeros((4,self.N))
        self.cov_prior = np.zeros((4,4,self.N))
        self.cov_prior[:,:,0] = cov_init
        self.cov_posterior = np.zeros((4,4,self.N))
        Fsub = np.array([[1, self.dt],[0, 1]])
        F = block_diag(Fsub, Fsub)
        G = np.array([[self.dt**2/2., 0],[self.dt, 0],[0,self.dt**2/2.],[0, self.dt]])
        Q = np.dot(G, np.dot(sigma_v, G.T))
        H = np.array([[1, 0, 0, 0],[0, 0, 1, 0]])
        self.R = np.zeros((2,2,self.N))
        self.R_polar = R_polar
        self.filter = KF(F, H, Q, np.zeros((2,2)), state_init, cov_init)
    


def state_elements(state, dt):
    v_N = state[1]
    v_E = state[3]
    w_threshold = 0.1*np.pi/180
    if np.abs(state[4]) > w_threshold:
        w = state[4]
    else:
        w = w_threshold
    wT = w*dt
    swT = np.sin(wT)
    cwT = np.cos(wT)
    return v_N, v_E, w, wT, swT, cwT
    # Go over each column and fill in
def CT_markov(x, dt):
    _, _, w, _, swT, cwT = state_elements(x, dt)
    f = np.zeros((5,5))
    f[0,0] = 1
    f[0,1] = swT/w
    f[1,1] = cwT
    f[2,1] = (1-cwT)/w
    f[3,1] = swT
    f[2,2] = 1
    f[0,3] = -(1-cwT)/w
    f[1,3] = -swT
    f[2,3] = swT/w
    f[3,3] = cwT
    f[4,4] = np.exp(-1/1)
    return np.dot(f, x)

def CT_markov_jacobian(x, dt):
    v_N, v_E, w, wT, swT, cwT = state_elements(x, dt)
    F = np.zeros((5,5))
    F[0,0] = 1
    F[0,1] = swT/w
    F[1,1] = cwT
    F[2,1] = (1-cwT)/w
    F[3,1] = swT
    F[2,2] = 1
    F[0,3] = -(1-cwT)/w
    F[1,3] = -swT
    F[2,3] = swT/w
    F[3,3] = cwT
    F[4,4] = 1
    F[0,4] = v_N*(wT*cwT-swT)/w**2 - v_E*(wT*swT-1+cwT)/w**2
    F[1,4] = -dt*swT*v_N - dt*cwT*v_E
    F[2,4] = v_N*(wT*swT-1+cwT)/w**2 + v_E*(wT*cwT-swT)/w**2
    F[3,4] = dt*cwT*v_N - dt*swT*v_E
    F[4,4] = np.exp(-1/1)
    return F

class CT_filter(TrackingFilter):
    def __init__(self, time, sigma_v, R_polar, state_init, cov_init):
        TrackingFilter.__init__(self)
        self.dt = time[1]-time[0]
        self.time = time
        self.N = len(time)
        self.est_prior = np.zeros((5, self.N))
        self.est_prior[:,0] = state_init
        self.cov_prior = np.zeros((5, 5, self.N))
        self.cov_prior[:,:,0] = cov_init
        self.est_posterior = np.zeros((5, self.N))
        self.cov_posterior = np.zeros((5, 5, self.N))
        self.R_polar = R_polar
        self.R = np.zeros((2, 2, self.N))
        H = np.array([[1, 0, 0, 0, 0],[0, 0, 1, 0, 0]])
        G = np.array([[self.dt**2/2., 0, 0],[self.dt, 0, 0],[0,self.dt**2/2., 0],[0, self.dt, 0],[0, 0, self.dt]])
        Q = np.dot(G, np.dot(sigma_v, G.T))
        self.filter = EKF(lambda x : CT_markov(x,self.dt), lambda x: np.dot(H,x), Q, np.zeros((2,2)), state_init, cov_init, lambda x : CT_markov_jacobian(x,self.dt), lambda x : H)
