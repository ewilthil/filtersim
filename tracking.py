import numpy as np
import ipdb
from base_classes import Sensor
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal
from estimators import KF, EKF
from autopy.conversion import heading_to_matrix_2D
def polar_to_cartesian(z):
    return np.array([z[0]*np.cos(z[1]), z[0]*np.sin(z[1])])
class IMM:
    def __init__(self, P_in, time, sigmas, state_init, cov_init, prob_init, R_polar, model_names, extra_args):
        self.time = time
        self.N = len(time)
        self.markov_probabilites = P_in
        self.r = P_in.shape[0]
        self.nx = 5#np.max([state_init[j].shape[0] for j in range(self.r)])
        self.estimates_prior = np.zeros((self.nx, 1, self.r, self.N))
        self.covariances_prior = np.zeros((self.nx, self.nx, self.r, self.N))
        for j in range(self.r):
            self.estimates_prior[:state_init[j].shape[0],:,j,0] = state_init[j][:,None]
            self.covariances_prior[:state_init[j].shape[0],:state_init[j].shape[0],j,0] = cov_init[j]
        self.estimates_posterior = np.zeros((self.nx, 1, self.r, self.N))
        self.covariances_posterior = np.zeros((self.nx, self.nx, self.r, self.N))
        self.est_posterior = np.zeros((self.nx, self.N))
        self.cov_posterior = np.zeros((self.nx, self.nx, self.N))
        self.probabilites = np.zeros((self.r, self.N))
        self.probabilites[:,0] = prob_init
        self.filter_bank = []
        for j in range(self.r):
            proc_cov = sigmas[j]
            x0 = state_init[j]
            cov0 = cov_init[j]
            current_class = model_dict[model_names[j]]
            current_filt = current_class(time, proc_cov, R_polar, x0, cov0, **extra_args[j])
            self.filter_bank.append(current_filt)

    def mix(self, prev_est, prev_cov, prev_prob):
        mu_ij = np.zeros((self.r, self.r))
        c_j = np.zeros(self.r)
        for j in range(self.r):
            for i in range(self.r):
                mu_ij[i,j] = self.markov_probabilites[i,j]*prev_prob[i]
                c_j[j] += mu_ij[i,j]
            mu_ij[:,j] = mu_ij[:,j]/c_j[j]
        x_out = np.zeros_like(prev_est)
        P_out = np.zeros_like(prev_cov)
        for j in range(self.r):
            for i in range(self.r):
                x_out[:,:,j] += mu_ij[i,j]*prev_est[:,:,i]
            for i in range(self.r):
                diff = np.squeeze(prev_est[:,:,i]-x_out[:,:,j])
                P_out[:,:,j] += mu_ij[i,j]*(prev_cov[:,:,i]+np.dot(diff[:,None], diff[:,None].T))
        return c_j, x_out, P_out

    def update_mode_probabilities(self, c_j, lambda_j):
        mu = np.zeros(self.r)
        for j in range(self.r):
            mu[j] = c_j[j]*lambda_j[j]
        mu = mu/np.sum(mu)
        return mu

    def output_combination(self, est, cov, prob):
        x_out = np.zeros(est.shape[0:2])
        cov_out = np.zeros(cov.shape[0:2])
        for j in range(self.r):
            x_out += prob[j]*est[:,:,j]
        for j in range(self.r):
            diff = np.squeeze(est[:,:,j]-x_out)
            cov_out += prob[j]*(cov[:,:,j]+np.dot(diff[:,None],diff[:,None].T))
        return x_out, cov_out

    def update_filters(self, z, x_mixed, cov_mixed, k):
        x = np.zeros_like(x_mixed)
        cov = np.zeros_like(cov_mixed)
        likelihood = np.zeros(self.r)
        for j in range(self.r):
            x_temp, cov_temp = self.filter_bank[j].step(z,k)
            if x_temp.shape[0] == 5:
                x[:,:,j] = x_temp[:,None]
                cov[:,:,j] = cov_temp
            else:
                x[:4,:,j] = x_temp[:,None]
                x[4,:,j] = self.filter_bank[j].omega
                cov[:4,:4,j] = cov_temp
            likelihood[j] = self.filter_bank[j].evaluate_likelihood(k)
        return x, cov, likelihood

    def step(self, z, k, new_pose, new_cov):
        for j in range(self.r):
            self.filter_bank[j].update_sensor_pose(new_pose, new_cov)
        if k == 0:
            c_j, x_mixed, cov_mixed = self.mix(self.estimates_prior[:,:,:,0], self.covariances_prior[:,:,:,0], self.probabilites[:,0])
        else:
            c_j, x_mixed, cov_mixed = self.mix(self.estimates_posterior[:,:,:,k-1], self.covariances_posterior[:,:,:,k-1], self.probabilites[:,k-1])
        x, cov, likelihood = self.update_filters(z, x_mixed, cov_mixed, k)
        probs = self.update_mode_probabilities(c_j, likelihood)
        x_out, cov_out = self.output_combination(x, cov, probs)
        self.probabilites[:,k] = probs
        self.estimates_posterior[:,:,:,k] = x
        self.covariances_posterior[:,:,:,k] = cov
        self.est_posterior[:,k] = np.squeeze(x_out)
        self.cov_posterior[:,:,k] = cov_out

class TrackingFilter:
    def __init__(self, time, state_init, cov_init, R_polar, pose=np.zeros(3)):
        self.pose = pose
        self.dt = time[1]-time[0]
        self.time = time
        self.N = len(time)
        self.nx = len(state_init)
        self.est_prior = np.zeros((self.nx, self.N))
        self.est_prior[:,0] = state_init
        self.est_posterior = np.zeros((self.nx,self.N))
        self.cov_prior = np.zeros((self.nx,self.nx,self.N))
        self.cov_prior[:,:,0] = cov_init
        self.cov_posterior = np.zeros((self.nx,self.nx,self.N))
        self.R_polar = R_polar

    def step(self, measurement, k):
        pos_meas, cov_meas = self.convert_measurement(measurement)
        self.R[:,:,k] = cov_meas
        self.filter.R = cov_meas
        if k > 0:
            self.est_prior[:,k], self.cov_prior[:,:,k] = self.filter.step_markov()
        self.est_posterior[:,k], self.cov_posterior[:,:,k] = self.filter.step_filter(pos_meas)
        return self.est_posterior[:,k], self.cov_posterior[:,:,k]

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
    
    def biased_conversion(self, z):
        range_measure = z[0]
        ang_measure = z[1]
        cov_range = self.R_polar[0,0]
        cov_ang = self.R_polar[1,1]
        cos_ang = np.cos(ang_measure)
        sin_ang = np.sin(ang_measure)
        mean_factor = 1- np.exp(-cov_ang)+np.exp(-cov_ang/2)
        x = range_measure*cos_ang*mean_factor
        y = range_measure*sin_ang*mean_factor
        R_11 = (range_measure**2)*np.exp(-2*cov_ang)*(cos_ang**2*(np.cosh(2*cov_ang)-np.cosh(cov_ang))+sin_ang**2*(np.sinh(2*cov_ang)-np.sinh(cov_ang)))
        R_11 += cov_range*np.exp(-2*cov_ang)*(cos_ang**2*(2*np.cosh(2*cov_ang)-np.cosh(cov_ang))+sin_ang**2*(2*np.sinh(2*cov_ang)-np.sinh(cov_ang)))
        R_22 = (range_measure**2)*np.exp(-2*cov_ang)*(sin_ang**2*(np.cosh(2*cov_ang)-np.cosh(cov_ang))+cos_ang**2*(np.sinh(2*cov_ang)-np.sinh(cov_ang)))
        R_22 += cov_range*np.exp(-2*cov_ang)*(sin_ang**2*(2*np.cosh(2*cov_ang)-np.cosh(cov_ang))+cos_ang**2*(2*np.sinh(2*cov_ang)-np.sinh(cov_ang)))
        R_12 = sin_ang*cos_ang*np.exp(-4*cov_ang)*(cov_range+(range_measure**2+cov_range)*(1-np.exp(cov_ang)))
        pos_local = np.array([x,y])
        cov_local = np.array([[R_11, R_12],[R_12,R_22]])
        pos_global = self.pose[0:2]+np.dot(heading_to_matrix_2D(self.pose[2]), pos_local)
        cov_global = cov_local
        return pos_global, cov_global


    def update_sensor_pose(self, pose, R_polar):
        self.pose = pose
        self.R_polar = R_polar

    def evaluate_likelihood(self, k):
        diff = self.filter.measurement-self.filter.measurement_prediction
        return multivariate_normal.pdf(diff, mean=np.zeros_like(diff), cov=self.filter.S)

class DWNA_filter(TrackingFilter):
    def __init__(self, time, sigma_v, R_polar, state_init, cov_init):
        TrackingFilter.__init__(self, time, state_init, cov_init, R_polar)
        self.omega = 0
        Fsub = np.array([[1, self.dt],[0, 1]])
        F = block_diag(Fsub, Fsub)
        G = np.array([[self.dt**2/2., 0],[self.dt, 0],[0,self.dt**2/2.],[0, self.dt]])
        Q = np.dot(G, np.dot(sigma_v, G.T))
        H = np.array([[1, 0, 0, 0],[0, 0, 1, 0]])
        self.R = np.zeros((2,2,self.N))
        self.filter = KF(F, H, Q, np.zeros((2,2)), state_init, cov_init)
    


def state_elements(state, dt):
    v_N = state[1]
    v_E = state[3]
    w_threshold = np.deg2rad(0.01)
    if np.abs(state[4]) > w_threshold:
        w = state[4]
    else:
        w = np.sign(state[4])*w_threshold
    wT = w*dt
    swT = np.sin(wT)
    cwT = np.cos(wT)
    return v_N, v_E, w, wT, swT, cwT

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
    f[4,4] = 1
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
    F[4,4] = 1
    return F

class CT_filter(TrackingFilter):
    def __init__(self, time, sigma_v, R_polar, state_init, cov_init):
        TrackingFilter.__init__(self, time, state_init, cov_init, R_polar)
        self.R = np.zeros((2, 2, self.N))
        H = np.array([[1, 0, 0, 0, 0],[0, 0, 1, 0, 0]])
        G = np.array([[self.dt**2/2., 0, 0],[self.dt, 0, 0],[0,self.dt**2/2., 0],[0, self.dt, 0],[0, 0, self.dt]])
        Q = np.dot(G, np.dot(sigma_v, G.T))
        self.filter = EKF(lambda x : CT_markov(x,self.dt), lambda x: np.dot(H,x), Q, np.zeros((2,2)), state_init, cov_init, lambda x : CT_markov_jacobian(x,self.dt), lambda x : H)

class CT_known(TrackingFilter):
    def __init__(self, time, sigma_v, R_polar, state_init, cov_init, omega):
        TrackingFilter.__init__(self, time, state_init, cov_init, R_polar)
        self.omega = omega
        F = self.construct_F(omega)
        G = np.array([[self.dt**2/2., 0],[self.dt, 0],[0,self.dt**2/2.],[0, self.dt]])
        Q = np.dot(G, np.dot(sigma_v, G.T))
        H = np.array([[1, 0, 0, 0],[0, 0, 1, 0]])
        self.R = np.zeros((2,2,self.N))
        self.filter = KF(F, H, Q, np.zeros((2,2)), state_init, cov_init)

    def construct_F(self, w):
        wT = w*self.dt
        swT = np.sin(wT)
        cwT = np.cos(wT)
        return np.array([[1, swT/w, 0, -(1.-cwT)/w],[0, cwT, 0, -swT],[0, (1.-cwT)/w, 1, swT/w],[0, swT, 0, cwT]])
model_dict = {'DWNA' : DWNA_filter, 'CT_unknown' : CT_filter, 'CT_known' : CT_known}
