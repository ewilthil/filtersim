import numpy as np
import ipdb
from base_classes import Sensor, pitopi
from scipy.linalg import block_diag, expm
from scipy.stats import multivariate_normal
from estimators import KF, EKF
import autopy.conversion as conv

def polar_to_cartesian(z):
    return np.array([z[0]*np.cos(z[1]), z[0]*np.sin(z[1])])

def sksym(qv):
    return np.array([[0,-qv[2],qv[1]],[qv[2],0,-qv[0]],[-qv[1],qv[0],0]])

class IMM:
    def __init__(self, P_in, time, sigmas, state_init, cov_init, prob_init, R_polar, model_names, extra_args):
        self.time = time
        self.N = len(time)
        self.markov_probabilities = P_in
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
        self.probabilities = np.zeros((self.r, self.N))
        self.probabilities[:,0] = prob_init
        self.likelihoods = np.zeros((self.r, self.N)) # Likelihood of 
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
                mu_ij[i,j] = self.markov_probabilities[i,j]*prev_prob[i]
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
        self.likelihoods[:,k] = likelihood
        return x, cov, likelihood

    def step(self, z, k, new_pose, new_cov):
        for j in range(self.r):
            self.filter_bank[j].update_sensor_pose(new_pose, new_cov)
        if k == 0:
            c_j, x_mixed, cov_mixed = self.mix(self.estimates_prior[:,:,:,0], self.covariances_prior[:,:,:,0], self.probabilities[:,0])
        else:
            c_j, x_mixed, cov_mixed = self.mix(self.estimates_posterior[:,:,:,k-1], self.covariances_posterior[:,:,:,k-1], self.probabilities[:,k-1])
        x, cov, likelihood = self.update_filters(z, x_mixed, cov_mixed, k)
        probs = self.update_mode_probabilities(c_j, likelihood)
        x_out, cov_out = self.output_combination(x, cov, probs)
        self.probabilities[:,k] = probs
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
        self.measurement_innovation = np.zeros((2, self.N))
        self.measurement_innovation_covariance = np.zeros((2,2,self.N))
        self.K_gains = np.zeros((2,self.N))

    def step(self, measurement, pose=np.zeros(3), cov = np.zeros((3,3)), k=10):
        self.update_sensor_pose(pose, cov)
        pos_meas, cov_meas = self.convert_measurement(measurement)
        self.R[:,:,k] = cov_meas
        self.filter.R = cov_meas
        if k > 0:
            self.est_prior[:,k], self.cov_prior[:,:,k] = self.filter.step_markov()
        self.est_posterior[:,k], self.cov_posterior[:,:,k], self.K_gains[:,k] = self.filter.step_filter(pos_meas)
        return self.est_posterior[:,k], self.cov_posterior[:,:,k]

    def convert_measurement(self, measurement):
        r = measurement[0]
        theta = measurement[1]
        x_o = self.pose[0]
        y_o = self.pose[1]
        psi = self.pose[2]
        s_tp = np.sin(theta+psi)
        c_tp = np.cos(theta+psi)
        x_t = x_o+r*c_tp
        y_t = y_o+r*s_tp
        mu_t = np.array([x_t, y_t])
        G_x = np.array([[1, 0, -r*s_tp],[0, 1, r*c_tp]])
        G_z = np.array([[c_tp, -r*s_tp],[s_tp, r*c_tp]])
        cov_t = np.dot(G_x, np.dot(self.pose_cov, G_x.T))+np.dot(G_z, np.dot(self.R_polar, G_z.T))
        return mu_t, cov_t
    
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
        pos_global = self.pose[0:2]+np.dot(conv.heading_to_matrix_2D(self.pose[2]), pos_local)
        cov_global = cov_local
        return pos_global, cov_global


    def update_sensor_pose(self, pose, pose_cov):
        self.pose = pose
        self.pose_cov = pose_cov

    def evaluate_likelihood(self, k):
        diff = self.filter.measurement-self.filter.measurement_prediction
        self.measurement_innovation[:,k] = diff
        self.measurement_innovation_covariance[:,:,k] = self.filter.S
        return multivariate_normal.pdf(diff, mean=np.zeros_like(diff), cov=self.filter.S)

    def ct_process_covar(self, x_target):
        _, _, w, wT, swT, cwT = state_elements(x_target, self.dt)
        Q_pos_vel = np.zeros((4,4))
        Q_pos_vel[0, 0] = 2*(wT - swT)/(w**3)
        Q_pos_vel[0, 1] = (1 - cwT)/(w**2)
        Q_pos_vel[0, 3] = (wT - swT)/(w**2)

        Q_pos_vel[1, 0] = Q_pos_vel[0, 1]
        Q_pos_vel[1, 1] = self.dt
        Q_pos_vel[1, 2] = -(wT-swT)/(w**2)

        Q_pos_vel[2, 1] = Q_pos_vel[1, 2]
        Q_pos_vel[2, 2] = Q_pos_vel[0, 0]
        Q_pos_vel[2, 3] = Q_pos_vel[0, 1]

        Q_pos_vel[3, 0] = Q_pos_vel[0, 3]
        Q_pos_vel[3, 2] = Q_pos_vel[2, 3]
        Q_pos_vel[3, 3] = self.dt
        Q_pos_vel *= self.cov_a

        Q_w = self.cov_w*self.dt
        Q = block_diag(Q_pos_vel, Q_w)
        return Q

class DWNA_filter(TrackingFilter):
    def __init__(self, time, Q, R_polar, state_init, cov_init):
        TrackingFilter.__init__(self, time, state_init, cov_init, R_polar)
        self.omega = 0
        Fsub = np.array([[1, self.dt],[0, 1]])
        F = block_diag(Fsub, Fsub)
        H = np.array([[1, 0, 0, 0],[0, 0, 1, 0]])
        self.R = np.zeros((2,2,self.N))
        self.filter = KF(F, H, Q, np.zeros((2,2)), state_init, cov_init)
    


def state_elements(state, dt):
    v_N = state[1]
    v_E = state[3]
    w_threshold = np.deg2rad(0.05)
    if state[4] == 0:
        w = w_threshold
    elif np.abs(state[4]) < w_threshold:
        w = np.sign(state[4])*w_threshold
    else:
        w = state[4]
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
        self.cov_a = sigma_v[0,0]
        self.cov_w = sigma_v[1,1]
        self.filter = EKF(lambda x : CT_markov(x,self.dt), lambda x: np.dot(H,x), lambda x : self.ct_process_covar(x), np.zeros((2,2)), state_init, cov_init, lambda x : CT_markov_jacobian(x,self.dt), lambda x : H)


class CT_known(TrackingFilter):
    def __init__(self, time, sigma_v, R_polar, state_init, cov_init, omega):
        TrackingFilter.__init__(self, time, state_init, cov_init, R_polar)
        self.omega = omega
        F = self.construct_F(omega)
        self.cov_a = sigma_v[0,0]
        self.cov_w = 0
        Q = self.ct_process_covar(np.hstack((state_init, omega)))[:4,:4]
        H = np.array([[1, 0, 0, 0],[0, 0, 1, 0]])
        self.R = np.zeros((2,2,self.N))
        self.filter = KF(F, H, Q, np.zeros((2,2)), state_init, cov_init)

    def construct_F(self, w):
        wT = w*self.dt
        swT = np.sin(wT)
        cwT = np.cos(wT)
        return np.array([[1, swT/w, 0, -(1.-cwT)/w],[0, cwT, 0, -swT],[0, (1.-cwT)/w, 1, swT/w],[0, swT, 0, cwT]])

class DWNA_schmidt():
    def __init__(self, time, Q, R_polar, est_init, cov_init):
        self.time = time
        self.dt =  time[1]-time[0]
        self.N = len(time)
        self.nx = 13
        self.nz = 2
        self.estimate_init()
        self.est_prior[:,0] = np.hstack((est_init, np.zeros(self.nx-4)))
        self.cov_prior[:,:,0] = block_diag(cov_init, np.zeros((self.nx-4, self.nx-4)))
        Fsub = np.array([[1, self.dt],[0, 1]])
        self.F_t = block_diag(Fsub, Fsub)
        F = block_diag(self.F_t, np.identity(self.nx-4))
        f = lambda x : np.dot(F, x)
        self.R_polar = R_polar
        self.track_cov = Q
        self.K_gains = np.zeros((2,self.N))
        self.filter = EKF(f, lambda x : self.measurement_full2(x), lambda x : block_diag(Q, np.zeros((self.nx-4,self.nx-4))), R_polar, self.est_prior[:,0], self.cov_prior[:,:,0], lambda x : F, H=np.hstack((self.measurement_jacobian, self.ownship_measurement_jacobian)))

    def step(self, radar_measurement, ownship_pose, ownship_cov, k, F, Q, full_state):
        self.quat = full_state[:4]
        self.vel = full_state[4:7]
        self.pos = full_state[7:]
        self.filter.est_posterior[4:] = np.zeros(self.nx-4)
        self.filter.cov_posterior[4:,4:] = ownship_cov
        Phi = expm(self.dt*F)
        #Phi = np.identity(9)
        Qd = self.discretize_system(F, Q)
        self.filter.f = lambda x : np.dot(block_diag(self.F_t, Phi), x)
        self.filter.F = lambda x : block_diag(self.F_t, Phi)
        self.filter.Q = lambda x : block_diag(self.track_cov, Qd)
        if k > 0:
            self.est_prior[:,k], self.cov_prior[:,:,k] = self.filter.step_markov()
        self.filter.h = lambda x : self.measurement_full2(x)
        self.filter.H = lambda x : numerical_jacobian(x, self.measurement_full2)
        self.est_posterior[:,k], self.cov_posterior[:,:,k], self.K_gains[:,k] = self.filter.step_filter(radar_measurement)

    def discretize_system(self, F ,Q):
        row1 = np.hstack((-F, Q))
        row2 = np.hstack((np.zeros_like(F), F.T))
        exp_arg = np.vstack((row1, row2))
        Loan2 = expm(exp_arg*self.dt)
        G2 = Loan2[:F.shape[0],F.shape[0]:]
        F3 = Loan2[F.shape[0]:,F.shape[0]:]
        A = np.dot(F3.T,G2)
        Q_out = 0.5*(A+A.T)
        return Q_out

    def estimate_init(self):
        self.est_posterior = np.zeros((self.nx, self.N))
        self.cov_posterior = np.zeros((self.nx, self.nx, self.N))
        self.est_prior = np.zeros((self.nx, self.N))
        self.cov_prior = np.zeros((self.nx, self.nx, self.N))

    def measurement(self, target_state, ownship_state):
        target_pos = np.hstack((target_state[0], target_state[2]))
        R = np.linalg.norm(target_pos-ownship_state[:2])
        theta = np.arctan2(target_pos[1]-ownship_state[1], target_pos[0]-ownship_state[0])-ownship_state[2]
        return np.hstack((R, pitopi(theta)))

    def measurement_full(self, x):
        x_t = x[:4]
        quat = x[4:8]
        euler_angs = conv.quaternion_to_euler_angles(quat)
        vel = x[8:11]
        pos = x[11:]
        dist = np.sqrt((x_t[0]-pos[0])**2+(x_t[2]-pos[1])**2)
        bearing = pitopi(np.arctan2(x_t[2]-pos[1],x_t[0]-pos[0])-euler_angs[2])
        return np.hstack((dist, bearing))

    def measurement_full2(self, x):
        quat = self.quat
        vel = self.vel
        pos = self.pos
        quat_est = conv.quat_mul(np.hstack((x[4:7]/2, 1)), quat)
        euler_angs = conv.quaternion_to_euler_angles(quat_est)
        dist = np.sqrt((x[0]-pos[0]-x[10])**2+(x[2]-pos[1]-x[11])**2)
        bearing = np.arctan2(x[2]-pos[1]-x[11],x[0]-pos[0]-x[10])-euler_angs[2]
        return np.hstack((dist, pitopi(bearing)))


    def measurement_jacobian(self, target_state, ownship_pose):
        own_pos = ownship_pose[7:9]
        target_pos = np.hstack((target_state[0], target_state[2]))
        dist = np.linalg.norm(target_pos-own_pos[:2])
        H = np.zeros((self.nz, 4))
        H[0,0] = (target_pos[0]-own_pos[0])/dist
        H[0,2] = (target_pos[1]-own_pos[1])/dist
        H[1,0] = (own_pos[1]-target_pos[1])/dist**2
        H[1,2] = (target_pos[0]-own_pos[0])/dist**2
        return H

    def ownship_measurement_jacobian(self, target_state, ownship_state):
        H_o = np.zeros((self.nz,self.nx-4))
        target_pos = np.hstack((target_state[0], target_state[2]))
        R = np.linalg.norm(target_pos-ownship_state[:2])
        H_o[0,6] = (ownship_state[0]-target_pos[0])/R
        H_o[0,7] = (ownship_state[1]-target_pos[1])/R
        H_o[1,6] = (target_pos[1]-ownship_state[1])/R**2
        H_o[1,7] = (ownship_state[0]-target_pos[0])/R**2
        H_o[1,2] = -1
        return H_o
    
    def ownship_error_jacobian(self, quat):
        H_dx = np.zeros((10,9))
        H_dx[0:4,0:3] = 0.5*np.vstack((quat[3]*np.identity(3)-sksym(quat[0:3]),-quat[0:3]))
        H_dx[4:,3:] = np.identity(6)
        return H_dx

def numerical_jacobian(x, h, epsilon=10**-7):
    """
    Calculate a Jacobian from h at x numerically using finite difference
    """
    x_dim = x.size
    h0 = h(x)
    h_dim = h0.size
    H = np.zeros((h_dim, x_dim))
    for i in range(x_dim):
        direction = np.zeros(x_dim)
        direction[i] = 1
        pert = epsilon*direction
        h_pert = h(x + pert)
        H[:,i] = (h_pert - h0)/epsilon
    return H

class DWNA_nocomp():
    def __init__(self, time, Q, R_polar, est_init, cov_init):
        self.time = time
        self.dt =  time[1]-time[0]
        self.N = len(time)
        self.nx = 4
        self.nz = 2
        self.estimate_init()
        self.est_prior[:,0] = est_init
        self.cov_prior[:,:,0] = cov_init
        Fsub = np.array([[1, self.dt],[0, 1]])
        self.F_t = block_diag(Fsub, Fsub)
        self.R_polar = R_polar
        self.track_cov = Q
        self.K_gains = np.zeros((2,self.N))
        H = lambda x : numerical_jacobian(x, self.measurement_full2)
        self.filter = EKF(lambda x : np.dot(self.F_t,x), lambda x : self.measurement_full2(x), lambda x : Q, R_polar, self.est_prior[:,0], self.cov_prior[:,:,0], lambda x : self.F_t, H)

    def step(self, measurement, pose=np.zeros(3), cov = np.zeros((3,3)), k=10):
        self.heading = pose[2]
        self.pos = pose[:2]
        if k > 0:
            self.est_prior[:,k], self.cov_prior[:,:,k] = self.filter.step_markov()
        self.est_posterior[:,k], self.cov_posterior[:,:,k], self.K_gains[:,k] = self.filter.step_filter(measurement)

    def estimate_init(self):
        self.est_posterior = np.zeros((self.nx, self.N))
        self.cov_posterior = np.zeros((self.nx, self.nx, self.N))
        self.est_prior = np.zeros((self.nx, self.N))
        self.cov_prior = np.zeros((self.nx, self.nx, self.N))

    def measurement_full2(self, x):
        dist = np.sqrt((x[0]-self.pos[0])**2+(x[2]-self.pos[1])**2)
        bearing = np.arctan2(x[2]-self.pos[1],x[0]-self.pos[0])-self.heading
        return np.hstack((dist, pitopi(bearing)))

model_dict = {'DWNA' : DWNA_filter, 'CT_unknown' : CT_filter, 'CT_known' : CT_known}
