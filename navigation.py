import numpy as np
import autopy.conversion as conv
from scipy.linalg import expm, block_diag
from scipy.stats import multivariate_normal
from estimators import EKF_navigation
from tf.transformations import euler_matrix, quaternion_from_euler
from ipdb import set_trace

gravity_n = np.array([0, 0, 9.81])

def sksym(x):
    return np.array([[0, -x[2], x[1]],[x[2], 0, -x[0]], [-x[1], x[0], 0]])

default_imu_params = {
        'sigma_a' : 1e-2,
        'sigma_w' : 1e-2,
        'b_a_max' : 1e-2,
        'b_w_max' : 1e-2,
        'b_a' : None,
        'b_w' : None,
        }

unbiased_imu_params = {
        'sigma_a' : 1e-2,
        'sigma_w' : 1e-2,
        'b_a_max' : 1e-2,
        'b_w_max' : 1e-2,
        'b_a' : 0,
        'b_w' : 0,
        }
class InertialMeasurementUnit(object):
    def __init__(self, imu_params=default_imu_params):
        self.sigma_a = imu_params['sigma_a']
        self.sigma_w = imu_params['sigma_w']
        if imu_params['b_a'] is None:
            b_max = imu_params['b_a_max']
            self.b_a = np.random.uniform(-b_max, b_max)
        else:
            self.b_a = imu_params['b_a']
        if imu_params['b_w'] is None:
            b_max = imu_params['b_w_max']
            self.b_w = np.random.uniform(-b_max, b_max)
        else:
            self.b_w = imu_params['b_w']


    def generate_measurement(self, acc, gyr):
        acc_noise = np.random.normal(scale=self.sigma_a, size=3)
        gyr_noise = np.random.normal(scale=self.sigma_w, size=3)
        return acc+self.b_a+acc_noise, gyr+self.b_w+gyr_noise

default_imu = InertialMeasurementUnit()
unbiased_imu = InertialMeasurementUnit(unbiased_imu_params)

default_gnss_params = {
        'sigma_pos' : 2,
        'sigma_ang' : np.deg2rad(1),
        'sigma_quat' : 0.05,
        }

class GnssCompass(object):
    def __init__(self, params=default_gnss_params):
        self.sigma_pos = params['sigma_pos']
        self.sigma_quat = params['sigma_quat']

    def generate_measurement(self, pos, eul):
        quat = conv.euler_angles_to_quaternion(eul)
        pos_noise = np.random.normal(scale=self.sigma_pos, size=3)
        quat_noise = np.random.normal(scale=self.sigma_quat, size=4)
        return np.hstack((pos+pos_noise, quat+quat_noise))

default_gnss = GnssCompass()

class StrapdownSystem(object):
    def __init__(self, quat, vel, pos, dt):
        self.quat = quat
        self.vel = vel
        self.pos = pos
        self.dt = dt

    def step(self, acc, gyr):
        quat_old = self.quat
        qv = quat_old[0:3]
        qw = quat_old[3]
        ang_arg = np.linalg.norm(gyr)*self.dt/2
        quat_inc = np.hstack((gyr/np.linalg.norm(gyr)*np.sin(ang_arg), np.cos(ang_arg)))
        self.quat = conv.quat_mul(self.quat, quat_inc)
        R = 0.5*(conv.quat_to_rot(quat_old)+conv.quat_to_rot(self.quat))
        vel_old = self.vel
        self.vel = self.vel+self.dt*(np.dot(R, acc)+gravity_n)
        self.pos = self.pos+self.dt/2.*(vel_old+self.vel)
        return self.quat, self.vel, self.pos

    def update_estimates(self, q, v, p):
        self.quat = q
        self.vel = v
        self.pos = p

class NavigationFilter(object):
    def __init__(self, dt):
        self.dt = dt
        self.nx = 9
        self.nz = 7
        I3 = np.identity(3)
        ang_cov = np.deg2rad(1)**2*I3
        vel_cov = 0.5**2*I3
        pos_cov = 5**2*I3
        self.est_posterior = np.zeros(self.nx)
        self.cov_posterior = block_diag(ang_cov, vel_cov, pos_cov)
        self.Q = self.Q_matrix()
        self.R = self.R_matrix()

    def step(self, z, quat_est, spec_force, pos_est):
        C = conv.quat_to_rot(quat_est)
        F = self.F_matrix(C, spec_force)
        Phi = np.identity(F.shape[0])+self.dt*F
        self.est_prior = np.zeros(self.nx)
        self.cov_prior = Phi.dot(self.cov_posterior).dot(Phi.T)+self.Q
        H = self.H_matrix(quat_est)
        S = H.dot(self.cov_prior).dot(H.T)+self.R
        K = self.cov_prior.dot(H.T).dot(np.linalg.inv(S))
        z_hat = np.hstack((pos_est, quat_est))
        self.est_posterior = self.est_prior+K.dot(z-z_hat)
        self.cov_posterior = (np.identity(self.nx)-K.dot(H)).dot(self.cov_posterior)
        self.cov_posterior = 0.5*(self.cov_posterior+self.cov_posterior.T)
        return self.est_posterior, self.cov_posterior

    def F_matrix(self, C, f):
        F_21 = -sksym(C.dot(f))
        F = np.zeros((self.nx, self.nx))
        F[3:6,:3] = F_21
        F[6:9, 3:6] = np.identity(3)
        return F

    def H_matrix(self, quat_est):
        H_pos = np.identity(3)
        qv = quat_est[:3]
        qs = quat_est[3]
        H_quat_err_r1 = np.hstack((qs*np.identity(3)-sksym(qv), qv[np.newaxis].T))
        H_quat_err_r2 = np.hstack((-qv.T, qs))
        H_quat_err = np.vstack((H_quat_err_r1, H_quat_err_r2))
        H_ang_err = np.vstack((0.5*np.identity(3), np.zeros((1,3))))
        H_quat = H_quat_err.dot(H_ang_err)
        H_total_pos = np.hstack((np.zeros((3,3)), np.zeros((3,3)), H_pos))
        H_total_quat = np.hstack((H_quat, np.zeros((4, 3)), np.zeros((4, 3))))
        H = np.vstack((H_total_pos, H_total_quat))
        return H

    def Q_matrix(self):
        I3 = np.identity(3)
        Q = np.zeros((self.nx, self.nx))
        Q[:3, :3] = default_imu_params['sigma_w']**2*I3
        Q[3:6, 3:6] = default_imu_params['sigma_a']**2*I3
        Q[6:9, 6:9] = (1e-6)**2*I3
        if self.nx > 9:
            Q[9:12, 9:12] = (1e-6)**2*I3
        if self.nx > 12:
            Q[12:15, 12:15] = (1e-6)**2*I3
        return Q
    
    def R_matrix(self):
        R = np.zeros((self.nz, self.nz))
        R[:3, :3] = default_gnss_params['sigma_pos']**2*np.identity(3)
        R[3:, 3:] = default_gnss_params['sigma_quat']**2*np.identity(4)
        return R

class NavigationSystem(object):
    quat = range(4)
    vel = range(4, 7)
    pos = range(7, 10)
    omega = range(10, 13)
    acc = range(13, 16)
    bias_omega = range(16, 19)
    bias_acc = range(19, 22)

    def __init__(self, base_time, dt_imu, dt_gnss, x0, imu=default_imu, gnss=default_gnss):
        self.base_time = base_time
        dt_base = base_time[1]-base_time[0]
        M_imu = int(np.floor(dt_imu/dt_base))
        M_gnss = int(np.floor(dt_gnss/dt_base))
        self.valid_imu = np.zeros_like(base_time, dtype=bool)
        self.valid_gnss = np.zeros_like(base_time, dtype=bool)
        for idx in range(len(base_time)):
            if np.mod(idx, M_imu) == 0:
                self.valid_imu[idx] = True
            if np.mod(idx, M_gnss) == 0:
                self.valid_gnss[idx] = True
        self.states = np.zeros((22, len(base_time)))
        # x0 = [eta, nu, tau]. Biases = 0
        quat_0 = conv.euler_angles_to_quaternion(x0[3:6])
        vel_0 = conv.euler_angles_to_matrix(x0[3:6]).dot(x0[6:9])
        pos_0 = x0[:3]
        self.states[self.quat, 0] = quat_0
        self.states[self.vel, 0] = vel_0
        self.states[self.pos, 0] = pos_0
        self.states[self.omega, 0] = x0[9:12]
        self.states[self.acc, 0] = x0[12:15]
        self.imu = imu
        self.gnss = gnss
        self.strapdown = StrapdownSystem(quat_0, vel_0, pos_0, dt_imu)
        self.navfilter = NavigationFilter(dt_gnss)

    def step(self, idx, true_sf, true_gyr, true_pos, true_eul):
        if idx > 0:
            if self.valid_imu[idx]:
                self.states[:16, idx] = self.step_imu(true_sf, true_gyr)
            else:
                self.states[:16, idx] = self.states[:16, idx-1]
            if self.valid_gnss[idx]:
                delta_x = self.step_gnss(idx, true_pos, true_eul)
            else:
                delta_x = np.zeros(15)

    def step_imu(self, true_sf, true_gyr):
        spec_force, gyr = self.imu.generate_measurement(true_sf, true_gyr)
        att, vel, pos = self.strapdown.step(spec_force, gyr)
        return np.hstack((att, vel, pos, gyr, spec_force))

    def step_gnss(self, idx, true_pos, true_angles):
        C = conv.quat_to_rot(self.states[self.quat, idx])
        quat_est = self.states[self.quat, idx]
        pos_est = self.states[self.pos, idx]
        spec_force = self.states[self.acc, idx]
        z = self.gnss.generate_measurement(true_pos, true_angles)
        est, cov = self.navfilter.step(z, quat_est, spec_force, pos_est)
        err_quat = np.hstack((0.5*est[:3], 1))
        err_vel = est[3:6]
        err_pos = est[6:9]
        self.states[self.quat, idx] = conv.quat_mul(err_quat, quat_est)
        self.states[self.vel, idx] += err_vel
        self.states[self.pos, idx] += err_pos
        self.strapdown.update_estimates(self.states[self.quat, idx], self.states[self.vel, idx], self.states[self.pos, idx])
        G = np.identity(9)
        G[:3, :3] += sksym(0.5*est[:3])
        self.navfilter.cov_posterior = G.dot(cov).dot(G.T)

    def plot_position(self, ax):
        ax.plot(self.states[self.pos[1], :], self.states[self.pos[0], :])
        ax.plot(self.states[self.pos[1], 0], self.states[self.pos[0], 0], 'ko')

    def plot_angles(self, axes):
        eul = conv.quaternion_to_euler_angles(self.states[self.quat,:].T)
        [axes[i].plot(self.base_time, np.rad2deg(eul[:,i])) for i in range(3)]

    def plot_velocity(self, axes):
        [axes[i].plot(self.base_time, self.states[self.vel[i], :]) for i in range(3)]
