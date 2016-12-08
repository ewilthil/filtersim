import numpy as np
import autopy.conversion as conv
from scipy.linalg import expm, block_diag
from scipy.stats import multivariate_normal
from estimators import EKF_navigation
from tf.transformations import euler_matrix, quaternion_from_euler
from ipdb import set_trace
from filtersim.shipmodels import van_loan_discretization

gravity_n = np.array([0, 0, 9.81])

def sksym(x):
    return np.array([[0, -x[2], x[1]],[x[2], 0, -x[0]], [-x[1], x[0], 0]])

default_imu_params = {
        'sigma_a' : 1e-2,
        'sigma_w' : np.deg2rad(0.2),
        'b_a_max' : 1e-2,
        'b_w_max' : 1e-2,
        'b_a' : None,
        'b_w' : None,
        }

unbiased_imu_params = {
        'sigma_a' : 1e-2,
        'sigma_w' : np.deg2rad(0.2),
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
        'sigma_quat' : 0.005,
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
        self.quat = self.quat/np.linalg.norm(self.quat)
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
        pos_cov = 2**2*I3
        self.est_posterior = np.zeros(self.nx)
        self.cov_posterior = block_diag(ang_cov, vel_cov, pos_cov)
        self.Q = self.Q_matrix()
        self.R = self.R_matrix()

    def step(self, z, quat_est, spec_force, pos_est):
        C = conv.quat_to_rot(quat_est)
        Fc = self.F_matrix(C, spec_force)
        Gc = self.G_matrix(C)
        Q_current = Gc.dot(self.Q).dot(Gc.T)
        #Phi = np.identity(F.shape[0])+self.dt*F
        Fk, _, Qk = van_loan_discretization(self.dt, Fc, Q=Q_current)
        self.est_prior = np.zeros(self.nx)
        self.cov_prior = Fk.dot(self.cov_posterior).dot(Fk.T)+Qk
        H = self.H_matrix(quat_est)
        self.S = H.dot(self.cov_prior).dot(H.T)+self.R
        self.K = self.cov_prior.dot(H.T).dot(np.linalg.inv(self.S))
        #set_trace()
        z_hat = np.hstack((pos_est, quat_est))
        self.est_posterior = self.est_prior+self.K.dot(z-z_hat)
        self.cov_posterior = (np.identity(self.nx)-self.K.dot(H)).dot(self.cov_prior)
        self.cov_posterior = 0.5*(self.cov_posterior+self.cov_posterior.T)
        return self.est_posterior, self.cov_posterior, z-z_hat

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
        Q = np.zeros((6, 6))
        Q[:3, :3] = default_imu_params['sigma_a']**2*I3
        Q[3:6, 3:6] = default_imu_params['sigma_w']**2*I3
        #Q[6:9, 6:9] = 0*(1e-8)**2*I3
        if self.nx > 9:
            Q[9:12, 9:12] = (1e-6)**2*I3
        if self.nx > 12:
            Q[12:15, 12:15] = (1e-6)**2*I3
        return Q

    def G_matrix(self, C):
        G = np.zeros((9, 6))
        G[:3, 3:6] = C
        G[3:6, :3] = C
        return G
    
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
        self.M_imu = int(np.floor(dt_imu/dt_base))
        self.M_gnss = int(np.floor(dt_gnss/dt_base))
        self.valid_imu = np.zeros_like(base_time, dtype=bool)
        self.valid_gnss = np.zeros_like(base_time, dtype=bool)
        for idx in range(len(base_time)):
            if np.mod(idx, self.M_imu) == 0:
                self.valid_imu[idx] = True
            if np.mod(idx, self.M_gnss) == 0:
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
        self.gnss_time = base_time[self.valid_gnss]
        N_gnss = len(self.gnss_time)
        self.est_errors = np.zeros((self.navfilter.nx, N_gnss))
        self.true_errors = np.zeros((self.navfilter.nx, N_gnss))
        self.cov_errors = np.zeros((self.navfilter.nx, self.navfilter.nx, N_gnss))
        self.innovations = np.zeros((self.navfilter.nz, N_gnss))
        self.pos_gain = np.zeros((3, N_gnss))
        self.inn_cov = np.zeros((3, N_gnss))
        self.cov_priors = np.zeros((9, N_gnss))
        self.cov_posteriors = np.zeros((9, 9, N_gnss))

    def step(self, idx, true_sf, true_gyr, true_pos, true_vel, true_eul):
        if idx > 0:
            if self.valid_imu[idx]:
                self.states[:16, idx] = self.step_imu(true_sf, true_gyr)
            else:
                self.states[:16, idx] = self.states[:16, idx-1]
        if self.valid_gnss[idx]:
            delta_x = self.step_gnss(idx, true_pos, true_vel, true_eul)
        else:
            delta_x = np.zeros(15)

    def step_imu(self, true_sf, true_gyr):
        spec_force, gyr = self.imu.generate_measurement(true_sf, true_gyr)
        att, vel, pos = self.strapdown.step(spec_force, gyr)
        return np.hstack((att, vel, pos, gyr, spec_force))

    def step_gnss(self, idx, true_pos, true_vel, true_angles):
        C = conv.quat_to_rot(self.states[self.quat, idx])
        quat_est = self.states[self.quat, idx]
        pos_est = self.states[self.pos, idx]
        spec_force = self.states[self.acc, idx]
        z = self.gnss.generate_measurement(true_pos, true_angles)
        est, cov, innovation = self.navfilter.step(z, quat_est, spec_force, pos_est)
        err_quat = np.hstack((0.5*est[:3], 1))
        err_quat = err_quat/np.linalg.norm(err_quat)
        err_vel = est[3:6]
        err_pos = est[6:9]
        self.states[self.quat, idx] = conv.quat_mul(err_quat, quat_est)
        self.states[self.vel, idx] += err_vel
        self.states[self.pos, idx] += err_pos
        self.strapdown.update_estimates(self.states[self.quat, idx], self.states[self.vel, idx], self.states[self.pos, idx])
        G = np.identity(9)
        G[:3, :3] += sksym(0.5*est[:3])
        self.navfilter.cov_posterior = G.dot(cov).dot(G.T)
        gnss_idx = int(np.floor(idx/self.M_gnss))
        self.est_errors[:, gnss_idx] = est
        true_err_quat = conv.quat_mul(conv.euler_angles_to_quaternion(true_angles), conv.quat_conj(self.states[self.quat, idx]))
        true_ang_err = 2*true_err_quat[:3]
        true_vel_err = true_vel-self.states[self.vel, idx]
        true_pos_err = true_pos-self.states[self.pos, idx]
        self.true_errors[:, gnss_idx] = np.hstack((true_ang_err, true_vel_err, true_pos_err))
        self.cov_errors[:,:,gnss_idx] = self.navfilter.cov_posterior
        self.innovations[:, gnss_idx] = innovation
        self.pos_gain[:, gnss_idx] = np.diag(self.navfilter.K[6:9, :3])
        self.inn_cov[:,gnss_idx] = np.diag(self.navfilter.S[:3, :3])
        self.cov_priors[:, gnss_idx] = np.diag(self.navfilter.cov_prior)
        self.cov_posteriors[:, :, gnss_idx] = self.navfilter.cov_posterior

    def get_nees(self):
        nees = np.zeros(len(self.gnss_time))
        for idx in range(len(self.gnss_time)):
            innov = self.true_errors[:,idx]-self.est_errors[:,idx]
            cov = self.cov_posteriors[:,:, idx]
            innov = innov[np.newaxis]
            nees[idx] =np.squeeze(innov.dot(np.linalg.inv(cov)).dot(innov.T))
        return nees



    def plot_position(self, ax):
        ax.plot(self.states[self.pos[1], :], self.states[self.pos[0], :])
        ax.plot(self.states[self.pos[1], 0], self.states[self.pos[0], 0], 'ko')

    def plot_angles(self, axes):
        eul = conv.quaternion_to_euler_angles(self.states[self.quat,:].T)
        [axes[i].plot(self.base_time, np.rad2deg(eul[:,i])) for i in range(3)]

    def plot_velocity(self, axes):
        [axes[i].plot(self.base_time, self.states[self.vel[i], :]) for i in range(3)]

    def plot_errors(self, axes=None):
        if axes is None:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(ncols=3, nrows=3)
        ang_axes=axes[0, :]
        vel_axes=axes[1, :]
        pos_axes=axes[2, :]
        N_err = 20
        covs = np.array([np.diag(self.cov_posteriors[:,:,i]) for i in range(self.cov_posteriors.shape[2])]).T
        titles = ['Roll error', 'Pitch error', 'Yaw error']
        for i in range(3):
            ang_axes[i].plot(self.gnss_time, np.rad2deg(self.true_errors[i, :]), label='True')
            ang_axes[i].errorbar(self.gnss_time, np.rad2deg(self.est_errors[i, :]), yerr=3*np.rad2deg(np.sqrt(covs[i, :])), label='Estimate', errorevery=N_err)
            ang_axes[i].set_title(titles[i])
            ang_axes[i].legend()
        titles = ['North vel error', 'East vel error', 'Down vel error']
        for i in range(3):
            vel_axes[i].plot(self.gnss_time, self.true_errors[i+3, :], label='True error')
            vel_axes[i].errorbar(self.gnss_time, self.est_errors[i+3, :], yerr=3*np.sqrt(covs[i+3, :]), label='Estimate', errorevery=N_err)
            vel_axes[i].set_title(titles[i])
            vel_axes[i].legend()
        titles = ['North pos error', 'East pos error', 'Down pos error']
        for i in range(3):
            pos_axes[i].plot(self.gnss_time, self.true_errors[i+6, :], label='True error')
            pos_axes[i].errorbar(self.gnss_time, self.est_errors[i+6, :], yerr=3*np.sqrt(covs[i+6, :]), label='Estimate', errorevery=N_err)
            pos_axes[i].set_title(titles[i])
            pos_axes[i].legend()

    def plot_innovations(self, axes=None):
        if axes is None:
            import matplotlib.pyplot as plt
            pos_fig, pos_axes = plt.subplots(nrows=3)
            ang_fig, ang_axes = plt.subplots(nrows=4)
            gain_fig, gain_ax = plt.subplots(nrows=3)
            inn_fig, inn_ax = plt.subplots(nrows=3)
            cov_fig, cov_ax = plt.subplots(nrows=3)

        covs = np.array([np.diag(self.cov_posteriors[:,:,i]) for i in range(self.cov_posteriors.shape[2])]).T
        for i in range(3):
            pos_axes[i].plot(self.gnss_time, self.innovations[i, :])
        for i in range(4):
            ang_axes[i].plot(self.gnss_time, self.innovations[i+3, :])

        for i in range(3):
            gain_ax[i].plot(self.gnss_time, self.pos_gain[i, :])

        for i in range(3):
            inn_ax[i].plot(self.gnss_time, self.inn_cov[i, :])

        for i  in range(3):
            cov_ax[i].plot(self.gnss_time, self.cov_priors[6+i, :], label='prior')
            cov_ax[i].plot(self.gnss_time, covs[6+i, :], label='posterior')
            cov_ax[i].legend()
