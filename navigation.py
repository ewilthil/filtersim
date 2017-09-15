import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from filtersim.common_math import van_loan_discretization, sksym
import autoseapy.conversion as conv

## Define constants
gravity_n = np.array([0, 0, 9.81])

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
        return pos+pos_noise, quat+quat_noise

default_gnss = GnssCompass()

class StrapdownSystem(object):
    quat = range(4)
    vel = range(4, 7)
    pos = range(7, 10)
    omega = range(10, 13)
    spec_force = range(13, 16)

    def __init__(self, time, x0):
        self.acc_bias = np.zeros(3)
        self.gyr_bias = np.zeros(3)
        self.states = np.zeros((16, len(time)))
        self.states[:, 0] = self.from_ShipModel(x0[0:6], x0[6:12], x0[12:18])
        self.time = time
        self.dt = time[1]-time[0]

    def from_ShipModel(self, eta, nu, tau):
        ang = eta[3:6]
        vel_b = nu[0:3]
        omega_b = nu[3:6] 
        quat = conv.euler_angles_to_quaternion(ang)
        pos = eta[:3]
        C_b_n = conv.quat_to_rot(quat)
        vel_n = C_b_n.dot(nu[:3])
        omega = nu[3:6]
        v_dot = tau[:3]
        acc_nb = v_dot-np.cross(vel_b, omega)
        spec_force = acc_nb-C_b_n.T.dot(gravity_n)
        return np.hstack((quat, vel_n, pos, omega, spec_force))

    def update_bias(self, acc_bias_increment=np.zeros(3), gyr_bias_increment=np.zeros(3)):
        self.acc_bias += acc_bias_increment
        self.gyr_bias += gyr_bias_increment

    def step(self, idx, spec_force, ang_rate):
        spec_force, ang_rate = self.correct_bias(spec_force, ang_rate)
        if idx > 0:
            old_quat = self.states[self.quat, idx-1]
            old_vel = self.states[self.vel, idx-1]
            old_pos = self.states[self.pos, idx-1]

            qv = old_quat[0:3]
            qw = old_quat[3]
            ang_arg = np.linalg.norm(ang_rate)*self.dt/2
            if np.linalg.norm(ang_rate) < (1e-8):
                quat_inc = np.hstack((0, 0, 0, 1))
            else:
                quat_inc = np.hstack((ang_rate/np.linalg.norm(ang_rate)*np.sin(ang_arg), np.cos(ang_arg)))
            new_quat = conv.quat_mul(old_quat, quat_inc)
            new_quat = new_quat/np.linalg.norm(new_quat)
            R = 0.5*(conv.quat_to_rot(old_quat)+conv.quat_to_rot(new_quat))
            new_vel = old_vel+self.dt*(np.dot(R, spec_force)+gravity_n)
            new_pos = old_pos+self.dt/2.*(old_vel+new_vel)

            self.states[self.quat, idx] = new_quat
            self.states[self.vel, idx] = new_vel
            self.states[self.pos, idx] = new_pos

    def update_strapdown(self, idx, delta_ang, delta_vel, delta_pos):
        delta_quat = np.hstack((0.5*delta_ang, 1))
        delta_quat = delta_quat/np.linalg.norm(delta_quat)
        self.states[self.quat, idx] = conv.quat_mul(delta_quat, self.states[self.quat, idx])
        self.states[self.vel, idx] += delta_vel
        self.states[self.pos, idx] += delta_pos

    def get_states(self, idx):
        return self.states[:, idx]
        #return self.states[self.quat, idx], self.states[self.vel, idx], self.states[self.pos, idx], self.states[self.omega, idx], self.states[self.spec_force, idx]

    def correct_bias(self, spec_force, ang_rate):
        spec_force -= self.acc_bias
        ang_rate -= self.gyr_bias
        return spec_force, ang_rate

    def plot_position(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        line = ax.plot(self.states[self.pos[1], :], self.states[self.pos[0], :])
        ax.plot(self.states[self.pos[1], :], self.states[self.pos[0], :], color= line[0].get_color())
        ax.set_aspect('equal')
        return fig, ax

    def plot_velocity(self, axes=None):
        if axes is None:
            fig, axes = plt.subplots(nrows=3)
        else:
            fig = axes[0].get_figure()
        [axes[i].plot(self.time, self.states[self.vel[i], :]) for i in range(3)]
        return fig, axes

class NavigationFilter(object):
    def __init__(self, time, imu_params=None, gnss_params=None):
        self.time = time
        self.states = np.zeros((9, len(time)))
        self.dt = time[1]-time[0]
        I3 = np.identity(3)
        ang_cov = np.deg2rad(1)**2*I3
        vel_cov = 0.5**2*I3
        pos_cov = 2**2*I3
        self.nx = 9
        self.nz = 7
        self.estimates = np.zeros((self.nx, len(time)))
        self.covariances = np.zeros((self.nx, self.nx, len(time)))
        self.est_posterior = np.zeros(self.nx)
        self.cov_posterior = block_diag(ang_cov, vel_cov, pos_cov)
        self.Q = self.Q_matrix(imu_params)
        self.R = self.R_matrix(gnss_params)

    def step(self, idx, z, u):
        quat_est = u[:4]
        pos_est = u[7:10]
        spec_force = u[13:16]
        C = conv.quat_to_rot(quat_est)
        Fc = self.F_matrix(C, spec_force)
        Gc = self.G_matrix(C)
        Q_current = Gc.dot(self.Q).dot(Gc.T)
        Fk, _, Qk = van_loan_discretization(self.dt, Fc, Q=Q_current)
        Hk = self.H_matrix(quat_est)
        if idx == 0:
            cov_prior = self.cov_posterior
        else:
            cov_prior = Fk.dot(self.cov_posterior).dot(Fk.T)+Qk
        S = Hk.dot(cov_prior).dot(Hk.T)+self.R
        K = cov_prior.dot(Hk.T).dot(np.linalg.inv(S))
        z_hat = np.hstack((pos_est, quat_est))
        self.estimates[:,idx] = K.dot(z-z_hat)
        self.cov_posterior = (np.identity(self.nx)-K.dot(Hk)).dot(cov_prior)
        self.cov_posterior = 0.5*(self.cov_posterior+self.cov_posterior.T)
        delta_states = self.estimates[:,idx]
        self.covariances[:,:,idx] = self.cov_posterior
        trans_mat = np.identity(self.nx)
        trans_mat[:3, :3] += sksym(delta_states[:3])
        self.cov_posterior = trans_mat.dot(self.cov_posterior).dot(trans_mat.T)
        return delta_states[:3], delta_states[3:6], delta_states[6:9]

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

    def Q_matrix(self, params):
        if params is None:
            params = default_imu_params
        I3 = np.identity(3)
        Q = np.zeros((6, 6))
        Q[:3, :3] = params['sigma_a']**2*I3
        Q[3:6, 3:6] = params['sigma_w']**2*I3
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

    def R_matrix(self, params):
        if params is None:
            params = default_gnss_params
        R = np.zeros((self.nz, self.nz))
        R[:3, :3] = params['sigma_pos']**2*np.identity(3)
        R[3:, 3:] = params['sigma_quat']**2*np.identity(4)
        return R
