import numpy as np
import autopy.conversion as conv
from scipy.linalg import expm, block_diag
from scipy.stats import multivariate_normal
from estimators import EKF_navigation
from tf.transformations import euler_matrix, quaternion_from_euler
from ipdb import set_trace

gravity_n = np.array([0, 0, 9.81])

default_imu_params = {
        'sigma_a' : 1e-6,
        'sigma_w' : 1e-6,
        'b_a_max' : 1e-2,
        'b_w_max' : 1e-2,
        'b_a' : None,
        'b_w' : None,
        }

unbiased_imu_params = {
        'sigma_a' : 1e-6,
        'sigma_w' : 1e-6,
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
        acc_noise = np.random.normal(scale=self.sigma_a)
        gyr_noise = np.random.normal(scale=self.sigma_w)
        return acc+self.b_a+acc_noise, gyr+self.b_w+gyr_noise

default_imu = InertialMeasurementUnit()
unbiased_imu = InertialMeasurementUnit(unbiased_imu_params)

class GnssCompass(object):
    def __init__(self):
        pass

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
        #R = 0.5*(conv.quat_to_rot(quat_old)+conv.quat_to_rot(self.quat))
        R = conv.quat_to_rot(self.quat)
        vel_old = self.vel
        self.vel = self.vel+self.dt*(np.dot(R, acc)+gravity_n)
        self.pos = self.pos+self.dt/2.*(vel_old+self.vel)
        return self.quat, self.vel, self.pos

class NavigationFilter(object):
    def __init__(self):
        pass

class NavigationSystem(object):
    quat = range(4)
    vel = range(4, 7)
    pos = range(7, 10)
    omega = range(10, 13)
    acc = range(13, 16)
    bias_omega = range(16, 19)
    bias_acc = range(19, 22)

    def __init__(self, base_time, dt_imu, dt_gnss, x0, imu=default_imu):
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
        self.strapdown = StrapdownSystem(quat_0, vel_0, pos_0, dt_imu)
        self.imu = imu

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
        acc, gyr = self.imu.generate_measurement(true_sf, true_gyr)
        att, vel, pos = self.strapdown.step(acc, gyr)
        return np.hstack((att, vel, pos, gyr, acc))

    def step_gnss(self, idx, pos, att):
        pass




    def plot_position(self, ax):
        ax.plot(self.states[self.pos[1], :], self.states[self.pos[0], :])
        ax.plot(self.states[self.pos[1], 0], self.states[self.pos[0], 0], 'ko')

    def plot_angles(self, axes):
        eul = conv.quaternion_to_euler_angles(self.states[self.quat,:].T)
        [axes[i].plot(self.base_time, np.rad2deg(eul[:,i])) for i in range(3)]

    def plot_velocity(self, axes):
        [axes[i].plot(self.base_time, self.states[self.vel[i], :]) for i in range(3)]





def imu_measurement(state, state_diff):
    C = conv.euler_angles_to_matrix(state[3:6])
    return np.hstack((state_diff[6:9]-np.cross(state[6:9], state[9:12])-np.dot(C.T,gravity_n), state[9:12]))

def gnss_measurement(state):
    pos = state[0:3]
    quat = conv.euler_angles_to_quaternion(state[3:6])
    return np.hstack((pos, quat))

def sksym(qv):
    return np.array([[0,-qv[2],qv[1]],[qv[2],0,-qv[0]],[-qv[1],qv[0],0]])

class StrapdownDataDeprectated:
    def __init__(self):
        self.orient = np.zeros(4)
        self.vel = np.zeros(3)
        self.pos = np.zeros(3)
        self.spec_force = np.zeros(3)
        self.ang_rate = np.zeros(3)
        self.acc_bias = np.zeros(3)
        self.gyr_bias = np.zeros(3)

class StrapdownDeprecated:
    def __init__(self, q0, v0, p0, dt):
        self.data = StrapdownData()
        self.data.orient = q0
        self.data.vel = v0
        self.data.pos = p0
        self.dt = dt

    def step(self, spec_force, ang_rate):
        self.data.spec_force, self.data.ang_rate = self.correct_biases(spec_force, ang_rate)
        self.update_attitude(self.data.ang_rate)
        self.update_velocity(self.data.spec_force)
        self.update_poisition()
        return self.data.orient, self.data.vel, self.data.pos

    def update_attitude(self, ang_rate):
        quat = self.data.orient
        qv = quat[0:3]
        qw = quat[3]
        ang_arg = np.linalg.norm(ang_rate)*self.dt/2
        quat_inc = np.hstack((ang_rate/np.linalg.norm(ang_rate)*np.sin(ang_arg), np.cos(ang_arg)))
        self.prev_orient = quat
        self.data.orient = conv.quat_mul(self.data.orient, quat_inc)

    def update_velocity(self, spec_force):
        quat = self.data.orient
        vel = self.data.vel
        R = 0.5*(conv.quat_to_rot(self.prev_orient)+conv.quat_to_rot(quat))
        self.prev_vel = vel
        self.data.vel = vel+self.dt*(np.dot(R,spec_force)+gravity_n)

    def update_poisition(self):
        self.data.pos = self.data.pos+self.dt*(self.prev_vel+self.data.vel)/2.

    def correct_biases(self, spec_force, ang_rate):
        return spec_force-self.data.acc_bias, ang_rate-self.data.gyr_bias

    def update_bias(self, delta_bias_acc, delta_bias_gyr):
        self.data.acc_bias -= delta_bias_acc
        self.data.gyr_bias -= delta_bias_gyr

    def correct_estimates(self, delta_ang, delta_vel, delta_pos):
        delta_quat = np.hstack((delta_ang/2,1))
        delta_quat = delta_quat/np.linalg.norm(delta_quat)
        self.data.orient = conv.quat_mul(delta_quat,self.data.orient)
        self.data.vel += delta_vel
        self.data.pos += delta_pos
        return self.data.orient, self.data.vel, self.data.pos

class NavigationFilterDeprecated():
    def __init__(self):
        pass

    def measurement_equation(self, state):
        pos = self.position_estimate+state
        return np.hstack((position, quaternion))
    pass

def step(self, measurement):
    pass












class NavigationSystemDeprecated:
    def __init__(self, q0, v0, p0, imu_time, gnss_time):
        self.acc_cov = 0.7**2
        self.gyr_cov = np.deg2rad(0.4)**2
        bias_init = np.array([0, 0, 0, 0, 0, 0])
        self.K_imu = len(imu_time)
        self.K_gnss = len(gnss_time)
        self.IMU = Sensor(imu_measurement, bias_init, block_diag(self.acc_cov*np.identity(3), self.gyr_cov*np.identity(3)), imu_time)
        self.GPS = Sensor(gnss_measurement, np.zeros(7), block_diag((3**2)*np.identity(3), (np.deg2rad(3e0)**2)*np.identity(4)), gnss_time)
        self.strapdown = Strapdown(q0, v0, p0, imu_time)
        cov_init = np.zeros((9,9))
        cov_init[0:3,0:3] = (1*np.pi/180)**2*np.identity(3)
        cov_init[3:6,3:6] = 1**2*np.identity(3)
        cov_init[6:9,6:9] = 1**2*np.identity(3)
        #cov_init[9:15,9:15] = 1e-6*np.identity(6)
        self.EKF = EKF_navigation(0, 0, self.GPS.R, np.zeros(9), cov_init, gnss_time)
        self.Q_cont = block_diag(self.gyr_cov*np.identity(3), self.acc_cov*np.identity(3), 1e-8*np.identity(3))
    def step_strapdown(self, state, state_diff, k_imu):
        self.IMU.generate_measurement((state, state_diff),k_imu)
        imu_data = self.IMU.data[:,k_imu]
        self.strapdown.step(imu_data, k_imu)

    def step_filter(self, state, k_imu, k_gnss):
        self.GPS.generate_measurement(state, k_gnss)
        quat, _, pos, omega, spec_force = self.get_strapdown_estimate(k_imu)
        Phi, H, Q, F, Q_temp = self.calculate_jacobians(omega, spec_force, quat)
        z = self.GPS.data[:,k_gnss]
        z_est = np.hstack((pos, quat))
        error_state, error_cov = self.EKF.step(z, z_est, Phi, H, Q, k_gnss)
        #self.strapdown.update_bias(error_state[9:12], error_state[12:15])
        self.strapdown.correct_estimates(error_state[0:3], error_state[3:6], error_state[6:9], k_imu)
        self.transform_covariance(k_gnss)
        return F, Q_temp


    def calculate_jacobians(self, omega_est, spec_force_est, quat_est):
        C = conv.quat_to_rot(quat_est)
        F = np.zeros((9,9))
        #F[0:3,12:15] = C
        F[3:6,0:3] = -sksym(np.dot(C, spec_force_est))
        #F[3:6,9:12] = C
        F[6:9,3:6] = np.identity(3)
        Phi = expm(self.EKF.dt*F)
        H = np.zeros((7,9))
        H[0:3,6:9] = np.identity(3)
        H[3:7,0:3] = 0.5*np.vstack((quat_est[3]*np.identity(3)+sksym(quat_est[0:3]),-quat_est[0:3]))

        B = block_diag(C, C, 0*np.identity(3))
        Q, Q_temp = self.discretize_system(F, B, self.Q_cont)
        return Phi, H, Q, F, Q_temp

    def get_strapdown_estimate(self, k_imu):
        state = self.strapdown.data[:,k_imu]
        return state[self.strapdown.orient], state[self.strapdown.vel], state[self.strapdown.pos], state[self.strapdown.rate], state[self.strapdown.spec_force]

    def transform_covariance(self, k):
        err_ang = self.EKF.est_posterior[0:3,k]
        err_cov = self.EKF.cov_posterior[:,:,k]
        G_ang = np.identity(3)+sksym(0.5*err_ang)
        G = block_diag(G_ang, np.identity(6))
        self.EKF.cov_posterior[:,:,k] = np.dot(G,np.dot(err_cov,G.T))

    def discretize_system(self, F, B, Q):
        Qc = np.dot(B, np.dot(Q, B.T))
        row1 = np.hstack((-F, Qc))
        row2 = np.hstack((np.zeros_like(F), F.T))
        exp_arg = np.vstack((row1, row2))
        Loan2 = expm(exp_arg*self.EKF.dt)
        G2 = Loan2[:F.shape[0],F.shape[0]:]
        F3 = Loan2[F.shape[0]:,F.shape[0]:]
        A = np.dot(F3.T,G2)
        Q_out = 0.5*(A+A.T)
        return Q_out, Qc


