import numpy as np
import autopy.conversion as conv
from scipy.linalg import expm, block_diag
from scipy.stats import multivariate_normal
from estimators import EKF_navigation
from tf.transformations import euler_matrix, quaternion_from_euler
from base_classes import Sensor


gravity_n = np.array([0, 0, 9.81])
def imu_measurement(state, state_diff):
    C = conv.euler_angles_to_matrix(state[3:6])
    return np.hstack((state_diff[6:9]-np.cross(state[6:9], state[9:12])-np.dot(C.T,gravity_n), state[9:12]))

def gps_measurement(state):
    pos = state[0:3]
    quat = conv.euler_angles_to_quaternion(state[3:6])
    return np.hstack((pos, quat))

def sksym(qv):
    return np.array([[0,-qv[2],qv[1]],[qv[2],0,-qv[0]],[-qv[1],qv[0],0]])

class NavigationSystem:
    def __init__(self, q0, v0, p0, imu_time, gps_time):
        self.acc_cov = 0.7**2
        self.gyr_cov = np.deg2rad(0.4)**2
        bias_init = np.array([0, 0, 0, 0, 0, 0])
        self.K_imu = len(imu_time)
        self.K_gps = len(gps_time)
        self.IMU = Sensor(imu_measurement, bias_init, block_diag(self.acc_cov*np.identity(3), self.gyr_cov*np.identity(3)), imu_time)
        self.GPS = Sensor(gps_measurement, np.zeros(7), block_diag((3**2)*np.identity(3), (np.deg2rad(3e0)**2)*np.identity(4)), gps_time)
        self.strapdown = Strapdown(q0, v0, p0, imu_time)
        cov_init = np.zeros((9,9))
        cov_init[0:3,0:3] = (1*np.pi/180)**2*np.identity(3)
        cov_init[3:6,3:6] = 1**2*np.identity(3)
        cov_init[6:9,6:9] = 1**2*np.identity(3)
        #cov_init[9:15,9:15] = 1e-6*np.identity(6)
        self.EKF = EKF_navigation(0, 0, self.GPS.R, np.zeros(9), cov_init, gps_time)
        self.Q_cont = block_diag(self.gyr_cov*np.identity(3), self.acc_cov*np.identity(3), 1e-8*np.identity(3))
    def step_strapdown(self, state, state_diff, k_imu):
        self.IMU.generate_measurement((state, state_diff),k_imu)
        imu_data = self.IMU.data[:,k_imu]
        self.strapdown.step(imu_data, k_imu)
    
    def step_filter(self, state, k_imu, k_gps):
        self.GPS.generate_measurement(state, k_gps)
        quat, _, pos, omega, spec_force = self.get_strapdown_estimate(k_imu)
        Phi, H, Q, F, Q_temp = self.calculate_jacobians(omega, spec_force, quat)
        z = self.GPS.data[:,k_gps]
        z_est = np.hstack((pos, quat))
        error_state, error_cov = self.EKF.step(z, z_est, Phi, H, Q, k_gps)
        #self.strapdown.update_bias(error_state[9:12], error_state[12:15])
        self.strapdown.correct_estimates(error_state[0:3], error_state[3:6], error_state[6:9], k_imu)
        self.transform_covariance(k_gps)
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


class Strapdown:
    def __init__(self, q0, v0, p0, time_vec):
        self.orient = np.arange(4)
        self.vel = np.arange(3)+self.orient[-1]+1
        self.pos = np.arange(3)+self.vel[-1]+1
        self.rate = np.arange(3)+self.pos[-1]+1
        self.spec_force = np.arange(3)+self.rate[-1]+1
        self.N = len(time_vec)
        self.time = time_vec
        self.data = np.zeros((16, self.N))
        self.dt = time_vec[1]-time_vec[0]
        self.data[self.orient, 0] = q0
        self.data[self.vel, 0] = v0
        self.data[self.pos, 0] = p0
        self.bias_acc = np.zeros(3)
        self.bias_gyr = np.zeros(3)
    
    def step(self, imu_data, k):
        self.correct_biases(imu_data, k)
        if k == 0:
            pass
        else:
            self.update_attitude(k)
            self.update_velocity(k)
            self.update_poisition(k)

    def update_attitude(self, k):
        q_prev = self.data[self.orient, k-1]
        ang_rate = self.data[self.rate, k]
        qv = q_prev[0:3]
        qw = q_prev[3]
        ang_arg = np.linalg.norm(ang_rate)*self.dt/2
        quat_inc = np.hstack((ang_rate/np.linalg.norm(ang_rate)*np.sin(ang_arg), np.cos(ang_arg)))
        T = 0.5*np.vstack((np.array(qw*np.eye(3)+sksym(qv)),-qv))
        #self.data[self.orient,k] = q_prev + self.dt*np.dot(T, ang_rate)
        self.data[self.orient,k] = conv.quat_mul(q_prev, quat_inc)
        #self.data[self.orient,k] = self.data[self.orient,k]/np.linalg.norm(self.data[self.orient,k])

    def update_velocity(self, k):
        q = self.data[self.orient, k]
        prev_q = self.data[self.orient, k-1]
        prev_vel = self.data[self.vel, k-1]
        spec_force = self.data[self.spec_force, k]
        R = 0.5*(conv.quat_to_rot(q)+conv.quat_to_rot(prev_q))
        self.data[self.vel,k] = prev_vel + self.dt*(np.dot(R,spec_force)+gravity_n)

    def update_poisition(self, k):
        self.data[self.pos,k] = self.data[self.pos,k-1] + self.dt*0.5*(self.data[self.vel,k]+self.data[self.vel,k-1])

    def correct_biases(self, imu_data, k):
        self.data[self.spec_force,k] = imu_data[0:3]-self.bias_acc
        self.data[self.rate,k] = imu_data[3:6]-self.bias_gyr
    
    def update_bias(self, delta_bias_acc, delta_bias_gyr):
        self.bias_acc -= delta_bias_acc
        self.bias_gyr -= delta_bias_gyr

    def correct_estimates(self, delta_ang, delta_vel, delta_pos, k):
        delta_quat = np.hstack((delta_ang/2,1))
        delta_quat = delta_quat/np.linalg.norm(delta_quat)
        self.data[self.orient,k] = conv.quat_mul(delta_quat,self.data[self.orient,k])
        self.data[self.vel,k] = self.data[self.vel,k]+delta_vel
        self.data[self.pos,k] = self.data[self.pos,k]+delta_pos
