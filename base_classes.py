import numpy as np
from scipy.stats import multivariate_normal
from autopy.conversion import euler_angles_to_matrix
from scipy.linalg import block_diag, expm
from scipy.integrate import odeint
from scipy.stats import chi2

def pitopi(ang):
    return (ang+np.pi)%(2*np.pi)-np.pi

def radar_measurement(x, x0):
    R = np.sqrt((x[0]-x0[0])**2+(x[1]-x0[1])**2)
    alpha = np.arctan2(x[1]-x0[1], x[0]-x0[0])-x0[5]
    alpha = pitopi(alpha)
    return np.array([R, alpha])

class Sensor:
    def __init__(self, h, bias, noise_cov, time_vec):
        self.h = h
        self.R = noise_cov
        self.bias = bias
        self.time = time_vec
        self.data = np.zeros((noise_cov.shape[0], time_vec.shape[0]))
        self.noise = multivariate_normal(cov=self.R)

    def generate_measurement(self, x, k):
        if isinstance(x, tuple):
            measurement = self.h(*x)+self.bias+self.noise.rvs()
        else:
            measurement = self.h(x)+self.bias+self.noise.rvs()
        self.data[:,k] = measurement
nx = 18
class Model:
    def __init__(self, D, T, Q, init_state, time_vector):
        self.D = D
        self.T = T
        self.Q = Q
        self.dt = time_vector[1]-time_vector[0]
        self.time = time_vector
        self.K = time_vector.size
        self.state = np.zeros((nx,self.K))
        self.state_diff = np.zeros((nx,self.K))
        self.ref = np.zeros((nx,self.K))
        self.state[:,0] = init_state
        self.noise = np.zeros((6,self.K))
        self.noise_dist = multivariate_normal(cov=Q)
        self.eta = range(6)
        self.nu = range(6,12)
        self.dist = range(12,18)
        self.eul = range(3,6)
        self.phi = self.eul[0]
        self.theta = self.eul[1]
        self.psi = self.eul[2]
        self.surge = self.nu[0]
    
    def kinematic_ode(self, x):
        s = lambda ang : np.sin(ang)
        c = lambda ang : np.cos(ang)
        t = lambda ang : np.tan(ang)
        phi, theta, psi = x[self.eul]
        R = euler_angles_to_matrix(x[self.eul])
        T = np.array([  [1, s(phi)*t(theta), c(phi)*t(theta)],
                        [0, c(phi), -s(phi)],
                        [0, s(phi)/c(theta), c(phi)/c(theta)]])
        J = block_diag(R,T)
        return np.dot(J, x[self.nu])
    
    def kinetic_ode(self, x,k):
        return np.dot(self.D, x[self.nu]-self.ref[self.nu, k])+x[self.dist]
    
    def disturbance_ode(self, x, k):
        return np.dot(np.linalg.inv(self.T), x[self.dist])+self.noise[:,k]
    
    def ode(self, x, t, k):
        return np.hstack((self.kinematic_ode(x), self.kinetic_ode(x,k), self.disturbance_ode(x, k)))
    
    def step(self, k, ref=None):
        c_w = 0.1
        c_p = 0.5
        c_q = 0.5
        c_r = 0.4
        self.ref[self.psi, k] = ref[1]
        self.ref[self.nu,k] = np.array([ref[0], 0, -c_w*self.state[2, k-1], -c_p*self.state[self.phi, k-1], -c_q*self.state[self.theta, k-1], -c_r*(self.state[self.psi, k-1]-self.ref[self.psi,k])])
        self.noise[:,k] = self.noise_dist.rvs()
        if k == 0:
            pass
        else:
            self.state[:,k] = odeint(self.ode, self.state[:,k-1], np.array([0, self.dt]),args=(k,))[-1,:]
        self.state_diff[:,k] = self.ode(self.state[:,k], 0, k)

    def NED_vel(self, k):
        R = euler_angles_to_matrix(self.state[self.eul,k])
        return np.dot(R, self.state[self.nu[0:3],k])

class ErrorStats:
    def __init__(self, time, N_mc, plot_args):
        self.time = time
        self.dt = time[1]-time[0]
        self.N = len(time)
        self.N_mc = N_mc
        self.NEES = np.zeros((N_mc, self.N))
        self.RMSE_pos = np.zeros((N_mc, self.N))
        self.RMSE_vel = np.zeros((N_mc, self.N))
        self.plot_args = plot_args

    def update_vals(self, true_state, est_state, cov, n, n_mc):
            diff_state = true_state-est_state
            self.NEES[n_mc, n] = np.dot(diff_state, np.dot(np.linalg.inv(cov), diff_state))
            if true_state.shape[0] >= 4:
                self.RMSE_pos[n_mc, n] = diff_state[0]**2+diff_state[2]**2
                self.RMSE_vel[n_mc, n] = diff_state[1]**2+diff_state[3]**2

    def plot_errors(self, NEES_ax, RMSE_pos_ax, RMSE_vel_ax,percentile=0.95):
        lower_lim = (1-percentile)/2
        upper_lim = 1-lower_lim
        UB = chi2(df=self.N_mc).ppf(upper_lim)*np.ones_like(self.time)
        LB = chi2(df=self.N_mc).ppf(lower_lim)*np.ones_like(self.time)
        NEES_ax.plot(self.time, np.sum(self.NEES, axis=0), **self.plot_args)
        NEES_ax.plot(self.time, UB, 'k')
        NEES_ax.plot(self.time, LB, 'k')
        RMSE_pos_ax.plot(self.time, np.sqrt(np.mean(self.RMSE_pos, axis=0)), **self.plot_args)
        RMSE_vel_ax.plot(self.time, np.sqrt(np.mean(self.RMSE_vel, axis=0)), **self.plot_args)


def discretize_system(Fc, Qc, dt):
    row1 = np.hstack((-Fc, Qc))
    row2 = np.hstack((np.zeros_like(Fc), Fc.T))
    exp_arg = np.vstack((row1, row2))
    Loan2 = expm(exp_arg*dt)
    nx = Fc.shape[0]
    G2 = Loan2[:nx,nx:]
    F = Loan2[nx:,nx:].T
    Q = np.dot(F,G2)
    Q = 0.5*(Q+Q.T)
    return F, Q
