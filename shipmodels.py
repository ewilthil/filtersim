import numpy as np
from scipy.linalg import block_diag, expm
from scipy.stats import multivariate_normal
from autoseapy.conversion import euler_angles_to_matrix
from filtersim.navigation import gravity_n
import matplotlib.pyplot as plt
from ipdb import set_trace
from filtersim.common_math import van_loan_discretization, sksym

def plot_with_title(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)

class LinearStochasticModel(object):
    def __init__(self, dt, n_dim, parameters=dict()):
        parameters = self.set_parameters(n_dim, parameters)
        self.n_dim = n_dim
        self.A = [None for _ in range(n_dim)]
        self.B = [None for _ in range(n_dim)]
        self.Q = [None for _ in range(n_dim)]
        for i in range(n_dim):
            self.A[i], self.B[i], self.Q[i] = self.matrices_1D(parameters, i)
        self.Ad, self.Bd, self.Qd = self.discretize_system(dt)

    def step(self, x, u, v):
        return self.Ad.dot(x)+self.Bd.dot(u)+v

    def set_parameters(self, n, params):
        new_params = self.default_parameters(n)
        for param_name in new_params.keys():
            if param_name in params.keys():
                new_params[param_name] = params[param_name]
        return new_params

    def discretize_system(self, dt, current_state):
        Ad = [None for _ in range(self.n_dim)]
        Bd = [None for _ in range(self.n_dim)]
        Qd = [None for _ in range(self.n_dim)]
        for i in range(self.n_dim):
            Ad[i], Bd[i], Qd[i] = van_loan_discretization(dt, self.A[i], self.B[i], self.Q[i])
        Ad = block_diag(*Ad)
        Bd = block_diag(*Bd)
        Qd = block_diag(*Qd)
        return Ad, Bd, Qd

class BestNorton(LinearStochasticModel):
    def __init__(self, dt, sigmas):
        pass

    def discretize_system(self, dt, current_state):
        F = np.array([[1, dt, 0, 0],[0, 1, 0, 0],[0, 0, 1, dt],[0, 0, 0, 1]])
        B = np.zeros(2)
        phi = np.arctan(current_state[1]/current_state[3])
        omega = 
        G = np.array([
            [],
            [],
            ])

class IntegratedOU(LinearStochasticModel):
    def default_parameters(self, n):
        params = {'sigmas' : 0.3*np.ones(n),
                'thetas' : 0.5*np.ones(n),
                }
        return params

    def matrices_1D(self, parameters, index):
        theta = parameters['thetas'][index]
        sigma = parameters['sigmas'][index]
        A = np.array([[0, 1], [0, -theta]])
        B = np.array([[0], [theta]])
        G = np.array([[0], [1]])
        Q = sigma*G.dot(G.T)
        return A, B, Q

class DiscreteWNA(LinearStochasticModel):
    def default_parameters(self, n):
        params = {'sigmas' : 0.3*np.ones(n)}
        return params

    def matrices_1D(self, parameters, index):
        sigma = parameters['sigmas'][index]
        A = np.array([[0, 1], [0, 0]])
        B = np.array([[0], [0]])
        G = np.array([[0], [1]])
        Q = sigma*G.dot(G.T)
        return A, B, Q

class IntegratedMOU(LinearStochasticModel):
    def default_parameters(self, n):
        params = {'thetas' : 0.5*np.zeros(n),
                'sigmas' : 0.1*np.zeros(n),
                }
        return params

    def matrices_1D(self, parameters, index):
        theta = parameters['thetas'][index]
        sigma = parameters['sigmas'][index]
        A = np.array([[0, 1, 0], [0, 0, 1], [-theta, 0, 0]])
        B = np.array([[0], [0], [theta]])
        G = np.array([[0], [0], [1]])
        Q = sigma*G.dot(G.T)
        return A, B, Q

class TargetShip(object):
    def __init__(self, time, model, x0):
        self.time = time
        self.dt = time[1]-time[0]
        self.states = np.zeros((len(x0), len(time)))
        self.states[:,0] = x0
        self.model = model
        self.noise = multivariate_normal(np.zeros_like(x0), model.Qd).rvs(size=len(time)).T

    def step(self, idx, v_ref):
        if idx > 0:
            x_now = self.states[:, idx-1]
            self.states[:,idx] = self.model.step(x_now, v_ref, self.noise[:,idx])
    
    def cartesian_position_measurement(self, idx):
        return np.array([self.states[0, idx], self.states[2, idx]])

    def plot_position(self, ax):
        ax.plot(self.states[2,:], self.states[0,:])
        ax.plot(self.states[2,0], self.states[0,0], 'ko')
        ax.set_aspect('equal')
        ax.set_title('Position')
        ax.set_xlabel('North')
        ax.set_ylabel('East')

    def plot_velocity(self, axes):
        axes[0].plot(self.time, self.states[1,:])
        axes[0].set_title('Velocity (north)')
        axes[1].plot(self.time, self.states[3,:])
        axes[1].set_title('Velocity (east)')

    def generate_trajectory(self, v_ref):
        for t_idx, timestamp in enumerate(self.time):
            self.step(t_idx, v_ref)

class NonlinearStochasticModel(object):
    def __init__(self):
        self.theta_eta = np.diag((0, 0, 2, 5, 5, 15))
        self.theta_nu = np.diag((2, 2, 10, 10, 10, 10))
        self.theta_tau = np.diag((0.5, 0.5, 1, 1, 1, 15))
        self.Qd = np.diag((1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-4))
        self.G = np.vstack((np.zeros((12,6)), np.identity(6)))
        self.B = np.vstack((np.zeros((12,12)), np.hstack((self.theta_eta, self.theta_nu))))

    def step(self, x, u, v, dt):
        return x+self.f(x)*dt+self.B.dot(u)*dt+self.G.dot(v)

    def f(self, x):
        f_tilde = np.zeros((18,18))
        f_tilde[:6,6:12] = self.J(x[:6])
        A = np.zeros((18,18))
        A[6:12,12:] = np.identity(6)
        A[12:,:6] = -self.theta_eta
        A[12:,6:12] = -self.theta_nu
        A[12:, 12:] = -self.theta_tau
        return (A+f_tilde).dot(x)
    
    def J(self, eta):
        ang = eta[3:]
        R = euler_angles_to_matrix(ang)
        sphi = np.sin(ang[0])
        cphi = np.cos(ang[0])
        ctheta = np.cos(ang[1])
        ttheta = np.tan(ang[1])
        T = np.array([[1, sphi*ttheta, cphi*ttheta], [0, cphi, -sphi], [0, sphi/ctheta, cphi/ctheta]])
        return block_diag(R, T)

        
class Ownship(object):
    eta = range(6)
    nu = range(6, 12)
    tau = range(12, 18)
    pos = range(3)
    ang = range(3, 6)
    def __init__(self, time, model, x0, imu=None, gnss=None):
        self.time = time
        self.dt = time[1]-time[0]
        self.states = np.zeros((len(x0), len(time)))
        self.states[:,0] = x0
        self.model = model
        self.noise = multivariate_normal(np.zeros(model.Qd.shape[0]), model.Qd).rvs(size=len(time)).T
        self.imu = imu
        self.gnss = gnss

    def step(self, idx, u_ref, psi_ref):
        if idx > 0:
            nu_ref = np.array([u_ref, 0, 0, 0, 0, 0])
            eta_ref = np.array([0, 0, 0, 0, 0, psi_ref])
            ref = np.hstack((eta_ref, nu_ref))
            x_now = self.states[:, idx-1]
            self.states[:,idx] = self.model.step(x_now, ref, self.noise[:,idx], self.dt)

    # Navigation

    def imu_states(self, idx):
        ang_rate = self.states[self.nu, idx][self.ang]
        acc = self.states[self.tau, idx][self.pos]+sksym(ang_rate).dot(self.states[self.nu, idx][self.pos])
        euler = self.states[self.eta, idx][self.ang]
        DCM_to_b = euler_angles_to_matrix(euler).T
        spec_force = acc-DCM_to_b.dot(gravity_n)
        if self.imu is not None:
            spec_force, ang_rate = self.imu.generate_measurement(spec_force, ang_rate)
        return spec_force, ang_rate

    def gnss_states(self, idx):
        pos = self.states[self.eta, idx][self.pos]
        eul = self.states[self.eta, idx][self.ang]
        if self.gnss is not None:
            pos, eul = self.gnss.generate_measurement(pos, eul)
        return np.hstack((pos, eul))
    
    # Utilities
    def vel_to_NED(self):
        vel_B = self.states[6:9,:]
        ang = self.states[3:6, :]
        vel_N = np.zeros_like(vel_B)
        for idx in range(len(self.time)):
            vel_N[:,idx] = euler_angles_to_matrix(ang[:,idx]).dot(vel_B[:,idx])
        return vel_N


    # Plot functions
    def plot_position(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        line = ax.plot(self.states[1,:], self.states[0,:])
        ax.plot(self.states[1,0], self.states[0,0], color=line[0].get_color())
        ax.set_aspect('equal')
        return fig, ax

    def plot_velocity(self, axes=None):
        if axes is None:
            fig, axes = plt.subplots(nrows=3)
        else:
            fig = axes[0].get_figure()
        vel_N = self.vel_to_NED()
        axes[0].plot(self.time, vel_N[0,:])
        axes[1].plot(self.time, vel_N[1,:])
        axes[2].plot(self.time, vel_N[2,:])
        return fig, axes

    def plot_velocity_body(self, axes=None):
        if axes is None:
            fig, axes = plt.subplots(ncols=2)
        plot_with_title(axes[0], self.time, self.states[6,:], 'Forward velocity [m/s]')
        plot_with_title(axes[1], self.time, self.states[7,:], 'Sidways velocity [m/s]')

    def plot_roll_pitch_heave(self, axes=None):
        vel_ned = self.vel_to_NED()
        if axes is None:
            fig, axes = plt.subplots(nrows=3, ncols=2)
        plot_with_title(axes[0,0], self.time, self.states[2,:], 'Heave position [m]')
        plot_with_title(axes[0,1], self.time, vel_ned[2,:], 'Heave velocity [m/s]')
        plot_with_title(axes[1,0], self.time, np.rad2deg(self.states[3,:]), 'Roll angle [deg]')
        plot_with_title(axes[1,1], self.time, np.rad2deg(self.states[9,:]), 'Roll rate [deg/s]')
        plot_with_title(axes[2,0], self.time, np.rad2deg(self.states[4,:]), 'Pitch angle [deg]')
        plot_with_title(axes[2,1], self.time, np.rad2deg(self.states[10,:]), 'Pitch rate [deg/s]')

    def plot_yaw(self, axes=None):
        if axes is None:
            fig, axes = plt.subplots(ncols=2)
        plot_with_title(axes[0], self.time, np.rad2deg(self.states[5,:]), 'Yaw [deg]')
        plot_with_title(axes[1], self.time, np.rad2deg(self.states[11,:]), 'Yaw rate [deg/s]')

    def plot_angles(self, axes):
        titles = ['Roll', 'Pitch', 'Yaw']
        [plot_with_title(axes[i], self.time, np.rad2deg(self.states[3+i,:]), titles[i]) for i in range(3)]
