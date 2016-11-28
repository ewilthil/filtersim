import numpy as np
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal

default_NCV_params = {'sigma_x' : 1, 'sigma_y' : 1}
default_IOU_params = {'sigma_x' : 1, 'sigma_y' : 1, 'theta_x' : 1, 'theta_y' : 1, 'mu_x' : 0, 'mu_y' : 0}

class IntegratedOU(object):
    def __init__(self, theta, mu, sigma):
        self.theta = theta
        self.v_ref = mu
        self.sigma = sigma

    def get_model(self, x):
        return np.array([x[1], -self.theta*(x[1]-self.v_ref)]), np.array([0, self.sigma])

class TargetShip(object):
    def __init__(self, time, x_init, y_init, x_model, y_model):
        self.time = time
        self.dt = time[1]-time[0]
        self.states = np.zeros((4, len(time)))
        self.states[:2,0] = x_init
        self.states[2:,0] = y_init
        self.x_model = x_model
        self.y_model = y_model
        self.noise = multivariate_normal(0, np.sqrt(self.dt)).rvs(size=(2, len(time)))

    def step(self, idx):
        x_now = self.states[:2, idx-1]
        y_now = self.states[2:, idx-1]
        F_x, G_x = self.x_model.get_model(x_now)
        F_y, G_y = self.y_model.get_model(y_now)
        self.states[:2,idx] = x_now+F_x*self.dt+G_x*self.noise[0,idx]
        self.states[2:,idx] = y_now+F_y*self.dt+G_y*self.noise[1,idx]

    def plot_position(self, ax):
        ax.plot(self.states[2,:], self.states[0,:])
        ax.plot(self.states[2,0], self.states[0,0], 'ko')
        ax.set_aspect('equal')

    def plot_veloicty(self, axes):
        axes[0].plot(self.time, self.states[1,:])
        axes[1].plot(self.time, self.states[3,:])


class OwnShip(object):
    kinematic = np.arange(6)
    kinetic = np.arange(6)+6
    disturbance = np.arange(6)+12
    def __init__(self, x0, Ku, Kr):
        self.Kr = Kr
        self.Ku = Ku
        self.state = np.hstack((x0, np.zeros(6)))

    def set_motion_parameters(self):
        self.T_inv = np.linalg.inv(T)
        self.D = D

    def set_constant_time(self, dt):
        pass
    
    def ode(self, x, u):
        pass

    def step(self, u, w):
        pass

    def kinematic_ode(self):
        pass

    def kinetic_ode(self, u):
        pass
    
    def kinetic_step(self, u, dt):
        pass
    
    def disturbance_ode(self, w_c):
        return -self.T_inv.dot(self.state[self.disturbance])+w_c

    def disturbance_step(self, dt, w):
        from scipy.linalg import expm
        dist_Phi = expm(-self.T_inv*dt)
        return dist_Phi.dot(self.state[self.disturbance])+w

    def step(self, dt, u):
        self.current_input = u
        self.x_dot = self.ode(self.state, self.current_input)
        x_next = self.state + self.x_dot*dt
        self.state = x_next
        return self.state, self.x_dot

    def get_pos(self):
        return np.array([self.x[0], self.x[1]])

    def generate_trajectory(self, dt, u_ref):
        pass

    def get_imu(self, R=None):
        if R is None:
            v = np.zeros(6)
        else:
            v = multivariate_normal(cov=R).rvs()
        return np.array([self.x_dot[3], 0, 0, 0, 0, self.x_dot[2]])+v

    def get_gps(self, R=None):
        pass

class Ownship2D(OwnShip):
    def __init__(self, x0, Ku, Kr):
        # Use super to initiate the full state
        pass

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
        self.ref[self.nu, k] = np.array([ref[0], 0, -c_w*self.state[2, k-1], -c_p*self.state[self.phi, k-1], -c_q*self.state[self.theta, k-1], -c_r*(self.state[self.psi, k-1]-self.ref[self.psi,k])])
        self.noise[:,k] = self.noise_dist.rvs()
        if k == 0:
            pass
        else:
            self.state[:,k] = odeint(self.ode, self.state[:,k-1], np.array([0, self.dt]),args=(k,))[-1,:]
        self.state_diff[:,k] = self.ode(self.state[:,k], 0, k)
