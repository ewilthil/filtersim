import numpy as np
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal

default_NCV_params = {'sigma_x' : 1, 'sigma_y' : 1}
default_IOU_params = {'sigma_x' : 1, 'sigma_y' : 1, 'theta_x' : 1, 'theta_y' : 1, 'mu_x' : 0, 'mu_y' : 0}

def set_model(name, args):
    NCV = 'NCV'
    IOU = 'IOU'
    if name is NCV:
        sigma_x = args[NCV]['sigma_x']
        sigma_y = args[NCV]['sigma_y']
        def A(state):
            return np.array([state[1], 0, state[3], 0])
        def B(state):
            B = np.zeros((4,2))
            B[1,0] = sigma_x
            B[3,1] = sigma_y
            return B
    elif name is IOU:
        sigma_x = args[IOU]['sigma_x']
        sigma_y = args[IOU]['sigma_y']
        theta_x = args[IOU]['theta_x']
        theta_y = args[IOU]['theta_y']
        mu_x = args[IOU]['mu_x']
        mu_y = args[IOU]['mu_y']
        def A(state):
            return np.array([state[1], theta_x*(mu_x-state[1]), state[3], theta_y*(mu_y-state[3])])
        def B(state):
            B = np.zeros((4,2))
            B[1,0] = sigma_x
            B[3,1] = sigma_y
            return B
    return A, B

class MouModel(object):
    def __init__(self, time, g1, g2, sigma, v_ref, x0):
        self.time = time
        self.states = np.zeros((2, len(time)))
        self.g1 = g1
        self.g2 = g2
        self.sigma = sigma
        self.v_ref = v_ref
        self.states[:,0] = x0
        self.x_now = x0
        self.x_ref = np.array([0, v_ref])
        self.A = np.array([[0, 1],[-g1, -g2]])
        self.B = np.array([0, sigma])
    
    def step(self, dt):
        v = np.sqrt(dt)*np.random.randn(1)
        x_next = self.x_now+self.A.dot(self.x_now-self.x_ref)*dt+self.B*(v)
        self.x_now = x_next
        return x_next

class TargetShip(object):
    def __init__(self,model, time, model_args):
        self.time = time
        self.dt = time[1]-time[0]
        self.states = np.zeros((4, len(time)))
        self.A, self.B = set_model(model, model_args)
        self.noise = multivariate_normal(np.zeros(2),self.dt*np.identity(2)).rvs(size=len(time)).T

    def set_initial_conditions(self, pos_0, vel_0):
        x0 = np.array([pos_0[0], vel_0[0], pos_0[1], vel_0[1]])
        self.states[:,0] = x0

    def step(self, idx, t_new):
        x_now = self.states[:,idx-1]
        self.states[:,idx] = x_now+self.A(x_now)*self.dt+self.B(x_now).dot(self.noise[:,idx])

    def update_model(self, name, args):
        self.A, self.B = set_model(name, args)

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
