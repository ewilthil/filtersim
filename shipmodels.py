import numpy as np
from scipy.linalg import block_diag, expm
from scipy.stats import multivariate_normal

def van_loan_discretization(dt, A, B=None, Q=None):
    n = A.shape[0]
    def get_input_mapping(A, B):
        if len(B.shape) == 1:
            B = B[np.newaxis].T
        m = B.shape[1]
        F_row_1 = np.hstack((A, B))
        F_row_2 = np.hstack((np.zeros((m,n)), np.zeros((m,m))))
        F = np.vstack((F_row_1, F_row_2))
        Fd = expm(F*dt)
        Ad = Fd[:n, :n]
        Bd = Fd[:n, n:]
        return Ad, Bd
    def get_noise_mapping(A, Q):
        F_row_1 = np.hstack((-A, Q))
        F_row_2 = np.hstack((np.zeros((n, n)), A.T))
        F = np.vstack((F_row_1, F_row_2))
        Fd = expm(F*dt)
        Ad = Fd[n:, n:].T
        Qd = Ad.dot(Fd[:n, n:])
        return Ad, Qd
    if B is not None:
        Ad, Bd = get_input_mapping(A, B)
    else:
        Bd = None
    if Q is not None:
        Ad, Qd = get_noise_mapping(A, Q)
    else:
        Qd = None
    return Ad, Bd, Qd

class IntegratedOU(object):
    def default_theta(self, n): return 0.5*np.ones(n)
    def default_sigma(self, n): return 0.3*np.ones(n)

    def __init__(self, dt, n_dim, thetas=None, sigmas=None):
        if thetas is None:
            thetas = self.default_theta(n_dim)
        if sigmas is None:
            sigmas = self.default_sigma(n_dim)
        Ad = [None for _ in range(n_dim)]
        Bd = [None for _ in range(n_dim)]
        Qd = [None for _ in range(n_dim)]
        for i in range(n_dim):
            A, B, Q = self.matrices_1D(thetas[i], sigmas[i])
            Ad[i], Bd[i], Qd[i] = van_loan_discretization(dt, A, B, Q)
        self.Ad = block_diag(*Ad)
        self.Bd = block_diag(*Bd)
        self.Qd = block_diag(*Qd)
        
    def step(self, x, u, v):
        return self.Ad.dot(x)+self.Bd.dot(u)+v

    def matrices_1D(self, theta, sigma):
        A = np.array([[0, 1], [0, -theta]])
        B = np.array([[0], [theta]])
        G = np.array([[0], [1]])
        Q = sigma*G.dot(G.T)
        return A, B, Q


class DiscreteWNA(object):
    def default_sigma(self, n): return 0.3*np.ones(n)

    def __init__(self, dt, n_dim, sigmas=None):
        if sigmas is None:
            sigmas = self.default_sigma(n_dim)
        Ad = [None for _ in range(n_dim)]
        Bd = [None for _ in range(n_dim)]
        Qd = [None for _ in range(n_dim)]
        for i, _ in enumerate(sigmas):
            A, B, Q = self.matrices_1D(sigmas[i])
            Ad[i], Bd[i], Qd[i] = van_loan_discretization(dt, A, B, Q)
        self.Ad = block_diag(*Ad)
        self.Bd = block_diag(*Bd)
        self.Qd = block_diag(*Qd)

    def step(self, x, u, v):
        return self.Ad.dot(x)+v

    def matrices_1D(self, sigma):
        A = np.array([[0, 1], [0, 0]])
        B = np.array([[0], [0]])
        G = np.array([[0], [1]])
        Q = sigma*G.dot(G.T)
        return A, B, Q

class IntegratedMOU(object):

    default_theta = 1*np.ones(2)
    default_sigma = 0.1*np.ones(2)

    def __init__(self, dt, thetas=default_theta, sigmas = default_sigma):
        Ad = [None, None]
        Bd = [None, None]
        Qd = [None, None]
        G = np.array([[0], [0], [1]])
        for i, _ in enumerate(thetas):
            A = np.array([[0, 1, 0],[0, 0, 1],[-thetas[i], 0, 0]])
            B = np.array([[0], [0], [thetas[i]]])
            Q = sigmas[i]*G.dot(G.T)
            Ad[i], Bd[i], Qd[i] = van_loan_discretization(dt, A, B, Q)
        self.Ad = block_diag(*Ad)
        self.Bd = block_diag(*Bd)
        self.Qd = block_diag(*Qd)

    def step(self, x, u, v):
        return self.Ad.dot(x)+self.Bd.dot(u)+v

class TargetShip(object):
    def __init__(self, time, model, x0):
        self.time = time
        self.dt = time[1]-time[0]
        self.states = np.zeros((len(x0), len(time)))
        self.states[:,0] = x0
        self.model = model
        self.noise = multivariate_normal(np.zeros_like(x0), model.Qd).rvs(size=len(time)).T

    def step(self, idx, v_ref):
        x_now = self.states[:, idx-1]
        self.states[:,idx] = self.model.step(x_now, v_ref, self.noise[:,idx])

    def plot_position(self, ax):
        ax.plot(self.states[2,:], self.states[0,:])
        ax.plot(self.states[2,0], self.states[0,0], 'ko')
        ax.set_aspect('equal')

    def plot_velocity(self, axes):
        axes[0].plot(self.time, self.states[1,:])
        axes[1].plot(self.time, self.states[3,:])

class Ownship(object):
    pass



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
