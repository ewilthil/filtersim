import numpy as np
from scipy.stats import multivariate_normal
from autopy.conversion import euler_angles_to_matrix
from scipy.linalg import block_diag
from scipy.integrate import odeint

import matplotlib.pyplot as plt

class ShipPlant:
    def __init__(self, Q, D_nu, T_xi, state_init):
        self.pos = [0, 1, 2]
        self.eul = [3, 4, 5]
        self.eta = self.pos+self.eul
        self.nu = [6, 7, 8, 9, 10, 11]
        self.nu_ref = np.zeros(6)
        self.disturbance = [12, 13, 14, 15, 16, 17]
        self.noise_cov = Q
        self.D_nu = D_nu
        self.D_xi = -np.linalg.inv(T_xi)
        self.state = np.hstack((state_init, np.zeros(6)))
        self.noise = np.zeros(6)

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
    
    def kinetic_ode(self, x):
        return np.dot(self.D_nu, x[self.nu]-self.nu_ref)+x[self.disturbance]
    
    def disturbance_ode(self, x):
        return np.dot(self.D_xi, x[self.disturbance])+self.noise
    
    def ode(self, x, t):
        return np.hstack((self.kinematic_ode(x), self.kinetic_ode(x), self.disturbance_ode(x)))

    def step(self, u, dt):
        self.nu_ref[0] = u[0]
        self.nu_ref[5] = u[1]
        self.noise = multivariate_normal(cov=self.noise_cov).rvs()
        self.state = odeint(self.ode, self.state, np.array([0, dt]))[-1,:]
        state_diff = self.ode(self.state, 0)
        return self.state, state_diff

cargo_ship = {
        'D_nu' : -np.diag((0.5, 1, 10, 10, 10, 1)),
        'T_xi' : np.diag((30, 1, 30, 10, 10, 60)),
        'Q' : np.diag((1e-2, 1e-2, 1e-4, 1e-4, 1e-4, 5e-4)),
        'state_init' : np.array([0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0]),
        }
