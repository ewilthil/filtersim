import numpy as np
class EKF:
    def __init__(self, f, h, R, est_init, cov_init, time, F=None, H=None):
        self.f = f
        self.h = h
        self.R = R
        self.time = time
        self.dt = time[1]-time[0]
        self.N = len(time)
        self.nx = len(est_init)
        self.F = np.zeros((self.nx, self.nx, self.N))
        self.H = np.zeros((self.R.shape[0],self.nx,self.N))
        self.cov_prior = np.zeros((self.nx, self.nx, self.N))
        self.cov_prior[:,:,0] = cov_init
        self.cov_posterior = np.zeros((self.nx, self.nx, self.N))
        self.est_posterior = np.zeros((self.nx, self.N))

    def step(self, z, z_est, F, H, Q, k):
        # Update F and H
        self.F[:,:,k] = F
        self.H[:,:,k] = H
        # Predict
        if k > 0:
            self.cov_prior[:,:,k] = np.dot(F,np.dot(self.cov_posterior[:,:,k-1],F.T))+Q
        # Update
        S = np.dot(H,np.dot(self.cov_prior[:,:,k],H.T))+self.R
        K = np.dot(self.cov_prior[:,:,k], np.dot(H.T, np.linalg.inv(S)))
        self.est_posterior[:,k] = np.dot(K,z-z_est)
        self.cov_posterior[:,:,k] = self.cov_prior[:,:,k] - np.dot(self.cov_prior[:,:,k],np.dot(H.T,np.dot(np.linalg.inv(S),np.dot(H,self.cov_prior[:,:,k]))))
        return self.est_posterior[:,k], self.cov_posterior[:,:,k]


class KF:
    def __init__(self, F, H, Q, R, est_init, cov_init):
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R
        self.cov_posterior = cov_init
        self.est_posterior = est_init

    def step_markov(self):
        self.est_prior = np.dot(F, self.est_posterior)
        self.cov_prior = np.dot(self.F, np.dot(self.est_posterior, self.F.T))+self.Q
        return est_prior, est_posterior

    def step_filter(self, measurement):
        S = np.dot(self.H, np.dot(self.cov_prior, self.H.T))+R
        K = np.dot(self.cov_prior, np.dot(self.H.T, np.linalg.inv(S)))
        self.est_posterior = np.dot(K, measurement-self.H*self.est_prior)
        cov = np.dot(np.identity(self.F.shape[0])-np.dot(K, self.H), self.cov_prior)
        self.cov_posterior = 0.5*(cov+cov.T)
        return est_posterior, cov_posterior
