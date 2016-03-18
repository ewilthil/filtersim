import numpy as np
import ipdb
from base_classes import pitopi
class EKF_navigation:
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
        #self.cov_posterior[:,:,k] = self.cov_prior[:,:,k] - np.dot(self.cov_prior[:,:,k],np.dot(H.T,np.dot(np.linalg.inv(S),np.dot(H,self.cov_prior[:,:,k]))))
        hea = np.identity(F.shape[0])-np.dot(K,self.H[:,:,k])
        cov = np.dot(hea, np.dot(self.cov_prior[:,:,k], hea.T))+np.dot(K, np.dot(self.R, K.T))
        self.cov_posterior[:,:,k] = cov
        return self.est_posterior[:,k], self.cov_posterior[:,:,k]

class KF:
    def __init__(self, F, H, Q, R, est_init, cov_init):
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R
        self.cov_prior = cov_init
        self.est_prior = est_init
        self.cov_posterior = cov_init
        self.est_posterior = est_init

    def step_markov(self):
        self.est_prior = np.dot(self.F, self.est_posterior)
        self.cov_prior = np.dot(self.F, np.dot(self.cov_posterior, self.F.T))+self.Q
        return self.est_prior, self.est_prior

    def step_filter(self, measurement):
        self.measurement = measurement
        self.S = np.dot(self.H, np.dot(self.cov_prior, self.H.T))+self.R
        K = np.dot(self.cov_prior, np.dot(self.H.T, np.linalg.inv(self.S)))
        K_pos = np.linalg.norm(np.vstack((np.linalg.norm(K[0,:]), np.linalg.norm(K[2,:]))))
        K_vel = np.linalg.norm(np.vstack((np.linalg.norm(K[1,:]), np.linalg.norm(K[3,:]))))
        self.measurement_prediction = np.dot(self.H, self.est_prior)
        self.est_posterior = self.est_prior+np.dot(K, measurement-self.measurement_prediction)
        cov = np.dot(np.identity(self.F.shape[0])-np.dot(K, self.H), self.cov_prior)
        self.cov_posterior = 0.5*(cov+cov.T)
        return self.est_posterior, self.cov_posterior, np.array([K_pos, K_vel])

class EKF:
    def __init__(self, f, h, Q, R, est_init, cov_init, F=None, H=None):
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.est_prior = est_init
        self.cov_prior = cov_init
        self.est_posterior = est_init
        self.cov_posterior = cov_init
        if F is not None:
            self.F = F
        else:
            #Construct the jacobian by differences
            pass
        if H is not None:
            self.H = H
        else:
            # Same
            pass
    def step_markov(self):
        self.est_prior = self.f(self.est_posterior)
        F_k = self.F(self.est_posterior)
        Q_k = self.Q(self.est_posterior)
        self.cov_prior = np.dot(F_k, np.dot(self.cov_posterior, F_k.T))+Q_k
        return self.est_prior, self.cov_prior
    
    def step_filter(self, measurement):
        self.measurement = measurement
        H_k = self.H(self.est_prior)
        self.S = np.dot(H_k, np.dot(self.cov_prior, H_k.T))+self.R
        K = np.dot(self.cov_prior, np.dot(H_k.T, np.linalg.inv(self.S)))

        
        #C = self.cov_prior[:4,4:]
        #B = self.cov_prior[4:,4:]
        #K[4:,:] = np.zeros((9,2))
        #Kb = K[4:,:]
        #H = H_k[:,:4]
        #Hb = H_k[:,4:]
        K_pos = np.linalg.norm(np.vstack((np.linalg.norm(K[0,0]), np.linalg.norm(K[2,0]))))
        K_vel = np.linalg.norm(np.vstack((np.linalg.norm(K[1,0]), np.linalg.norm(K[3,0]))))


        self.measurement_prediction = self.h(self.est_prior)
        innovation = self.measurement-self.measurement_prediction
        innovation[1] = pitopi(innovation[1])
        self.est_posterior = self.est_prior+np.dot(K, innovation)
        #self.est_posterior[4:] = np.zeros(9)
        nx = len(self.est_posterior)
        hea = np.identity(nx)-np.dot(K,H_k)
        cov = np.dot(hea, np.dot(self.cov_prior, hea.T))+np.dot(K, np.dot(self.R, K.T))
        if np.min(np.linalg.eig(cov)[0]) < 0:
            ipdb.set_trace()
        #cov = np.dot(np.identity(nx)-np.dot(K, H_k), self.cov_prior)
        #cov[4:,:4] = cov[:4,4:].T
        #cov[:4,4:] = cov[4:,:4].T
        self.cov_posterior = cov#0.5*(cov+cov.T)
        #print np.linalg.eig(cov)[0]
        #print np.linalg.eig(self.cov_posterior)[0]
        #print '-----------------------' 
        return self.est_posterior, self.cov_posterior, np.array([K_pos, K_vel])
