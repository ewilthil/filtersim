import numpy as np
import matplotlib.pyplot as plt
from tf.transformations import euler_matrix
from autopy.conversion import euler_angles_to_quaternion
from scipy.stats import multivariate_normal

def euler_to_matrix(ang):
        phi, theta, psi = ang[0], ang[1], ang[2]
        R = euler_matrix(psi, theta, phi, 'rzyx')
        return R[0:3,0:3]




def plot_pos_err(ship, navsys, ax=None):
    if ax == None:
        fig, ax = plt.subplots(3,1)
    true_err = ship.state[0:3,:]-navsys.strapdown.data[navsys.strapdown.pos,:]
    est_err = navsys.EKF.est_posterior[6:9,:]
    est_cov = navsys.EKF.cov_posterior[6:9,6:9,:]
    for n_ax in range(3):
        ax[n_ax].plot(ship.time, true_err[n_ax,:])
        yerr = 3*np.sqrt(np.squeeze(est_cov[n_ax, n_ax, :]))
        ax[n_ax].errorbar(navsys.EKF.time, est_err[n_ax, :] ,yerr=yerr,errorevery=len(navsys.EKF.time)/20)

def plot_vel_err(ship, navsys, ax=None, boxplot=False):
    vel = np.zeros((3,len(ship.time)))
    for k, _ in enumerate(ship.time):
        C = euler_to_matrix(ship.state[3:6,k])
        vel[:,k] = np.dot(C,ship.state[6:9,k])
    if ax == None:
        fig, ax = plt.subplots(3,1)
    true_err = vel-navsys.strapdown.data[navsys.strapdown.vel,:]
    est_err = navsys.EKF.est_posterior[6:9,:]
    est_cov = navsys.EKF.cov_posterior[6:9,6:9,:]
    for n_ax in range(3):
        ax[n_ax].plot(ship.time, true_err[n_ax,:])
        yerr = 3*np.sqrt(np.squeeze(est_cov[n_ax, n_ax, :]))
        ax[n_ax].errorbar(navsys.EKF.time, est_err[n_ax, :],yerr=yerr, errorevery=len(navsys.EKF.time)/20)
    if boxplot:
        box_plot_vel(est_err, est_cov)

def plot_stuff(ax, x, y, title='', xlabel='', ylabel='', legend=''):
    ax.plot(x,y,label=legend)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

def plot_strapdown_pos(NE_ax, D_ax, strapdown):
    NED = strapdown.data[strapdown.pos,:]
    NE_ax.plot(NED[1,:], NED[0,:],label='strapdown')
    D_ax.plot(strapdown.time, -NED[2,:],label='strapdown')
def plot_true_pos(NE_ax, D_ax, state, time):
    NED = state[0:3,:]
    NE_ax.plot(NED[1,:], NED[0,:],label='true')
    D_ax.plot(time, -NED[2,:],label='true')

def plot_strapdown_vel(ax, strapdown):
    vel = strapdown.data[strapdown.vel,:]
    for i in range(3):
        ax[i].plot(strapdown.time, vel[i,:], label='strapdown')

def plot_true_vel(ax, state, time):
    vel = np.zeros((3,len(time)))
    for k, _ in enumerate(time):
        C = euler_to_matrix(state[3:6,k])
        vel[:,k] = np.dot(C,state[6:9,k])
    for i in range(3):
        ax[i].plot(time, vel[i,:], label='true')

def plot_quat(quat_ax, strapdown, true_state, time):
    quats = euler_angles_to_quaternion(true_state[3:6,:].T)
    for i in range(4):
        quat_ax[i].plot(strapdown.time, strapdown.data[strapdown.orient[i],:],label='strapdown')     
        quat_ax[i].plot(time, quats[:,i], label='true')
        plt.legend()

def box_plot_vel(err, cov):
    K = err.shape[1]
    rel_err = np.zeros(K)
    rel_gauss = np.zeros(K)
    gauss_err = multivariate_normal(cov=np.identity(3))
    for k in range(K):
        rel_err[k] = np.sqrt(np.dot(err[:,k], np.dot(np.linalg.inv(cov[:,:,k]), err[:,k])))
        a = gauss_err.rvs()
        rel_gauss[k] = np.linalg.norm(a)
    plt.figure()
    plt.boxplot([rel_err, rel_gauss],whis=[0,95],labels=['filter results', 'gauss'])
    plt.title('Velocity consistency analysis')

def plot_xy_pos(targets):
    fig, ax = plt.subplots(1,1)
    for t in targets:
        ax.plot(t.state[1,:], t.state[0,:],)
    return fig, ax
