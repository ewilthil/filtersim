import numpy as np
import matplotlib.pyplot as plt
from tf.transformations import euler_matrix
from autopy.conversion import euler_angles_to_quaternion, quaternion_to_euler_angles
from scipy.stats import multivariate_normal
from autopy.plotting import get_ellipse
import ipdb
def euler_to_matrix(ang):
        phi, theta, psi = ang[0], ang[1], ang[2]
        R = euler_matrix(psi, theta, phi, 'rzyx')
        return R[0:3,0:3]


# Navigation
def plot_angle_error(ship, navsys, ax=None):
    if ax == None:
        fig, ax = plt.subplots(3,1)
    time = ship.time
    true_eul = ship.state[3:6,:]
    est_quat = navsys.strapdown.data[navsys.strapdown.orient, :]
    est_eul = quaternion_to_euler_angles(est_quat.T).T
    meas_eul = quaternion_to_euler_angles(navsys.GPS.data[3:,:].T).T
    ax[0].plot(time, np.rad2deg(true_eul[0,:]), 'k', time, np.rad2deg(est_eul[0,:]), navsys.GPS.time, np.rad2deg(meas_eul[0,:]))
    ax[1].plot(time, np.rad2deg(true_eul[1,:]), 'k', time, np.rad2deg(est_eul[1,:]), navsys.GPS.time, np.rad2deg(meas_eul[1,:]))
    ax[2].plot(time, np.rad2deg(true_eul[2,:]), 'k', time, np.rad2deg(est_eul[2,:]), navsys.GPS.time, np.rad2deg(meas_eul[2,:]))
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

def plot_xy_pos(targets, ax, arg):
    for t in targets:
        plot_xy_trajectory(ax, t.state[1,:], t.state[0,:], arg)

def plot_xy_trajectory(ax, xdata, ydata, args):
    ax.plot(xdata, ydata, **args)

# Target tracking
def target_xy(target, est, ax, args):
    ax.plot(target.state[1,:], target.state[0,:], 'k')
    plot = ax.plot(est.est_posterior[2,:], est.est_posterior[0,:], **args)
    cov = est.cov_posterior[[[2],[0]],[2,0],:] # pick out E and N elements
    inds = np.arange(0, len(est.time)+1, len(est.time)/15)
    for k in inds:
        el = get_ellipse(est.est_posterior[[2,0],k], cov[:,:,k], gamma=6)
        el.set_color(plot[0].get_color())
        ax.add_artist(el)
    ax.set_aspect('equal')

def target_velocity(target, est, ax, args):
    ax[0].plot(target.time, target.state_diff[0,:], 'k', label='True state')
    ax[0].errorbar(est.time, est.est_posterior[1,:], yerr=2*np.sqrt(np.squeeze(est.cov_posterior[1,1,:])), errorevery=len(est.time)/30, **args)
    ax[0].set_title('North velocity')
    ax[1].plot(target.time, target.state_diff[1,:], 'k', label='True state')
    ax[1].errorbar(est.time, est.est_posterior[3,:], yerr=2*np.sqrt(np.squeeze(est.cov_posterior[3,3,:])), errorevery=len(est.time)/30, **args)
    ax[1].set_title('East velocity')

def target_velocity_error(target, est, ax, args):
    ax[0].plot(est.time, target.state_diff[0,0::100]-est.est_posterior[1,:], **args)
    ax[0].set_title('North velocity error')
    ax[1].plot(est.time, target.state_diff[1,0::100]-est.est_posterior[3,:], **args)
    ax[1].set_title('East velocity error')


def target_angular_rate(target, imm, ax, args):
    ax.plot(target.time, np.rad2deg(target.state_diff[5,:]), 'k', label='True state')
    ax.plot(imm.time, np.rad2deg(imm.est_posterior[4,:]), **args)
    ax.set_title('Angular rate')

def DWNA_probability(imm, ax, args):
    ax.plot(imm.time, imm.probabilities[0,:], **args)
    ax.set_title('Probability of constant velocity')

def likelihoods(imm, ax, args):
    local_args = ({'color' : 'k', 'label' : 'DWNA'}, {'color' : 'b', 'label' : '-CT'},{'color' : 'r', 'label' : 'CT'})
    [ax.plot(imm.time, imm.likelihoods[j,:], **local_args[j]) for j in range(3)]
    ax.set_title(args['label'])

def animate_likelihood(model1, model2):
    def update_mean_cov(num, model1, model2, ax, clr=False):
        if clr:
            ax.clear()
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        mean = model1.measurement_innovation[:,num]
        cov = model1.measurement_innovation_covariance[:,:,num]
        el = get_ellipse(np.zeros(2), cov, gamma=9, alpha =0.1)
        el.set_color('r')
        el.set_edgecolor('k')
        ax.add_artist(el)
        ax.plot(mean[0], mean[1], 'rx', markersize=10)
        if num >= 1:
            running_mean = np.mean(model1.measurement_innovation[:,max(num-15,0):num], axis=1)
            ax.plot(running_mean[0], running_mean[1], 'ro', markersize=10)
        mean = model2.measurement_innovation[:,num]
        cov = model2.measurement_innovation_covariance[:,:,num]
        el = get_ellipse(np.zeros(2), cov, gamma=9, alpha =0.1)
        el.set_color('b')
        el.set_edgecolor('k')
        ax.add_artist(el)
        ax.plot(mean[0], mean[1], 'bx', markersize=10)
        if num >= 1:
            running_mean = np.mean(model2.measurement_innovation[:,max(num-15,0):num], axis=1)
            ax.plot(running_mean[0], running_mean[1], 'bo', markersize=10)
        ax.set_title('k='+str(num))

    fig1 = plt.figure()
    plt.xlabel('x')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.title('likelihoods')
    line_ani = animation.FuncAnimation(fig1, update_mean_cov, len(model1.time), interval=100, blit=False, fargs=(model1, model2, fig1.get_axes()[0], True))
