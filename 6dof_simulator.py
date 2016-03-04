import numpy as np
import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion
import navigation as nav
import visualization as viz
import tracking as track
import datetime
from base_classes import Model, Sensor
from scipy.stats import chi2
from autopy.sylte import load_pkl
from autopy.conversion import quaternion_to_euler_angles
from autopy.plotting import get_ellipse
import matplotlib.animation as animation
plt.close('all')

def pitopi(ang):
    return (ang+np.pi)%(2*np.pi)-np.pi

def radar_measurement(x, x0):
    R = np.sqrt((x[0]-x0[0])**2+(x[1]-x0[1])**2)
    alpha = np.arctan2(x[1]-x0[1], x[0]-x0[0])-x0[5]
    alpha = pitopi(alpha)
    return np.array([R, alpha])

def polar_to_cartesian(data):
    x = data[0]*np.cos(data[1])
    y = data[0]*np.sin(data[1])
    return np.array([x,y])

def ct_cov(w, T):
    wT = w*T
    swT = np.sin(wT)
    cwT = np.cos(wT)
    return np.array([[2*(wT-swT)/w**3, (1-cwT)/w**2, 0, (wT-swT)/w**2],
        [(1-cwT)/w**2, T, -(wT-swT)/w**2, 0],
        [0, -(wT-swT)/w**2, 2*(wT-swT)/w**3, (1-cwT)/w**2],
        [(wT-swT)/w**2, 0, (1-cwT)/w**2, T]])

N_MC = 1
target = load_pkl('target_constant_turn.pkl')
ownship = load_pkl('ownship_traj.pkl')
time = ownship.time
dt = time[1]-time[0]
Tend = time[-1]
M_imu = 1
M_gps = 20
M_radar = 100
imu_time = np.arange(0, Tend+M_imu*dt, M_imu*dt)
gps_time = np.arange(0, Tend+M_gps*dt, M_gps*dt)
radar_time = np.arange(0, Tend+M_radar*dt, M_radar*dt)
NEES_nav = np.zeros((N_MC, len(radar_time)))
RMSE_nav = np.zeros((N_MC, len(radar_time)))
NEES_perf = np.zeros((N_MC, len(radar_time)))
RMSE_perf = np.zeros((N_MC, len(radar_time)))
def tuple_elements(names, dictionary):
    return tuple([dictionary[name] for name in names])
for n_mc in range(N_MC):
    q0 = np.array([0,0,0,1])
    v0 = np.array([10,0,0])
    p0 = np.array([0,0,0])
    navsys = nav.NavigationSystem(q0, v0, p0, imu_time, gps_time)
    cov_radar = np.diag((20**2, (0.5*np.pi/180)**2))
    ownship_radar = Sensor(radar_measurement, np.zeros(2), cov_radar, radar_time)
    ground_radar = Sensor(radar_measurement, np.zeros(2), cov_radar, radar_time)

    track_init_pos = np.hstack((np.hstack((target.state[0:2,0], target.NED_vel(0)[0:2]))[[0,2,1,3]],0))
    track_init_cov = np.diag((10**2, 1**2, 10**2, 1**2, (0.1*np.pi/180)**2))
    dwna_cov = 0.18**2
    ct_cov = 0.12**2
    DWNA = 'DWNA'
    CT_known = 'CT_known'
    CT_unknown = 'CT_unknown'
    state_init_dict = {
            DWNA : track_init_pos[:4],
            CT_known : track_init_pos[:4],
            CT_unknown : track_init_pos,
            }
    cov_init_dict = {
            DWNA : track_init_cov[:4,:4],
            CT_known : track_init_cov[:4,:4],
            CT_unknown : track_init_cov,
            }
    radar_dt = dt*M_radar
    G_CT = np.array([[radar_dt**2/2., 0, 0],[radar_dt, 0, 0],[0,radar_dt**2/2., 0],[0, radar_dt, 0],[0, 0, radar_dt]])
    G_DWNA = np.array([[radar_dt**2/2., 0],[radar_dt, 0],[0,radar_dt**2/2.],[0, radar_dt]])
    sigma_dict = {
            DWNA : np.dot(G_DWNA, np.dot(np.diag((dwna_cov, dwna_cov)),G_DWNA.T)),
            CT_known : np.diag((ct_cov, ct_cov)),
            CT_unknown : np.diag((ct_cov, np.deg2rad(0.1)**2))
            }

    names = (DWNA, CT_unknown)
    pi = np.array([[0.7, 0.3],[0.3,0.7]])
    prob_init = np.array([0.9, 0.1])
    sigmas = tuple_elements(names, sigma_dict)
    imm_init_pos = tuple_elements(names, state_init_dict)
    imm_init_cov = tuple_elements(names, cov_init_dict)
    extra_args = ({}, {})
    unknown_rate_pose_perfect = track.IMM(pi, radar_time, sigmas, imm_init_pos, imm_init_cov, prob_init, cov_radar, names, extra_args)
    
    names = (DWNA, CT_known, CT_known)
    pi = np.array([[0.5, 0.25, 0.25],[0.1, 0.9, 0],[0.1, 0, 0.9]])
    prob_init = np.array([0.8, 0.1, 0.1])
    sigmas = tuple_elements(names, sigma_dict)
    imm_init_pos = tuple_elements(names, state_init_dict)
    imm_init_cov = tuple_elements(names, cov_init_dict)
    extra_args = ({}, {'omega' : np.deg2rad(-2)},{'omega' : np.deg2rad(2)})
    known_rate_pose_perfect = track.IMM(pi, radar_time, sigmas, imm_init_pos, imm_init_cov, prob_init, cov_radar, names, extra_args)

    names = (CT_known, CT_known, CT_known, DWNA, CT_known, CT_known, CT_known)
    pi = 0.5*np.identity(7)+0.25*np.eye(7,k=1)+0.25*np.eye(7,k=-1)
    pi[0,1] += 0.25
    pi[-1,-2] += 0.25

    prob_init = np.zeros(7)
    prob_init[3] = 1
    sigmas = tuple_elements(names, sigma_dict)
    imm_init_pos = tuple_elements(names, state_init_dict)
    imm_init_cov = tuple_elements(names, cov_init_dict)
    extra_args = ({'omega' : np.deg2rad(-3)},{'omega' : np.deg2rad(-2)},{'omega' : np.deg2rad(-1)},{},{'omega' : np.deg2rad(1)},{'omega' : np.deg2rad(2)},{'omega' : np.deg2rad(3)},)
    multi_imm = track.IMM(pi, radar_time, sigmas, imm_init_pos, imm_init_cov, prob_init, cov_radar, names, extra_args)

    # Main loop
    print str(datetime.datetime.now())
    for k, t in enumerate(time):
        # Generate sensor data and update navigation / tracking
        k_imu, rest_imu = int(np.floor(k/M_imu)), np.mod(k,M_imu)
        if rest_imu == 0:
            navsys.step_strapdown(ownship.state[:,k], ownship.state_diff[:,k], k_imu)
        k_gps, rest_gps = int(np.floor(k/M_gps)), np.mod(k,M_gps)
        if rest_gps == 0:
            navsys.step_filter(ownship.state[:,k], k_imu, k_gps)
        k_radar, rest_radar = int(np.floor(k/M_radar)), np.mod(k, M_radar)
        if rest_radar == 0:
            ownship_radar.generate_measurement((target.state[:,k], ownship.state[:,k]), k_radar)
            ground_radar.generate_measurement((target.state[:,k], np.zeros(6)), k_radar)
            nav_quat, _, nav_pos, _, _ = navsys.get_strapdown_estimate(k_imu)
            nav_eul = quaternion_to_euler_angles(nav_quat)
            navigation_pose = np.hstack((nav_pos[0:2], nav_eul[2]))
            perfect_pose = np.hstack((ownship.state[0:2,k], ownship.state[5,k]))
            perfect_pose = np.zeros(3)
            cov_heading = np.array([[0, 0],[0, navsys.EKF.cov_posterior[2,2,k_gps]]])
            unknown_rate_pose_perfect.step(ground_radar.data[:,k_radar], k_radar, perfect_pose, cov_radar)
            known_rate_pose_perfect.step(ground_radar.data[:,k_radar], k_radar, perfect_pose, cov_radar)
            multi_imm.step(ground_radar.data[:,k_radar], k_radar, perfect_pose, cov_radar)
            #unknown_rate_pose_perfect.step(ownship_radar.data[:,k_radar], k_radar, perfect_pose, cov_radar)
            #known_rate_pose_perfect.step(ownship_radar.data[:,k_radar], k_radar, perfect_pose, cov_radar)
            #multi_imm.step(ownship_radar.data[:,k_radar], k_radar, perfect_pose, cov_radar)

            # Evaluate error stuff
            #true_track_state = np.hstack((target.state[0,k],target.state_diff[0,k], target.state[1,k], target.state_diff[1,k]))
            #nav_error = true_track_state-navigation_imm.est_posterior[:4,k_radar]
            #nav_cov = navigation_imm.cov_posterior[:4,:4,k_radar]
            #perf_error = true_track_state-perfect_pose_imm.est_posterior[:4,k_radar]
            #perf_cov = perfect_pose_imm.cov_posterior[:4,:4,k_radar]
            #NEES_nav[n_mc, k_radar] = np.dot(nav_error, np.dot(np.linalg.inv(nav_cov), nav_error))
            #NEES_perf[n_mc, k_radar] = np.dot(perf_error, np.dot(np.linalg.inv(perf_cov), perf_error))
            #RMSE_perf[n_mc, k_radar] = perf_error[0]**2+perf_error[2]**2
            #RMSE_nav[n_mc, k_radar] = nav_error[0]**2+nav_error[2]**2

print str(datetime.datetime.now())
# Navigation results
#viz.plot_pos_err(ownship, navsys)
#viz.plot_vel_err(ownship, navsys,boxplot=False)
# Tracking results
xy_measurements = [polar_to_cartesian(ground_radar.data[:,k]) for k in range(len(radar_time))]
xy_measurements = np.vstack(xy_measurements).T
_, ax_xy = plt.subplots(1,2)
ax_xy[0].plot(ownship.state[1,:], ownship.state[0,:])
viz.target_xy(target, unknown_rate_pose_perfect, ax=ax_xy[0], measurements=xy_measurements)
ax_xy[0].set_title('IMM - perfect pose, unknown rate')
ax_xy[1].plot(ownship.state[1,:], ownship.state[0,:])
viz.target_xy(target, known_rate_pose_perfect, ax=ax_xy[1], measurements=xy_measurements)
ax_xy[1].set_title('IMM - perfect pose, known rate')

_, ax = viz.target_velocity(target, unknown_rate_pose_perfect)
ax[0].set_title('unknown rate')
_, ax = viz.target_velocity(target, known_rate_pose_perfect)
ax[0].set_title('known rate')
# NEES plot
UB = chi2(df=2*N_MC).ppf(0.975)/N_MC*np.ones_like(radar_time)
LB = chi2(df=2*N_MC).ppf(0.025)/N_MC*np.ones_like(radar_time)
NEES_fig, consistency_ax = plt.subplots(1,2)
NEES_ax = consistency_ax[0]
NEES_ax.plot(radar_time, UB, 'k')
NEES_ax.plot(radar_time, LB, 'k')
NEES_ax.plot(radar_time, np.mean(NEES_nav, axis=0), label='navigation pose')
NEES_ax.plot(radar_time, np.mean(NEES_perf, axis=0), label='perfect pose')
NEES_ax.legend()
NEES_ax.set_title('NEES of tracking for ' + str(N_MC) + ' monte carlo runs')
RMS_ax = consistency_ax[1]
RMS_ax.plot(radar_time, np.sqrt(np.mean(RMSE_nav, axis=0)), label='navigation pose')
RMS_ax.plot(radar_time, np.sqrt(np.mean(RMSE_perf, axis=0)), label='perfect pose')
RMS_ax.legend()
RMS_ax.set_title('Position RMS error for ' + str(N_MC) + ' monte carlo runs')

omega_est = np.zeros_like(radar_time)
for k, t in enumerate(radar_time):
    omega_est[k] = known_rate_pose_perfect.probabilites[1,k]*(-1.5)+known_rate_pose_perfect.probabilites[2,k]*(1.5)
rate_fig, rate_ax = plt.subplots(3,1)
rate_ax[0].plot(known_rate_pose_perfect.time, np.rad2deg(known_rate_pose_perfect.est_posterior[4,:]),label='known turn rate, 3 models')
rate_ax[0].plot(radar_time, np.rad2deg(multi_imm.est_posterior[4,:]), label='known turn rate, 7 models')
rate_ax[0].plot(unknown_rate_pose_perfect.time, np.rad2deg(unknown_rate_pose_perfect.est_posterior[4,:]), label='unknown turn rate')
rate_ax[0].plot(target.time, np.rad2deg(target.state_diff[5,:]), 'k', label='true rate')
rate_ax[0].set_title('Estimated turn rate')
rate_ax[0].set_xlabel('time')
rate_ax[0].set_ylabel('deg/s')
rate_ax[1].plot(known_rate_pose_perfect.time, known_rate_pose_perfect.probabilites[0,:], label='known turn rate, 3 models')
rate_ax[1].plot(multi_imm.time, multi_imm.probabilites[3,:], label='known turn rate, 7 models')
rate_ax[1].plot(known_rate_pose_perfect.time, unknown_rate_pose_perfect.probabilites[0,:], label='unknown turn rate')
rate_ax[1].set_title('Probability of constant velocity model')
rate_ax[2].plot(known_rate_pose_perfect.time, known_rate_pose_perfect.likelihoods[0,:], 'b', label='known turn rate, 3 models - dwna likelihood')
rate_ax[2].plot(known_rate_pose_perfect.time, np.sum(known_rate_pose_perfect.likelihoods[1:,:],axis=0), 'b--', label='known turn rate, 3 models - turn likelihood')
rate_ax[2].plot(multi_imm.time, multi_imm.likelihoods[3,:], 'g', label='known turn rate, 7 models, dwna')
rate_ax[2].plot(multi_imm.time, np.sum(multi_imm.likelihoods[:3,:],axis=0)+np.sum(multi_imm.likelihoods[4:,:],axis=0), 'g--', label='known turn rate, 7 models, ct')
rate_ax[2].plot(known_rate_pose_perfect.time, unknown_rate_pose_perfect.likelihoods[0,:], 'r', label='unknown turn rate, dwna')
rate_ax[2].plot(known_rate_pose_perfect.time, unknown_rate_pose_perfect.likelihoods[1,:], 'r--', label='unknown turn rate, ct')
rate_ax[2].set_title('likelihood of DWNA')
#rate_ax[2].plot(unknown_rate_pose_perfect.time, np.ones_like(known_rate_pose_perfect.probabilites[0,:])-known_rate_pose_perfect.probabilites[0,:], label='known turn rate, 3 models')
#rate_ax[2].plot(unknown_rate_pose_perfect.time, np.ones_like(multi_imm.probabilites[3,:])-multi_imm.probabilites[3,:], label='known turn rate, 7 models')
#rate_ax[2].plot(unknown_rate_pose_perfect.time, unknown_rate_pose_perfect.probabilites[1,:], label='unknown turn rate')
#rate_ax[2].set_title('Probability of turning')
for j in range(3):
    rate_ax[j].legend()
hea_fig, hea_ax = plt.subplots(1,1)
hea_ax.plot(unknown_rate_pose_perfect.time, known_rate_pose_perfect.likelihoods[0,:], 'k')
hea_ax.plot(unknown_rate_pose_perfect.time, known_rate_pose_perfect.likelihoods[1,:], 'b')
hea_ax.plot(unknown_rate_pose_perfect.time, known_rate_pose_perfect.likelihoods[2,:], 'r')
vel_fig, vel_ax = plt.subplots(2,1)
vel_ax[0].plot(target.time, target.state_diff[0,:], 'k', label='true north velocity')
vel_ax[0].plot(radar_time, known_rate_pose_perfect.filter_bank[0].est_posterior[1,:], label='dwna')
vel_ax[0].plot(radar_time, known_rate_pose_perfect.filter_bank[1].est_posterior[1,:], label='ct')
vel_ax[0].plot(radar_time, known_rate_pose_perfect.filter_bank[2].est_posterior[1,:], label='ct')
vel_ax[0].legend()
vel_ax[1].plot(target.time, target.state_diff[1,:], 'k', label='true east velocity')
vel_ax[1].plot(radar_time, known_rate_pose_perfect.filter_bank[0].est_posterior[3,:], 'b', label='dwna')
vel_ax[1].plot(radar_time, known_rate_pose_perfect.filter_bank[1].est_posterior[3,:], 'g', label='ct')
vel_ax[1].plot(radar_time, known_rate_pose_perfect.filter_bank[2].est_posterior[3,:], 'r', label='ct')
vel_ax[1].plot(radar_time, known_rate_pose_perfect.filter_bank[0].est_prior[3,:], 'k--', label='dwna prior')
vel_ax[1].plot(radar_time, known_rate_pose_perfect.filter_bank[1].est_prior[3,:], 'k--', label='ct prior')
vel_ax[1].plot(radar_time, known_rate_pose_perfect.filter_bank[2].est_prior[3,:], 'k--', label='ct prior')
vel_ax[1].legend()
pos_fig, pos_ax = plt.subplots(2,1)
#pos_ax[1].plot(target.time, target.state_diff[1,:], 'k', label='true east position')
pos_ax[1].plot(radar_time, known_rate_pose_perfect.filter_bank[0].est_posterior[2,:]-known_rate_pose_perfect.filter_bank[0].est_prior[2,:], 'b', label='dwna')
pos_ax[1].plot(radar_time, known_rate_pose_perfect.filter_bank[1].est_posterior[2,:]-known_rate_pose_perfect.filter_bank[1].est_prior[2,:], 'g', label='ct')
pos_ax[1].plot(radar_time, known_rate_pose_perfect.filter_bank[2].est_posterior[2,:]-known_rate_pose_perfect.filter_bank[2].est_prior[2,:], 'r', label='ct')
#pos_ax[1].plot(radar_time, known_rate_pose_perfect.filter_bank[1].est_posterior[2,:], 'g', label='ct')
#pos_ax[1].plot(radar_time, known_rate_pose_perfect.filter_bank[2].est_posterior[2,:], 'r', label='ct')
#pos_ax[1].plot(radar_time, known_rate_pose_perfect.filter_bank[0].est_prior[2,:], 'k--', label='dwna prior')
#pos_ax[1].plot(radar_time, known_rate_pose_perfect.filter_bank[1].est_prior[2,:], 'k--', label='ct prior')
#pos_ax[1].plot(radar_time, known_rate_pose_perfect.filter_bank[2].est_prior[2,:], 'k--', label='ct prior')
pos_ax[1].legend()

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
line_ani = animation.FuncAnimation(fig1, update_mean_cov, len(radar_time), interval=100, blit=False, fargs=(known_rate_pose_perfect.filter_bank[0], known_rate_pose_perfect.filter_bank[1], fig1.get_axes()[0], True))
plt.show()
