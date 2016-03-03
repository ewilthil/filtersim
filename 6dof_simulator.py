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

N_MC = 1
target = load_pkl('target_long_duration.pkl')
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
    cov_radar = np.diag((5**2, (0.1*np.pi/180)**2))
    ownship_radar = Sensor(radar_measurement, np.zeros(2), cov_radar, radar_time)
    ground_radar = Sensor(radar_measurement, np.zeros(2), cov_radar, radar_time)

    track_init_pos = np.hstack((np.hstack((target.state[0:2,0], target.NED_vel(0)[0:2]))[[0,2,1,3]],0))
    track_init_cov = np.diag((20**2, 2**2, 20**2, 2**2, (0.1*np.pi/180)**2))
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
    sigma_dict = {
            DWNA : np.diag((0.3**2, 0.3**2)),
            CT_known : np.diag((0.1**2, 0.1**2)),
            CT_unknown : np.diag((0**2, 0**2, (0.1*np.pi/180)**2))
            }

    names = (DWNA, CT_unknown)
    pi = np.array([[0.9, 0.1],[0.1,0.9]])
    prob_init = np.array([0.9, 0.1])
    sigmas = tuple_elements(names, sigma_dict)
    imm_init_pos = tuple_elements(names, state_init_dict)
    imm_init_cov = tuple_elements(names, cov_init_dict)
    extra_args = ({}, {})
    unknown_rate_pose_perfect = track.IMM(pi, radar_time, sigmas, imm_init_pos, imm_init_cov, prob_init, cov_radar, names, extra_args)
    
    names = (DWNA, CT_known, CT_known)
    pi = np.array([[0.6, 0.2, 0.2],[0.3, 0.7, 0],[0.3, 0, 0.7]])
    prob_init = np.array([0.8, 0.1, 0.1])
    sigmas = tuple_elements(names, sigma_dict)
    imm_init_pos = tuple_elements(names, state_init_dict)
    imm_init_cov = tuple_elements(names, cov_init_dict)
    extra_args = ({}, {'omega' : np.deg2rad(-1.5)},{'omega' : np.deg2rad(1.5)})
    known_rate_pose_perfect = track.IMM(pi, radar_time, sigmas, imm_init_pos, imm_init_cov, prob_init, cov_radar, names, extra_args)

    names = (CT_known, CT_known, CT_known, DWNA, CT_known, CT_known, CT_known)
    pi = 0.6*np.identity(7)+0.2*np.eye(7,k=1)+0.2*np.eye(7,k=-1)
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
            cov_heading = np.array([[0, 0],[0, navsys.EKF.cov_posterior[2,2,k_gps]]])
            unknown_rate_pose_perfect.step(ownship_radar.data[:,k_radar], k_radar, perfect_pose, cov_radar)
            known_rate_pose_perfect.step(ownship_radar.data[:,k_radar], k_radar, perfect_pose, cov_radar)
            multi_imm.step(ownship_radar.data[:,k_radar], k_radar, perfect_pose, cov_radar)

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

viz.target_velocity(target, unknown_rate_pose_perfect)
viz.target_velocity(target, known_rate_pose_perfect)
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
rate_ax[0].plot(known_rate_pose_perfect.time, np.rad2deg(known_rate_pose_perfect.est_posterior[4,:]),label='known turn rate')
rate_ax[0].plot(unknown_rate_pose_perfect.time, np.rad2deg(unknown_rate_pose_perfect.est_posterior[4,:]), label='unknown turn rate')
rate_ax[0].plot(target.time, np.rad2deg(target.state_diff[5,:]), 'k', label='true rate')
rate_ax[0].plot(radar_time, omega_est, 'y--', label='manual rate')
rate_ax[0].set_title('Turn rate')
rate_ax[0].set_xlabel('time')
rate_ax[0].set_ylabel('deg/s')
rate_ax[1].plot(known_rate_pose_perfect.time, known_rate_pose_perfect.probabilites[0,:], label='DWNA probability')
rate_ax[1].plot(known_rate_pose_perfect.time, known_rate_pose_perfect.probabilites[1,:], label='CT probability (negative)')
rate_ax[1].plot(known_rate_pose_perfect.time, known_rate_pose_perfect.probabilites[2,:], label='CT probability (positive)')
rate_ax[1].set_title('Mode probabilities with known turn rate')
rate_ax[2].plot(unknown_rate_pose_perfect.time, unknown_rate_pose_perfect.probabilites[0,:], label='DWNA probability')
rate_ax[2].plot(unknown_rate_pose_perfect.time, unknown_rate_pose_perfect.probabilites[1,:], label='CT probability')
rate_ax[2].set_title('Mode probabilities with unknown turn rate')
for j in range(3):
    rate_ax[j].legend()
plt.show()
