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
for n_mc in range(N_MC):
    q0 = np.array([0,0,0,1])
    v0 = np.array([10,0,0])
    p0 = np.array([0,0,0])
    navsys = nav.NavigationSystem(q0, v0, p0, imu_time, gps_time)
    cov_radar = np.diag((5**2, (0.1*np.pi/180)**2))
    ownship_radar = Sensor(radar_measurement, np.zeros(2), cov_radar, radar_time)
    ground_radar = Sensor(radar_measurement, np.zeros(2), cov_radar, radar_time)

    track_init_pos = np.hstack((np.hstack((target.state[0:2,0], target.NED_vel(0)[0:2]))[[0,2,1,3]],0))
    track_init_cov = np.diag((25**2, 4**2, 25**2, 4**2, (1*np.pi/180)**2))
    pi_imm = np.array([[0.9, 0.1],[0.1, 0.9]])
    sigma_ct_u = np.diag((0.1**2, 0.1**2, (1*np.pi/180)**2))
    imm_init_pos = (track_init_pos[0:4], track_init_pos[0:4])
    imm_init_cov = (track_init_cov[0:4, 0:4], track_init_cov[0:4,0:4])
    sigma_dwna = np.diag((3**2, 3**2))
    sigma_ct = np.diag((0.07**2, 0.07**2))
    sigmas = (sigma_dwna, sigma_ct)
    extra_args = ({}, {})
    names = ('DWNA', 'CT_known')
    names_2 = ('DWNA', 'CT_unknown')
    extra_args_2 = ({}, {'omega' : np.deg2rad(-1.5)})
    prob_init = np.array([1, 0])
    perfect_pose_imm = track.IMM(pi_imm, radar_time, sigmas, imm_init_pos, imm_init_cov, prob_init, cov_radar, names, extra_args_2)
    perfect_pose_unknown_ct = track.IMM(pi_imm, radar_time, (sigma_dwna, sigma_ct_u), (track_init_pos[:4], track_init_pos), (track_init_cov[:4,:4], track_init_cov), prob_init, cov_radar, names_2, extra_args)
    navigation_imm = track.IMM(pi_imm, radar_time, sigmas, imm_init_pos, imm_init_cov, prob_init, cov_radar, names, extra_args_2)

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
            perfect_pose_imm.step(ownship_radar.data[:,k_radar], k_radar, perfect_pose, cov_radar)
            perfect_pose_unknown_ct.step(ownship_radar.data[:,k_radar], k_radar, perfect_pose, cov_radar)
            cov_heading = np.array([[0, 0],[0, navsys.EKF.cov_posterior[2,2,k_gps]]])
            navigation_imm.step(ownship_radar.data[:,k_radar], k_radar, navigation_pose, cov_radar+cov_heading)
            # Evaluate error stuff
            true_track_state = np.hstack((target.state[0,k],target.state_diff[0,k], target.state[1,k], target.state_diff[1,k]))
            nav_error = true_track_state-navigation_imm.est_posterior[:4,k_radar]
            nav_cov = navigation_imm.cov_posterior[:4,:4,k_radar]
            perf_error = true_track_state-perfect_pose_imm.est_posterior[:4,k_radar]
            perf_cov = perfect_pose_imm.cov_posterior[:4,:4,k_radar]
            NEES_nav[n_mc, k_radar] = np.dot(nav_error, np.dot(np.linalg.inv(nav_cov), nav_error))
            NEES_perf[n_mc, k_radar] = np.dot(perf_error, np.dot(np.linalg.inv(perf_cov), perf_error))
            RMSE_perf[n_mc, k_radar] = perf_error[0]**2+perf_error[2]**2
            RMSE_nav[n_mc, k_radar] = nav_error[0]**2+nav_error[2]**2

print str(datetime.datetime.now())
# Navigation results
#viz.plot_pos_err(ownship, navsys)
#viz.plot_vel_err(ownship, navsys,boxplot=False)
# Tracking results
xy_measurements = [polar_to_cartesian(ground_radar.data[:,k]) for k in range(len(radar_time))]
xy_measurements = np.vstack(xy_measurements).T
_, ax_xy = plt.subplots(1,2)
ax_xy[0].plot(ownship.state[1,:], ownship.state[0,:])
viz.target_xy(target, perfect_pose_imm, ax=ax_xy[0], measurements=xy_measurements)
ax_xy[0].set_title('IMM - perfect pose')
ax_xy[1].plot(ownship.state[1,:], ownship.state[0,:])
viz.target_xy(target, navigation_imm, ax=ax_xy[1], measurements=xy_measurements)
ax_xy[1].set_title('IMM - navigation pose')

viz.target_velocity(target, navigation_imm)
viz.target_velocity(target, perfect_pose_imm)
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

omega_est = np.zeros(len(radar_time))
for k, t in enumerate(radar_time):
    omega_est[k] = perfect_pose_imm.probabilites[1,k]*np.deg2rad(-1.5)
rate_fig, rate_ax = plt.subplots(3,1)
rate_ax[0].plot(navigation_imm.time, np.rad2deg(navigation_imm.est_posterior[4,:]))
rate_ax[0].plot(navigation_imm.time, np.rad2deg(perfect_pose_unknown_ct.est_posterior[4,:]))
rate_ax[0].plot(perfect_pose_imm.time, np.rad2deg(perfect_pose_imm.est_posterior[4,:]))
rate_ax[0].plot(target.time, np.rad2deg(target.state_diff[5,:]),label='true')
rate_ax[1].plot(navigation_imm.time, navigation_imm.probabilites[0,:], 'g')
rate_ax[1].plot(navigation_imm.time, navigation_imm.probabilites[1,:], 'b')
rate_ax[2].plot(perfect_pose_imm.time, perfect_pose_imm.probabilites[0,:], 'y')
rate_ax[2].plot(perfect_pose_imm.time, perfect_pose_imm.probabilites[1,:], 'r')
plt.show()
