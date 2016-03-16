import numpy as np
import matplotlib.pyplot as plt
import navigation as nav
import visualization as viz
import tracking as track
import datetime
import autopy.conversion as conv
from base_classes import Model, Sensor, ErrorStats, radar_measurement
from autopy.sylte import load_pkl
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal
plt.close('all')

N_MC = 10
target = load_pkl('target_straight.pkl')
ownship = load_pkl('ownship_straight.pkl')
time = ownship.time
dt = time[1]-time[0]
Tend = time[-1]
M_imu = 1
M_gps = 20
M_radar = 100
imu_time = np.arange(0, Tend+M_imu*dt, M_imu*dt)
gps_time = np.arange(0, Tend+M_gps*dt, M_gps*dt)
radar_time = np.arange(0, Tend+M_radar*dt, M_radar*dt)
radar_dt = M_radar*dt

# Initial states for different stuff
q0 = conv.euler_angles_to_quaternion(ownship.state[3:6,0])
v0 = ownship.state_diff[0:3,0]
p0 = ownship.state[0:3,0]
track_state_init = np.hstack((target.state[0,0], target.state_diff[0,0], target.state[1,0], target.state_diff[1,0]))
# Covariances
cov_radar = np.diag((50**2, (1*np.pi/180)**2))
track_cov_init = np.diag((5**2, 1**2, 5**2, 1**2))
acc_cov = 1**2
Q_sub = np.array([[radar_dt**4/4, radar_dt**3/2],[radar_dt**3/2, radar_dt**2]])
Q_DWNA = acc_cov*block_diag(Q_sub, Q_sub)

#Plot arguments
own_arg = {'color' : 'k', 'label' : 'Ownship'}
schmidt_args = {'color' : 'g', 'label': 'Schmidt', 'linestyle' : '-'}
ground_truth_args = {'color' : 'b', 'label' : 'Ground truth pose'}
uncomp_args = {'color' : 'r', 'label' : 'Uncompensated', 'linestyle' : '-'}
conv_meas_args = {'color' : 'y', 'label' : 'Converted measurement'}
ucomp_ekf_args = {'color' : 'm', 'label' : 'EKF ucomp'}
arg = (ground_truth_args, uncomp_args, schmidt_args, conv_meas_args)
errs = ()
# Error statistics
for a in arg:
    errs =errs+(ErrorStats(radar_time, N_MC, a),)

ang_args = {'color' : 'g', 'label': 'Angle', 'linestyle' : '-'}
vel_args = {'color' : 'b', 'label' : 'Velocity'}
pos_args = {'color' : 'r', 'label' : 'Position', 'linestyle' : '-'}
total_args = {'color' : 'y', 'label' : 'Total'}
bias_args = {'color' : 'c', 'label' : 'Acc bias'}
gyr_bias_args = {'color' : 'm', 'label' : 'Gyr bias'}
nav_args = (ang_args, vel_args, pos_args, total_args)
nav_errs = ()
for a in nav_args:
    nav_errs = nav_errs+(ErrorStats(gps_time, N_MC, a),)
#schmidt_errs = ErrorStats(radar_time, N_MC, schmidt_args)
#ground_truth_errs = ErrorStats(radar_time, N_MC, ground_truth_args)
#uncomp_errs = ErrorStats(radar_time, N_MC, uncomp_args)
#conv_meas_errs = ErrorStats(radar_time, N_MC, conv_meas_args)
# Group up
for n_mc in range(N_MC):
    # (re)initialize instances
    navsys = nav.NavigationSystem(q0, v0, p0, imu_time, gps_time)
    ownship_radar = Sensor(radar_measurement, np.zeros(2), cov_radar, radar_time)
    track_state_k = track_state_init+multivariate_normal(cov=track_cov_init).rvs()
    schmidt = track.DWNA_schmidt(radar_time, Q_DWNA, cov_radar, track_state_k, track_cov_init)
    ground_truth = track.DWNA_filter(radar_time, Q_DWNA, cov_radar, track_state_k, track_cov_init)
    uncompensated_tracker = track.DWNA_filter(radar_time, Q_DWNA, cov_radar, track_state_k, track_cov_init)
    conv_meas = track.DWNA_filter(radar_time, Q_DWNA, cov_radar, track_state_k, track_cov_init)
    uncomp_EKF = track.DWNA_nocomp(radar_time, Q_DWNA, cov_radar, track_state_k, track_cov_init)
    trackers = (ground_truth, uncompensated_tracker, schmidt, conv_meas)

    
    # run the navigation and tracking filters
    for k, t in enumerate(time):
        k_imu, rest_imu = int(np.floor(k/M_imu)), np.mod(k,M_imu)
        if rest_imu == 0:
            navsys.step_strapdown(ownship.state[:,k], ownship.state_diff[:,k], k_imu)
        k_gps, rest_gps = int(np.floor(k/M_gps)), np.mod(k,M_gps)
        if rest_gps == 0:
            F_ownship, Q_ownship = navsys.step_filter(ownship.state[:,k], k_imu, k_gps)
            est_quat, est_vel, est_pos, _, _ = navsys.get_strapdown_estimate(k_imu)
            true_pos_err = ownship.state[:3,k]-est_pos
            true_vel_err = ownship.state_diff[:3,k]-est_vel
            true_quat = conv.euler_angles_to_quaternion(ownship.state[3:6,k])
            true_ang_err = conv.quat_mul(true_quat, conv.quat_conj(est_quat))[:3]*2
            est_ang_err = navsys.EKF.est_posterior[:3,k_gps]
            est_vel_err = navsys.EKF.est_posterior[3:6,k_gps]
            est_pos_err = navsys.EKF.est_posterior[6:9,k_gps]
            cov_ang = navsys.EKF.cov_posterior[:3,:3,k_gps]
            cov_vel = navsys.EKF.cov_posterior[3:6,3:6,k_gps]
            cov_pos = navsys.EKF.cov_posterior[6:9,6:9,k_gps]
            #cov_acc = navsys.EKF.cov_posterior[9:12,9:12,k_gps]
            #cov_gyr = navsys.EKF.cov_posterior[12:,12:,k_gps]
            cov_all = navsys.EKF.cov_posterior[:9,:9,k_gps]
            nav_errs[0].update_vals(true_ang_err, est_ang_err, cov_ang, k_gps, n_mc)
            nav_errs[1].update_vals(true_vel_err, est_vel_err, cov_vel, k_gps, n_mc)
            nav_errs[2].update_vals(true_pos_err, est_pos_err, cov_pos, k_gps, n_mc)
            true_all = np.hstack((true_ang_err, true_vel_err, true_pos_err))
            est_all = np.hstack((est_ang_err, est_vel_err, est_pos_err))
            nav_errs[3].update_vals(true_all, est_all, cov_all, k_gps, n_mc)
        k_radar, rest_radar = int(np.floor(k/M_radar)), np.mod(k, M_radar)
        if rest_radar == 0:
            ownship_radar.generate_measurement((target.state[:,k], ownship.state[:,k]), k_radar)
            nav_quat, _, nav_pos, _, _ = navsys.get_strapdown_estimate(k_imu)
            nav_eul = conv.quaternion_to_euler_angles(nav_quat)
            navigation_pose = np.hstack((nav_pos[0:2], nav_eul[2]))
            ground_truth_pose = np.hstack((ownship.state[0:2,k], ownship.state[5,k]))
            navigation_cov = np.squeeze(navsys.EKF.cov_posterior[[[[6],[7],[2]]],[6,7,2], k_gps])
            #navigation_pose = ground_truth_pose+multivariate_normal(cov=navigation_cov).rvs()
            true_track_state = np.hstack((target.state[0,k], target.state_diff[0,k], target.state[1,k], target.state_diff[1,k]))
            # The poses and covs match the order in tracker defined on the top
            poses = (ground_truth_pose, navigation_pose, navigation_pose, navigation_pose)
            covs = (np.zeros((3,3)), np.zeros((3,3)), navsys.EKF.cov_posterior[:,:,k_gps], navigation_cov)
            est_quat, est_vel, est_pos, _, _ = navsys.get_strapdown_estimate(k_imu)
            exargs = ({}, {}, {'F' : F_ownship, 'Q' : Q_ownship, 'full_state' : np.hstack((est_quat, est_vel, est_pos))}, {})
            # Step the tracking filters
            for j, tracker in enumerate(trackers):
                tracker.step(ownship_radar.data[:,k_radar], poses[j], covs[j], k_radar, **exargs[j])
                errs[j].update_vals(true_track_state, tracker.est_posterior[:4,k_radar], tracker.cov_posterior[:4,:4,k_radar], k_radar, n_mc)
    print str(datetime.datetime.now()), n_mc+1, '/', N_MC

mark_every = 10
marker_size=5
ground_truth_style = {'label' : 'Ground truth', 'marker' : 'o', 'markevery' : mark_every, 'ms' : marker_size}
schmidt_style = {'label' : 'Schmidt', 'marker' : 's', 'markevery' : mark_every, 'ms' : marker_size}
uncomp_style = {'label' : 'Uncompensated', 'marker' : 'v', 'markevery' : mark_every, 'ms' : marker_size}
conv_style = {'label' : 'Converted measurement', 'marker' : 'h', 'markevery' : mark_every, 'ms' : marker_size}
hea = (ground_truth_style, uncomp_style, schmidt_style, conv_style)
with plt.style.context(('filter')):
    consistency_fig, consistency_ax = plt.subplots(3,1)
    for err in errs:
        err.plot_errors(consistency_ax[0], consistency_ax[1], consistency_ax[2])
    consistency_ax[0].set_title('NEES for ' + str(N_MC) + ' Monte Carlo runs')
    consistency_ax[1].set_title('Position RMSE for ' + str(N_MC) + ' Monte Carlo runs')
    consistency_ax[2].set_title('Velocity RMSE for ' + str(N_MC) +' Monte Carlo runs')
    [consistency_ax[j].legend() for j in range(3)]

    nav_consistency_fig, nav_consistency_ax = plt.subplots(3,1)
    for err in nav_errs:
        err.plot_errors(nav_consistency_ax[0], nav_consistency_ax[1], nav_consistency_ax[2])
    nav_consistency_ax[0].set_title('NEES for ' + str(N_MC) + ' Monte Carlo runs')
    nav_consistency_ax[1].set_title('Position RMSE for ' + str(N_MC) + ' Monte Carlo runs')
    nav_consistency_ax[2].set_title('Velocity RMSE for ' + str(N_MC) +' Monte Carlo runs')
    [nav_consistency_ax[j].legend() for j in range(3)]
    xy_fig, xy_ax = plt.subplots(1,1)
    #xy_ax = xy_ax.reshape(4)
    vel_fig, vel_ax = plt.subplots(2,1)
    velerr_fig, velerr_ax = plt.subplots(2,1)
    K_fig, K_ax = plt.subplots(2,1)
    for j in range(len(arg)):
        viz.target_xy(target, trackers[j], xy_ax, hea[j])
        #viz.plot_xy_pos((ownship, ), xy_ax[j], own_arg)
        xy_ax.set_xlim((300,1600))
        xy_ax.set_ylim((2800,4000))
        xy_ax.set_xlabel('East')
        xy_ax.set_ylabel('North')
        xy_ax.legend(loc=2)
        viz.target_velocity(target, trackers[j], vel_ax, arg[j])
        viz.target_velocity_error(target, trackers[j], velerr_ax, arg[j])
        K_ax[0].plot(radar_time, trackers[j].K_gains[0,:], **arg[j])
        K_ax[1].plot(radar_time, trackers[j].K_gains[1,:], **arg[j])
        K_ax[0].legend()

    xy_2, xy_2ax = plt.subplots(1,1)
    xy_2ax.plot(ownship.state[1,:], ownship.state[0,:], 'k--', label='Ownship')
    xy_2ax.plot(target.state[1,:], target.state[0,:], 'k', label='Target')
    xy_2ax.set_xlabel('East')
    xy_2ax.set_ylabel('North')
    xy_2ax.legend(loc=2)
    #plt.figure(xy_fig.number)
    head_fig, head_ax = plt.subplots(1,1)
    est_quat = navsys.strapdown.data[navsys.strapdown.orient, :]
    est_eul = conv.quaternion_to_euler_angles(est_quat.T).T
    head_ax.plot(ownship.time, ownship.state[5,:], 'k', label='True heading')
    head_ax.errorbar(navsys.EKF.time, est_eul[2,0::20], yerr=3*np.sqrt(np.squeeze(navsys.EKF.cov_posterior[2,2,:])),errorevery=10)
    plt.savefig('hea.pdf')
    plt.show()

