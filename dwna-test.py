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
plt.close('all')

N_MC = 50
target = load_pkl('target_straight.pkl')
ownship = load_pkl('ownship_turn.pkl')
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
cov_radar = np.diag((20**2, (0.5*np.pi/180)**2))
track_cov_init = np.diag((5**2, 1**2, 5**2, 1**2))
acc_cov = 1**2
Q_sub = np.array([[radar_dt**4/4, radar_dt**3/2],[radar_dt**3/2, radar_dt**2]])
Q_DWNA = acc_cov*block_diag(Q_sub, Q_sub)

#Plot arguments
own_arg = {'color' : 'k', 'label' : 'Ownship'}
schmidt_args = {'color' : 'g', 'label': 'Schmidt'}
ground_truth_args = {'color' : 'b', 'label' : 'Ground truth pose'}
uncomp_args = {'color' : 'r', 'label' : 'Uncompensated'}
# Error statistics
schmidt_errs = ErrorStats(radar_time, N_MC, schmidt_args)
ground_truth_errs = ErrorStats(radar_time, N_MC, ground_truth_args)
uncomp_errs = ErrorStats(radar_time, N_MC, uncomp_args)
# Group up
arg = (ground_truth_args, uncomp_args, schmidt_args)
errs = (ground_truth_errs, uncomp_errs, schmidt_errs)
print str(datetime.datetime.now())
for n_mc in range(N_MC):
    # (re)initialize instances
    navsys = nav.NavigationSystem(q0, v0, p0, imu_time, gps_time)
    ownship_radar = Sensor(radar_measurement, np.zeros(2), cov_radar, radar_time)
    schmidt = track.DWNA_schmidt(radar_time, Q_DWNA, cov_radar, track_state_init, track_cov_init)
    ground_truth = track.DWNA_schmidt(radar_time, Q_DWNA, cov_radar, track_state_init, track_cov_init)
    uncompensated_tracker = track.DWNA_schmidt(radar_time, Q_DWNA, cov_radar, track_state_init, track_cov_init)
    trackers = (ground_truth, uncompensated_tracker, schmidt)

    
    # run the navigation and tracking filters
    for k, t in enumerate(time):
        k_imu, rest_imu = int(np.floor(k/M_imu)), np.mod(k,M_imu)
        if rest_imu == 0:
            navsys.step_strapdown(ownship.state[:,k], ownship.state_diff[:,k], k_imu)
        k_gps, rest_gps = int(np.floor(k/M_gps)), np.mod(k,M_gps)
        if rest_gps == 0:
            navsys.step_filter(ownship.state[:,k], k_imu, k_gps)
        k_radar, rest_radar = int(np.floor(k/M_radar)), np.mod(k, M_radar)
        if rest_radar == 0:
            ownship_radar.generate_measurement((target.state[:,k], ownship.state[:,k]), k_radar)
            nav_quat, _, nav_pos, _, _ = navsys.get_strapdown_estimate(k_imu)
            nav_eul = conv.quaternion_to_euler_angles(nav_quat)
            navigation_pose = np.hstack((nav_pos[0:2], nav_eul[2]))
            ground_truth_pose = np.hstack((ownship.state[0:2,k], ownship.state[5,k]))
            navigation_cov = np.squeeze(navsys.EKF.cov_posterior[[[[6],[7],[2]]],[6,7,2], k_gps])
            true_track_state = np.hstack((target.state[0,k], target.state_diff[0,k], target.state[1,k], target.state_diff[1,k]))
            # The poses and covs match the order in tracker defined on the top
            poses = (ground_truth_pose, navigation_pose, navigation_pose)
            covs = (np.zeros((3,3)), np.zeros((3,3)), navigation_cov)
            # Step the tracking filters
            for j, tracker in enumerate(trackers):
                tracker.step(ownship_radar.data[:,k_radar], poses[j], covs[j], k_radar)
                errs[j].update_vals(true_track_state, tracker.est_posterior[:,k_radar], tracker.cov_posterior[:,:,k_radar], k_radar, n_mc)
    print str(datetime.datetime.now())


consistency_fig, consistency_ax = plt.subplots(3,1)
for err in errs:
    err.plot_errors(consistency_ax[0], consistency_ax[1], consistency_ax[2])
consistency_ax[0].set_title('NEES for ' + str(N_MC) + ' Monte Carlo runs')
consistency_ax[1].set_title('Position RMSE for ' + str(N_MC) + ' Monte Carlo runs')
consistency_ax[2].set_title('Velocity RMSE for ' + str(N_MC) +' Monte Carlo runs')
[consistency_ax[j].legend() for j in range(3)]

xy_fig, xy_ax = plt.subplots(1,3)
vel_fig, vel_ax = plt.subplots(2,1)
for j in range(len(arg)):
    viz.target_xy(target, trackers[j], xy_ax[j], arg[j])
    viz.plot_xy_pos((ownship, ), xy_ax[j], own_arg)
    viz.target_velocity(target, trackers[j], vel_ax, arg[j])
plt.show()
