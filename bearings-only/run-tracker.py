import numpy as np
import ipdb
import matplotlib.pyplot as plt
from autopy.sylte import load_pkl
from filtersim.base_classes import Sensor, Model, radar_measurement
from bearings_only_tracking import BearingsOnlyEKF, BearingsOnlyMP, BearingsOnlyPF
N_mc = 1
target = load_pkl('target_traj_long.pkl')
ownship = load_pkl('ownship_traj_long.pkl')
time = ownship.time
Tend = time[-1]
dt = time[1]-time[0]
M_imu = 1
M_gps = 20
M_radar = 100
imu_time = np.arange(0, Tend+M_imu*dt, M_imu*dt)
gps_time = np.arange(0, Tend+M_gps*dt, M_gps*dt)
radar_time = np.arange(0, Tend+M_radar*dt, M_radar*dt)
cov_radar = np.diag((1**2, np.deg2rad(0.7)**2))
true_range = np.zeros_like(radar_time)
true_tracking_state = np.zeros((4, len(radar_time)))
true_polar_state = np.zeros((4, len(radar_time)))
true_conv_state = np.zeros((4, len(radar_time)))
true_measurement = np.zeros_like(radar_time)
for n_mc in range(N_mc):
    ownship_radar = Sensor(radar_measurement, np.zeros(2), cov_radar, radar_time)
    true_ownship_tracking_init = np.hstack((ownship.state[ownship.pos_x,0], ownship.state_diff[ownship.vel_x,0], ownship.state[ownship.pos_y, 0], ownship.state_diff[ownship.vel_y, 0]))
    tracker = BearingsOnlyEKF(radar_time, 0.1**2, cov_radar[1,1], np.array([5000, 0, 5000, 0]), np.diag((5**2, 1**2, 5**2, 1**2)), true_ownship_tracking_init)
    trackerMP = BearingsOnlyMP(radar_time, 0.1**2, cov_radar[1,1], np.array([5000, 0, 5000, 0]), np.diag((5**2, 1**2, 5**2, 1**2)), true_ownship_tracking_init)
    trackerPF = BearingsOnlyPF(radar_time, 0.1**2, cov_radar[1,1], np.array([5000, 0, 5000, 0]), np.diag((5**2, 1**2, 5**2, 1**2)), true_ownship_tracking_init)
    c2p = trackerMP.cartesian_to_polar
    p2c = trackerMP.polar_to_cartesian
    for k, t in enumerate(time):
        k_radar, rest_radar = int(np.floor(k/M_radar)), np.mod(k, M_radar)
        if rest_radar == 0:
            true_ownship_tracking_state = np.hstack((ownship.state[ownship.pos_x,k], ownship.state_diff[ownship.vel_x,k], ownship.state[ownship.pos_y, k], ownship.state_diff[ownship.vel_y, k]))
            true_target_tracking_state = np.hstack((target.state[target.pos_x,k], target.state_diff[target.vel_x,k], target.state[target.pos_y, k], target.state_diff[target.vel_y, k]))
            true_tracking_state[:,k_radar] = true_target_tracking_state-true_ownship_tracking_state
            true_polar_state[:,k_radar] = c2p(true_tracking_state[:,k_radar])
            true_conv_state[:,k_radar] = p2c(true_polar_state[:,k_radar])
            true_measurement[k_radar] = np.arctan2(true_tracking_state[2,k_radar], true_tracking_state[0,k_radar])
            ownship_pose  = np.hstack((ownship.state[ownship.pos_x, k], ownship.state[ownship.pos_y, k], ownship.state[ownship.psi, k]))
            # Step tracking
            ownship_radar.generate_measurement((target.state[:,k], ownship.state[:,k]), k_radar)
            tracker.step(ownship_radar.data[1,k_radar]+ownship.state[ownship.psi,k], true_ownship_tracking_state, ownship_pose, k_radar)
            trackerMP.step(ownship_radar.data[1,k_radar]+ownship.state[ownship.psi,k], true_ownship_tracking_state, ownship_pose, k_radar)
            trackerPF.step(ownship_radar.data[1,k_radar]+ownship.state[ownship.psi,k], true_ownship_tracking_state, ownship_pose, k_radar)


plt.figure()
plt.subplot(121)
plt.plot(tracker.local_est_posterior[0,:], tracker.local_est_posterior[2,:], '--*', label='EKF tracker')
plt.plot(trackerMP.local_est_posterior[0,:], trackerMP.local_est_posterior[2,:], ':s', label='MP tracker')
plt.plot(trackerPF.local_est_posterior[0,:], trackerPF.local_est_posterior[2,:], '-.d', label='PF tracker')
plt.plot(true_tracking_state[0,:], true_tracking_state[2,:], 'k',label='true')
plt.plot(true_conv_state[0,0], true_conv_state[2,0], 'o')
plt.title('Local position')
plt.legend()
plt.subplot(122)
plt.plot(tracker.global_est_posterior[0,:], tracker.global_est_posterior[2,:], label='EKF tracker')
plt.plot(trackerMP.global_est_posterior[0,:], trackerMP.global_est_posterior[2,:], label='MP tracker')
plt.plot(target.state[0,:], target.state[1,:], label='true')
plt.plot(ownship.state[ownship.pos_x,:], ownship.state[ownship.pos_y,:], label='ownship')
plt.legend()
plt.title('Global position')
plt.figure()
plt.subplot(211)
plt.plot(ownship_radar.time, ownship_radar.data[0,:])
plt.plot(radar_time, true_range)
plt.subplot(212)
plt.plot(ownship_radar.time, np.rad2deg(ownship_radar.data[1,:]))
plt.plot(radar_time, np.rad2deg(true_measurement))

plt.figure()
plt.plot(true_polar_state[0,:], 1./true_polar_state[2,:], '-*', label='true polar')
plt.plot(true_polar_state[0,0], 1./true_polar_state[2,0], 'o')
plt.plot(trackerMP.polar_est[0,:], 1./trackerMP.polar_est[2,:], '-*', label='Estimated polar')
plt.plot(trackerMP.polar_est[0,0], 1./trackerMP.polar_est[2,0], 'o')
