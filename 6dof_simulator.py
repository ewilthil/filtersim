import numpy as np
import matplotlib.pyplot as plt
import navigation as nav
import visualization as viz
import tracking as track
import datetime
from base_classes import Model, Sensor, ErrorStats, radar_measurement
from autopy.sylte import load_pkl
from autopy.conversion import quaternion_to_euler_angles
plt.close('all')

N_MC = 1
target = load_pkl('target_constant_turn.pkl')
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
perfect_plot_args = {'color' : 'b', 'label' : 'Perfect pose'}
uncomp_plot_args = {'color' : 'r', 'label' : 'Uncompensated navigation'}
perfect_pose_errs = ErrorStats(radar_time, N_MC, perfect_plot_args)
uncompensated_errs = ErrorStats(radar_time, N_MC, uncomp_plot_args)
comp_plot_args = {'color' : 'g', 'label' : 'Compensated navigation'}
compensated_errs = ErrorStats(radar_time, N_MC, comp_plot_args)
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
    track_init_cov = np.diag((10**2, 2**2, 10**2, 2**2, (0.1*np.pi/180)**2))
    dwna_cov = 2**2
    ct_cov = 0.08**2
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

    names = (DWNA, CT_known, CT_known)
    pi = np.array([[0.8, 0.1, 0.1],[0.1, 0.9, 0],[0.1, 0, 0.9]])
    prob_init = np.array([0.8, 0.1, 0.1])
    sigmas = tuple_elements(names, sigma_dict)
    imm_init_pos = tuple_elements(names, state_init_dict)
    imm_init_cov = tuple_elements(names, cov_init_dict)
    extra_args = ({}, {'omega' : np.deg2rad(-1.5)},{'omega' : np.deg2rad(1.5)})
    perfect_pose_imm = track.IMM(pi, radar_time, sigmas, imm_init_pos, imm_init_cov, prob_init, cov_radar, names, extra_args)
    uncompensated_imm = track.IMM(pi, radar_time, sigmas, imm_init_pos, imm_init_cov, prob_init, cov_radar, names, extra_args)
    compensated_imm = track.IMM(pi, radar_time, sigmas, imm_init_pos, imm_init_cov, prob_init, cov_radar, names, extra_args)

    schmidt = track.DWNA_schmidt(radar_time, sigma_dict[DWNA], cov_radar, state_init_dict[DWNA], cov_init_dict[DWNA])
    no_schmidt = track.DWNA_schmidt(radar_time, sigma_dict[DWNA], cov_radar, state_init_dict[DWNA], cov_init_dict[DWNA])
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
            navigation_cov = np.squeeze(navsys.EKF.cov_posterior[[[[6],[7],[2]]],[6,7,2], k_gps])
            perfect_pose_imm.step(ownship_radar.data[:,k_radar], k_radar, perfect_pose, np.zeros((3,3)))
            uncompensated_imm.step(ownship_radar.data[:,k_radar], k_radar, navigation_pose, np.zeros((3,3)))
            compensated_imm.step(ownship_radar.data[:,k_radar], k_radar, navigation_pose, navigation_cov)

            schmidt.step(ownship_radar.data[:,k_radar], navigation_pose, navigation_cov, k_radar)

            true_target_state = np.array([target.state[0,k], target.state_diff[0,k], target.state[1,k], target.state_diff[1,k], target.state_diff[5,k]])
            perfect_pose_errs.update_vals(true_target_state, perfect_pose_imm.est_posterior[:,k_radar], perfect_pose_imm.cov_posterior[:,:,k_radar], k_radar, n_mc)
            uncompensated_errs.update_vals(true_target_state, uncompensated_imm.est_posterior[:,k_radar], uncompensated_imm.cov_posterior[:,:,k_radar], k_radar, n_mc)
            compensated_errs.update_vals(true_target_state, compensated_imm.est_posterior[:,k_radar], compensated_imm.cov_posterior[:,:,k_radar], k_radar, n_mc)

print str(datetime.datetime.now())
imms = (perfect_pose_imm, uncompensated_imm, compensated_imm)
errs = (perfect_pose_errs, uncompensated_errs, compensated_errs)
args = (perfect_plot_args, uncomp_plot_args, comp_plot_args)
# Consistency results
consistency_fig, consistency_ax = plt.subplots(3,1)
for err in errs:
    err.plot_errors(consistency_ax[0], consistency_ax[1], consistency_ax[2])
consistency_ax[0].set_title('NEES for ' + str(N_MC) + ' Monte Carlo runs')
consistency_ax[1].set_title('Position RMSE for ' + str(N_MC) + ' Monte Carlo runs')
consistency_ax[2].set_title('Velocity RMSE for ' + str(N_MC) +' Monte Carlo runs')
[consistency_ax[j].legend() for j in range(3)]

# Sample trajectories from latest MC run
xy_fig, xy_ax = plt.subplots(1,3)
vel_fig, vel_ax = plt.subplots(2,1)
ang_fig, ang_ax = plt.subplots(2,1)
likelihood_fig, likelihood_ax = plt.subplots(3,1)
own_arg = {'color' : 'k', 'label' : 'Ownship pose'}
for j, imm in enumerate(imms):
    viz.target_xy(target, imm, xy_ax[j], args[j])
    viz.target_xy(ownship, imm, xy_ax[j], own_arg)
    xy_ax[j].set_title(args[j]['label'])
    viz.target_velocity(target, imm, vel_ax, args[j])
    viz.target_angular_rate(target, imm, ang_ax[0], args[j])
    viz.DWNA_probability(imm, ang_ax[1], args[j])
    viz.likelihoods(imms[j], likelihood_ax[j], args[j])
plt.show()
