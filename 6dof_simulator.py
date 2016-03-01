import numpy as np
import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion
import navigation as nav
import visualization as viz
import tracking as track
import datetime
from base_classes import Model, Sensor
from scipy.stats import chi2
plt.close('all')

gravity_n = np.array([0, 0, 9.81])

dt, Tend = 0.01, 300
time = np.arange(0, Tend+dt, dt)
D = -np.diag((0.5, 1, 10, 10, 10, 1))
T = -np.diag((30, 1, 30, 10, 10, 60))
Q = np.diag((1e-1, 1, 1e-1, 1, 1, 1e-4))

initial_target_heading = 225*np.pi/180
final_target_heading = np.pi
target_velocity = 12
target_init = np.zeros(18)
target_init[0] = 4000
target_init[1] = 1600
target_init[6] = target_velocity
target_init[5] = initial_target_heading

ownship_heading = 0
ownship_velocity = 10
ownship_init = np.zeros(18)
ownship_init[6] = ownship_velocity

target = Model(D, T, Q, target_init, time)
ownship = Model(D, T, Q, ownship_init, time)

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

M_imu = 1
M_gps = 20
M_radar = 100
N_MC = 1
imu_time = np.arange(0, Tend+M_imu*dt, M_imu*dt)
gps_time = np.arange(0, Tend+M_gps*dt, M_gps*dt)
radar_time = np.arange(0, Tend+M_radar*dt, M_radar*dt)
q0 = np.array([0,0,0,1])
v0 = np.array([10,0,0])
p0 = np.array([0,0,0])
navsys = nav.NavigationSystem(q0, v0, p0, imu_time, gps_time)
cov_radar = np.diag((35**2, (1*np.pi/180)**2))
ownship_radar = Sensor(radar_measurement, np.zeros(2), cov_radar, radar_time)
ground_radar = Sensor(radar_measurement, np.zeros(2), cov_radar, radar_time)
stationary_dwna = track.DWNA_filter(radar_time, np.diag((0.1**2,0.1**2)), cov_radar, np.hstack((target.state[0:2,0], target.NED_vel(0)[0:2]))[[0,2,1,3]], np.diag((20**2, 5**2, 20**2, 5**2)))
# Main loop
print str(datetime.datetime.now())
for k, t in enumerate(time):
    # Set reference
    if t < 150:
        target_ref = np.array([target_velocity, initial_target_heading])
    else:
        target_ref = np.array([target_velocity, final_target_heading])
    # Propagate state
    target.step(k, target_ref)
    ownship.step(k, np.array([ownship_velocity, ownship_heading]))
    # Generate sensor data
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
        stationary_dwna.step(ground_radar.data[:,k_radar],k_radar)
print str(datetime.datetime.now())
viz.plot_pos_err(ownship, navsys)
viz.plot_vel_err(ownship, navsys,boxplot=False)
_, ax = viz.plot_xy_pos((ownship, target))
viz.plot_xy_trajectory(ax, stationary_dwna.est_prior[2,:], stationary_dwna.est_prior[0,:], 'r--')
viz.plot_xy_trajectory(ax, stationary_dwna.est_posterior[2,:], stationary_dwna.est_posterior[0,:], 'g--')
xy_measurements = [polar_to_cartesian(ground_radar.data[:,k]) for k in range(len(radar_time))]
xy_measurements = np.vstack(xy_measurements).T
viz.plot_xy_trajectory(ax, xy_measurements[1,:], xy_measurements[0,:], 'b*')
ax.set_aspect('equal')
vel_fig, vel_ax = plt.subplots(2,1)
vel_ax[0].errorbar(stationary_dwna.time, stationary_dwna.est_prior[1,:], 3*np.sqrt(np.squeeze(stationary_dwna.cov_prior[1,1,:])))
vel_ax[0].errorbar(stationary_dwna.time, stationary_dwna.est_posterior[1,:], 3*np.sqrt(np.squeeze(stationary_dwna.cov_posterior[1,1,:])))
vel_ax[0].plot(target.time, target.state_diff[0,:])
vel_ax[1].errorbar(stationary_dwna.time, stationary_dwna.est_prior[3,:], 3*np.sqrt(np.squeeze(stationary_dwna.cov_prior[3,3,:])))
vel_ax[1].errorbar(stationary_dwna.time, stationary_dwna.est_posterior[3,:], 3*np.sqrt(np.squeeze(stationary_dwna.cov_posterior[3,3,:])))
vel_ax[1].plot(target.time, target.state_diff[1,:])

NEES = np.zeros_like(stationary_dwna.time)
for k,_ in enumerate(NEES):
    print target.time[k*M_radar], stationary_dwna.time[k]
    true_vel = target.state_diff[0:2,k*M_radar]
    est_vel = stationary_dwna.est_posterior[[1,3],k]
    cov_vel = stationary_dwna.cov_posterior[[[1],[3]],[1,3],k]
    NEES[k] = np.dot(np.dot(true_vel-est_vel, np.linalg.inv(cov_vel)), true_vel-est_vel)
UB = chi2(df=2*N_MC).ppf(0.975)*np.ones_like(NEES)/N_MC
LB = chi2(df=2*N_MC).ppf(0.025)*np.ones_like(NEES)/N_MC
time_vel = stationary_dwna.time
const_fig, const_ax = plt.subplots(1,1)
[const_ax.plot(time_vel, elem) for elem in [NEES, UB, LB]]
plt.show()
