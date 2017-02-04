import numpy as np
from filtersim import shipmodels, tracking

def check_trajectory(trajectory, limits):
    outside = False
    for N in trajectory[0,:]:
        if N > limits[1] or N < limits[0]:
            outside = True
    for E in trajectory[2,:]:
        if E > limits[1] or E < limits[0]:
            outside = True
    return outside

def generate_trajectory(T=30, dt=1, alpha=1, tracking_limits=np.array([-250, 250]),sigma=0.3):
    time = np.arange(0, T+dt, dt)
    v_ref = np.zeros(2)
    OoB = True
    motion_model_params = {'sigmas' : sigma*np.ones(3)}
    while OoB:
        pos0 = np.random.uniform(low=tracking_limits[0], high=tracking_limits[1],size=2)
        V_N = alpha*np.random.uniform(low=(tracking_limits[0]-pos0[0])/(1.*T), high=(tracking_limits[1]-pos0[0])/(1.*T))
        V_E = alpha*np.random.uniform(low=(tracking_limits[0]-pos0[1])/(1.*T), high=(tracking_limits[1]-pos0[1])/(1.*T))
        x0 = np.array([pos0[0], V_N, pos0[1], V_E])
        model = shipmodels.DiscreteWNA(dt, 2, motion_model_params)
        target = shipmodels.TargetShip(time, model, x0)
        target.generate_trajectory(np.zeros(2))
        OoB = check_trajectory(target.states, tracking_limits)
    return time, target.states, model

def generate_measurements(time, trajectory, measurement_model):
    measurements_all = []
    for t, x in zip(time, trajectory.T):
        measurements = measurement_model.generate_measurements([x], t)
        measurements_all.append(measurements)
    return measurements_all

def generate_scenario(P_D=1, clutter_density=10./(500**2), sigma=0.3):
    time, traj, motion_model = generate_trajectory(sigma=sigma)
    R = 7**2*np.identity(2)
    measurement_model = tracking.MeasurementModel(
            target_cov=R,
            clutter_density=clutter_density,
            x_lims=np.array([-250, 250]),
            P_D=P_D)
    measurements = generate_measurements(time, traj, measurement_model)
    return time, traj, measurements, motion_model, measurement_model

def visualize_mn_dict(est_dict, pos_ax, vel_axes=None):
    for track_id, est_list in est_dict.items():
        prelim_ests = []
        conf_ests = []
        for est, status in est_list:
            if status == 'CONFIRMED':
                conf_ests.append(est)
            elif status == 'PRELIMINARY':
                prelim_ests.append(est)
        if len(conf_ests) > 0:
            prelim_track = np.zeros((4, len(prelim_ests)))
            prelim_time = np.zeros(len(prelim_ests))
            for idx, est in enumerate(prelim_ests):
                prelim_track[:, idx] = est.est_posterior
                prelim_time[idx] = est.timestamp
            pos_ax.plot(prelim_track[2,:], prelim_track[0,:], 'k--')
            conf_track = np.zeros((4, len(conf_ests)+1))
            conf_time = np.zeros(len(conf_ests)+1)
            for idx, est in enumerate(conf_ests):
                conf_track[:, idx+1] = est.est_posterior
                conf_time[idx+1] = est.timestamp
            conf_track[:,0] = prelim_track[:,-1]
            conf_time[0] = prelim_time[-1]
            pos_ax.plot(conf_track[2,:], conf_track[0,:], 'k')
            if vel_axes is not None:
                l = vel_axes[0].plot(prelim_time, prelim_track[1,:])
                vel_axes[0].plot(conf_time, conf_track[1, :],color=l[0].get_color())
                l = vel_axes[1].plot(prelim_time, prelim_track[3,:])
                vel_axes[1].plot(conf_time, conf_track[3, :],color=l[0].get_color())
