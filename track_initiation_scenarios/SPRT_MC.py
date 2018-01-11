import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import Greys, Greens
import matplotlib.animation as manimation
from ipdb import set_trace

import autoseapy.simulation as autosim
import autoseapy.tracking as autotrack
import autoseapy.track_initiation as autoinit
import autoseapy.visualization as autovis
from autoseapy.sylte import load_pkl
import filtersim.track_initiation as filterinit

confirmed_color = np.array([214, 39, 40])/255.
ais_color = np.array([23, 190, 207])/255.
own_color = np.array([255, 127, 14])/255.

q = 0.05**2
r = 6**2
R = r*np.identity(2)
H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
dt = 2.5
v_max = 15
time = np.arange(0,25, dt)
F, Q = autotrack.DWNAModel.model(dt, q)
# Set up true targets
x0 = np.array([-100, 10, 0, 0])
radar_range = 150
P_D = 0.9
N_MC = 20
# Set up track initiation
gate = autotrack.TrackGate(0.99, v_max)
clutter_density = 2e-5
birth_density = 1e-4
np.random.seed(seed=250190)
radar = autosim.SquareRadar(radar_range, clutter_density, P_D, R)
target_model = autotrack.DWNAModel(q)
PDAF_tracker = autotrack.PDAFTracker(0.9, target_model, gate)
SPRT = autoinit.SequentialRatioTest(0.01, 0.99, clutter_density, birth_density, v_max, P_D, target_model)
track_termination = autotrack.TrackTerminator(5)
SPRT_manager = autotrack.Manager(PDAF_tracker, SPRT, track_termination)
N_total_targets = 0
N_total_tracks = 0
N_true_tracks = 0
N_false_tracks = 0
true_target_state = np.zeros((4, len(time)))
true_target = []
def get_rmse(true_track, est_track):
    timestamps = set([estimate.timestamp for estimate in est_track])
    partial_true_track = [state for state in true_track if state.timestamp in timestamps]
    est_states = np.array([est.est_posterior for est in est_track])
    true_states = np.array([state.est_posterior for state in partial_true_track])
    x_pos_err = true_states[:,0]-est_states[:,0]
    y_pos_err = true_states[:,2]-est_states[:,2]
    pos_rmse = np.sqrt(x_pos_err**2+y_pos_err**2)
    return [est.timestamp for est in est_track], pos_rmse

def is_true_track(true_track, est_track):
    # Assume all the estimates in est_track is synchronized with true_track, such that the timestamps in est_track also is in true_track
    _, pos_rmse = get_rmse(true_track, est_track)
    return np.all(pos_rmse < 50)

for k, t in enumerate(time):
    if k is 0:
        true_target_state[:,k] = x0
    else:
        true_target_state[:,k] = F.dot(true_target_state[:,k-1])
    true_target.append(autotrack.Estimate(t, true_target_state[:,k], None, True, 'true_target'))

RMSE_fig, RMSE_ax = plt.subplots()
for n_mc in range(N_MC):
    N_true_this = 0
    N_false_this = 0
    SPRT_manager.reset()
    measurements_all = []
    for k, timestamp in enumerate(time):
        measurements_target = radar.generate_target_measurements([H.dot(true_target_state[:,k])], timestamp)
        for est, measurement in zip(true_target, measurements_target):
            est.measurements = [measurement]
        measurements_clutter = radar.generate_clutter_measurements(timestamp)
        measurements_all.append(measurements_target+measurements_clutter)
    filterinit.run_SPRT_manager(SPRT_manager, measurements_all, time)
    N_total_targets += 1
    N_total_tracks += len(SPRT_manager.track_file)
    for track in SPRT_manager.track_file.values():
        time, rmse = get_rmse(true_target, track)
        RMSE_ax.plot(time, rmse)
        if is_true_track(true_target, track):
            N_true_tracks += 1
            N_true_this += 1
        else:
            N_false_tracks += 1
            N_false_this += 1
    #n_total_targets += len(true_target)
    if N_false_this > 0 or N_true_this is not 1:
        track_fig, track_ax = autovis.plot_measurements(measurements_all, cmap=Greys)
        #true_target_state = np.array([est.est_posterior for est in true_target])
        track_ax.plot(true_target_state[2,:], true_target_state[0,:], 'k')
        track_ax.plot(true_target_state[2,0], true_target_state[0,0], 'ko')
        autovis.plot_track_pos(SPRT_manager.track_file, track_ax, confirmed_color)
        #[track_ax.plot(est.est_posterior[2], est.est_posterior[0], 'r*') for est in initial_estimates]
        meas_fig, meas_ax = autovis.plot_measurements(measurements_all, cmap=Greys)
        for ax in [track_ax, meas_ax]:
            ax.set_aspect('equal')
            ax.set_xlim(-150, 150)
            ax.set_ylim(-150, 150)
            ax.set_title('MC-run number {}, N_FT={},N_TT={}'.format(n_mc, N_false_this, N_true_this))
print "P_DT={}".format(float(N_true_tracks)/N_total_targets)
print "P_FT={}".format(float(N_false_tracks)/N_total_tracks)
plt.show()
