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
true_target_state = np.zeros((4, len(time)))
x0 = np.array([-100, 10, 0, 0])
radar_range = 150
P_D = 0.9
N_MC = 2
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
N_total_targets = 0
N_true_tracks = 0
N_total_tracks = 0
N_false_tracks = 0
for n_mc in range(N_MC):
    track_manager = autotrack.Manager(PDAF_tracker, SPRT, track_termination)
    initial_estimates = []
    measurements_all = []
    for k, timestamp in enumerate(time):
        if k is 0:
            true_target_state[:,k] = x0
        else:
            true_target_state[:,k] = F.dot(true_target_state[:,k-1])
        measurements = radar.generate_measurements([H.dot(true_target_state[:,k])], timestamp)
        measurements_all.append(measurements)
        old_estimates, new_tracks = track_manager.step(measurements, timestamp)
        if len(new_tracks) > 0:
            for ests in new_tracks:
                initial_estimates.append(ests[-1])
                print "target confirmed after {} steps".format(k)
    # Analyze the outcome
    N_total_targets += 1
    track_fig, track_ax = autovis.plot_measurements(measurements_all, cmap=Greys)
    track_ax.plot(true_target_state[2,:], true_target_state[0,:], 'k')
    track_ax.plot(true_target_state[2,0], true_target_state[0,0], 'ko')
    autovis.plot_track_pos(track_manager.track_file, track_ax, confirmed_color)
    [track_ax.plot(est.est_posterior[2], est.est_posterior[0], 'r*') for est in initial_estimates]
    #scat_fig, scat_ax = plt.subplots()
    #scat_ax.scatter(np.array([d[0] for d in time_data_all]), np.array([d[1] for d in time_data_all]))
    meas_fig, meas_ax = autovis.plot_measurements(measurements_all, cmap=Greys)
    for ax in [track_ax, meas_ax]:
        ax.set_aspect('equal')
        ax.set_xlim(-150, 150)
        ax.set_ylim(-150, 150)
plt.show()
