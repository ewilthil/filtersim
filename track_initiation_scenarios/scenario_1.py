import numpy as np

import autoseapy.simulation as autosim
import autoseapy.clutter_maps as autoclutter
import autoseapy.tracking as autotrack  
import autoseapy.visualization as autovis

def setup_radar():
    radar_range = 500
    clutter_density = 5e-5
    high_clutter_density = 2e-4
    P_D = 0.9
    R = 7.**2*np.identity(2)
    radar = autosim.SquareRadar(radar_range, clutter_density, detection_probability=P_D, measurement_covariance=R)
    clutter_map = autoclutter.GeometricClutterMap(-radar_range, radar_range, -radar_range, radar_range, clutter_density)
    high_density_region = autoclutter.SquareRegion(high_clutter_density, [[-radar_range, -radar_range], [0, -radar_range], [0, 0], [-radar_range, 0]])
    clutter_map.add_region(high_density_region)
    radar.add_clutter_map(clutter_map)
    return radar

def target_positions(t):
    initial_states = [np.array([250, -7, -450, 0]), np.array([-250, 0, -250, 5])]
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    states = []
    for init in initial_states:
        F, _ = autotrack.DWNA_model(t)
        states.append(H.dot(F.dot(init)))
    return states

dt = 2.4
N_scans = 40
time_vector = np.arange(0, dt*N_scans, dt)
radar = setup_radar()
measurements_all = []
for n_t, t in enumerate(time_vector):
    measurements = radar.generate_measurements(target_positions(t), t)
    measurements_all.append(measurements)

fig, ax = autovis.plot_measurements(measurements_all)
fig.show()
