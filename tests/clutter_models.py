import numpy as np
from autoseapy import clutter_maps
import matplotlib.pyplot as plt
import autoseapy.visualization as autovis

N_timesteps = 25
grid_density = 100
true_clutter_map = clutter_maps.custom_map()
true_clutter_map = clutter_maps.nonuniform_musicki_map()
classic_clutter_map = clutter_maps.ClassicClutterMap.from_geometric_map(true_clutter_map, grid_density, N_timesteps)
spatial_clutter_map = clutter_maps.SpatialClutterMap.from_geometric_map(true_clutter_map, grid_density, N_timesteps)
temporal_clutter_map = clutter_maps.TemporalClutterMap.from_geometric_map(true_clutter_map, grid_density, N_timesteps)
measurements_all  = [true_clutter_map.generate_clutter(timestamp) for timestamp in range(N_timesteps)]
map_list = [classic_clutter_map, spatial_clutter_map, temporal_clutter_map]
for measurements in measurements_all:
    for clutter_map in map_list:
        clutter_map.update_estimate(measurements)

for idx, est_map in enumerate(map_list):
    fig, ax = plt.subplots(ncols=3)
    clutter_maps.plot_pair_of_clutter_map(true_clutter_map, est_map, ax)
    fig.savefig('clutter_map_%d.pdf' % (idx, ))

z_fig, z_ax = plt.subplots(ncols=2)
true_clutter_map.plot_density_map(z_ax[0])
autovis.plot_measurements(measurements_all, z_ax[1])
z_ax[1].set_aspect('equal')
for ax in z_ax:
    ax.set_xlabel('East')
    ax.set_ylabel('North')

plt.show()
