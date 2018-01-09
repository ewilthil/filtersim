import numpy as np
from filtersim import clutter_models
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import autoseapy.visualization as autovis

N_timesteps = 10
grid_density = 20
true_clutter_map = clutter_models.custom_map()
#true_clutter_map = clutter_models.nonuniform_musicki_map()
classic_clutter_map = clutter_models.ClassicClutterMap.from_geometric_map(true_clutter_map, grid_density, N_timesteps)
spatial_clutter_map = clutter_models.SpatialClutterMap.from_geometric_map(true_clutter_map, grid_density, N_timesteps)
temporal_clutter_map = clutter_models.TemporalClutterMap.from_geometric_map(true_clutter_map, grid_density, N_timesteps)
measurements_all  = [true_clutter_map.generate_clutter(timestamp) for timestamp in range(N_timesteps)]
map_list = [classic_clutter_map, spatial_clutter_map, temporal_clutter_map]
for measurements in measurements_all:
    for clutter_map in map_list:
        clutter_map.update_estimate(measurements)

diff_args = {'cmap' : cm.bwr}
im_args = {'cmap' : cm.Blues}
for idx, est_map in enumerate(map_list):
    fig, ax = plt.subplots(ncols=3)
    clutter_models.plot_pair_of_clutter_map(true_clutter_map, est_map, ax, im_args, diff_args)
    fig.savefig('clutter_map_%d.pdf' % (idx, ))

z_fig, z_ax = plt.subplots(ncols=2)
true_clutter_map.plot_density_map(z_ax[0], {'cmap' : cm.Greys})
autovis.plot_measurements(measurements_all, z_ax[1], cmap=cm.Greys)
z_ax[1].set_aspect('equal')
for ax in z_ax:
    ax.set_xlabel('East')
    ax.set_ylabel('North')

plt.show()
