import numpy as np
from filtersim import clutter_models
import matplotlib.pyplot as plt
import matplotlib.cm as cm

N_timesteps = 100
grid_density = 20
true_clutter_map = clutter_models.generate_true_clutter_map()
classic_clutter_map = clutter_models.ClassicClutterMap.from_geometric_map(true_clutter_map, grid_density, N_timesteps)
spatial_clutter_map = clutter_models.SpatialClutterMap.from_geometric_map(true_clutter_map, grid_density, N_timesteps)
measurements_all  = [true_clutter_map.generate_clutter() for _ in range(N_timesteps)]
for measurements in measurements_all:
    classic_clutter_map.update_estimate(measurements)
    spatial_clutter_map.update_estimate(measurements)

diff_args = {'cmap' : cm.bwr}
im_args = {'cmap' : cm.Blues}
for est_map in [classic_clutter_map, spatial_clutter_map]:
    fig, ax = plt.subplots(ncols=3)
    clutter_models.plot_pair_of_clutter_map(true_clutter_map, est_map, ax, im_args, diff_args)
plt.show()
