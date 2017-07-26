import numpy as np
from collections import deque
import ipdb
from autoseapy.tracking import Measurement

class CircleRegion(object):
    def __init__(self, density, center, radius):
        self.N_min = center[0]-radius
        self.N_max = center[0]+radius
        self.E_min = center[1]-radius
        self.E_max = center[1]+radius
        self.area = np.pi*radius**2
        self.density = density
        self.center = center
        self.radius = radius

    def point_in_region(self, pt):
        if (pt[0]-self.center[0])**2+(pt[1]-self.center[1])**2 <= self.radius**2:
            return True
        else:
            return False

class PolygonRegion(object):
    def __init__(self, density, vertices):
        self.density = density
        self.vertices = vertices
        self.N_min = np.min([v[0] for v in vertices])
        self.N_max = np.max([v[0] for v in vertices])
        self.E_min = np.min([v[1] for v in vertices])
        self.E_max = np.max([v[1] for v in vertices])
        self.area = self.compute_area()

    def point_in_region(self, pt):
        times_crossed = 0
        for edge_start, edge_end in self.get_edges():
            edge = (edge_start, edge_end)
            if self.ray_cross_edge(pt, edge):
                times_crossed += 1
        return np.mod(times_crossed, 2) == 1

    def get_edges(self):
        p0 = self.vertices[-1]
        for p1 in self.vertices:
            yield p0, p1
            p0 = p1

    def compute_area(self):
        x = np.array([v[0] for v in self.vertices])
        y = np.array([v[1] for v in self.vertices])
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1))) # Shoelace method

    def ray_cross_edge(self, start_pt, edge):
        ray_start = start_pt
        ray_end = [start_pt[0], 1.1*self.E_max]
        ray = (ray_start, ray_end)
        # The ray cross the edge if and only if the endpoints of the first segment are on opposite sides of the second, and vice versa.
        return self.opposite_side_of_line(ray, edge) and self.opposite_side_of_line(edge, ray)
        
    def opposite_side_of_line(self, first_segment, second_segment):
        # help variables. The segments are defined as (start_point, end_point)
        x1 = second_segment[0][0]
        y1 = second_segment[0][1]
        x2 = second_segment[1][0]
        y2 = second_segment[1][1]
        ax = first_segment[0][0]
        ay = first_segment[0][1]
        bx = first_segment[1][0]
        by = first_segment[1][1]
        return ((y1-y2)*(ax-x1)+(x2-x1)*(ay-y1))*((y1-y2)*(bx-x1)+(x2-x1)*(by-y1)) < 0

class GeometricClutterMap(object):
    def __init__(self, N_min, N_max, E_min, E_max, base_density):
        self.area = (N_max-N_min)*(E_max-E_min)
        self.base_area = self.area
        self.N_min, self.N_max = N_min, N_max
        self.E_min, self.E_max = E_min, E_max
        self.base_density = base_density
        self.regions = []

    def __repr__(self):
        exp_measurements = self.area*self.base_density
        for region in self.regions:
            exp_measurements += region.area*region.density
        return "Clutter map of %d m^2 with %4.1f average clutter measurements" % (self.base_area, exp_measurements)

    def detailed_info(self):
        pass

    def add_region(self, region):
        self.regions.append(region)
        self.area -= region.area

    def generate_clutter(self, timestamp):
        measurements = []
        for region in self.regions:
            n_region_measurement = np.random.poisson(lam=region.area*region.density)
            n_region_generated = 0
            while n_region_generated < n_region_measurement:
                meas_pos = np.random.uniform(low=[region.N_min, region.E_min], high=[region.N_max, region.E_max])
                if region.point_in_region(meas_pos):
                    measurements.append(Measurement(meas_pos, timestamp))
                    n_region_generated += 1

        n_base_region_measurements = np.random.poisson(lam=self.base_density*self.area)
        n_base_region_generated = 0
        while n_base_region_generated < n_base_region_measurements:
            meas_pos = np.random.uniform(low=[self.N_min, self.E_min], high=[self.N_max, self.E_max])
            in_other_cell = False
            for region in self.regions:
                if region.point_in_region(meas_pos):
                    in_other_cell = True
            if not in_other_cell:
                measurements.append(Measurement(meas_pos, timestamp))
                n_base_region_generated += 1
        return measurements
    
    def get_density(self, N, E):
        pt = np.array([N+0.1, E+0.1])
        density = self.base_density
        for region in self.regions:
            if region.point_in_region(pt):
                density = region.density
        return density

    def plot_density_map(self, ax, im_args):
        #TODO should plot shapes instead of lookup density. Use cmap, vmin and vmax to get the color.
        N_vec = np.linspace(self.N_min, self.N_max, 201)
        E_vec = np.linspace(self.E_min, self.E_max, 201)
        N_grid, E_grid = np.meshgrid(N_vec, E_vec)
        dens_vec = np.vectorize(self.get_density)
        density_grid = dens_vec(N_grid, E_grid)
        cax = ax.pcolormesh(E_grid, N_grid, density_grid, **im_args)
        fig = ax.get_figure()
        cbar = fig.colorbar(cax, orientation='horizontal',ax=ax)
        ax.set_aspect('equal')

    def plot_with_measurements(self, ax, measurements, im_args = dict()):
        self.plot_density_map(ax, im_args)
        for z in measurements:
            ax.plot(z[1], z[0], 'ko')

    def discretize_map(self, resolution):
        N_vec = np.arange(self.N_min, self.N_max+resolution, resolution)
        E_vec = np.arange(self.E_min, self.E_max+resolution, resolution)
        N_grid, E_grid = np.meshgrid(N_vec, E_vec)
        density_grid = np.vectorize(self.get_density)(N_grid, E_grid)
        return DiscreteClutterMap(N_grid, E_grid, density_grid)

class DiscreteClutterMap(object):
    def __init__(self, N_grid, E_grid, density_grid):
        self.N_grid = N_grid
        self.E_grid = E_grid
        self.density_grid = density_grid
        self.N_min = np.amin(N_grid)
        self.E_min = np.amin(E_grid)
        self.resolution = np.amin(np.diff(N_grid))

    def get_density(self, N, E):
        n_idx, e_idx = self.pt2idx(N, E)
        return self.density_grid[n_idx, e_idx]

    def pt2idx(self, N, E):
        N_idx = np.floor(float(N-self.N_min)/self.resolution)
        E_idx = np.floor(float(E-self.E_min)/self.resolution)
        return int(N_idx), int(E_idx)



class ClutterMap(object):
    def __init__(self, N_min, N_max, E_min, E_max, resolution, averaging_length, celltype):
        self.N_min, self.N_max = N_min, N_max
        self.E_min, self.E_max = E_min, E_max
        self.resolution = resolution
        # Construct a grid from the input
        self.N_vec = np.arange(N_min, N_max+resolution, resolution)
        self.E_vec = np.arange(E_min, E_max+resolution, resolution)
        self.N_idx = len(self.N_vec)
        self.E_idx = len(self.E_vec)
        self.grid = [[celltype(N, E, resolution, averaging_length) for E in self.E_vec] for N in self.N_vec]

    def plot_density_map(self, ax, im_args):
        N_grid, E_grid = np.meshgrid(self.N_vec, self.E_vec)
        dens_vec = np.vectorize(self.get_density)
        density_grid = dens_vec(N_grid, E_grid)
        cax = ax.pcolormesh(E_grid, N_grid, density_grid, **im_args)
        fig = ax.get_figure()
        cbar = fig.colorbar(cax, orientation='horizontal',ax=ax)
        ax.set_aspect('equal')

    def update_estimate(self, measurements):
        average_values = self.get_averaging_values(measurements)
        for n_idx in range(self.N_idx):
            for e_idx in range(self.E_idx):
                self.grid[n_idx][e_idx].add_measurement(average_values[n_idx][e_idx])
        

    def pt2idx(self, N, E):
        N_idx = np.floor(float(N-self.N_min)/self.resolution)
        E_idx = np.floor(float(E-self.E_min)/self.resolution)
        return int(N_idx), int(E_idx)

    def get_density(self, N, E):
        N_idx, E_idx = self.pt2idx(N, E)
        return self.grid[N_idx][E_idx].get_density()

    def get_densities(self):
        densities = [[self.grid[i][j].get_density() for i in range(self.N_idx)] for j in range(self.E_idx)]
        return np.array(densities)
    
    @classmethod
    def from_geometric_map(cls, other_map, resolution, averaging_length, celltype):
        return cls(other_map.N_min, other_map.N_max, other_map.E_min, other_map.E_max, resolution, averaging_length, celltype)

    def plot_difference(self, true_map, ax, im_args, resolution=None, rel_diff_max=10):
        if resolution == None:
            resolution = self.resolution
        N_vec = np.arange(self.N_min, self.N_max+resolution, resolution)
        E_vec = np.arange(self.E_min, self.E_max+resolution, resolution)
        N_grid, E_grid = np.meshgrid(N_vec, E_vec)
        relative_difference = np.zeros((len(E_vec), len(N_vec)))
        for n_idx, n in enumerate(N_vec):
            for e_idx, e in enumerate(E_vec):
                rel_diff = (true_map.get_density(n, e)-self.get_density(n, e))/true_map.get_density(n, e)
                if np.abs(rel_diff) > rel_diff_max:
                    rel_diff = np.sign(rel_diff)*rel_diff_max
                relative_difference[e_idx, n_idx] = rel_diff
        diff_max = np.amax(np.abs(relative_difference))
        im_args['vmin'] = -rel_diff_max
        im_args['vmax'] = rel_diff_max
        cax = ax.pcolormesh(E_grid, N_grid, relative_difference, **im_args)
        fig = ax.get_figure()
        cbar = fig.colorbar(cax, orientation='horizontal',ax=ax)
        ax.set_aspect('equal')

class ClutterCell(object):
    def __init__(self, N, E, resolution, averaging_length):
        self.averagor = deque(maxlen=averaging_length)
        self.area = resolution**2

    def get_density(self):
        return np.mean(self.averagor)

    def add_measurement(self, z):
        if not np.isnan(z):
            self.averagor.append(z)

class ClassicClutterCell(ClutterCell):
    def get_density(self):
        N, L = np.sum(self.averagor), len(self.averagor)
        if N == 0:
            N = 1.0
        inv_density = L*self.area/N
        return 1.0/inv_density

class SpatialClutterCell(ClutterCell):
    def get_density(self):
        N = np.mean(self.averagor)
        inv_density = N
        return 1.0/inv_density

class TemporalClutterCell(ClutterCell):
    def get_density(self):
        tau = np.mean(self.averagor)
        inv_density = self.area*tau
        return 1.0/inv_density

class ClassicClutterMap(ClutterMap):
    def get_averaging_values(self, measurements):
        number_of_measurements = [[0 for _ in self.E_vec] for _ in self.N_vec]
        for measurement in measurements:
            N_idx, E_idx = self.pt2idx(measurement[0], measurement[1])
            number_of_measurements[N_idx][E_idx] += 1
        return number_of_measurements

    @classmethod
    def from_geometric_map(cls, other_map, resolution, averaging_length):
        return super(ClassicClutterMap, cls).from_geometric_map(other_map, resolution, averaging_length, ClassicClutterCell)

class SpatialClutterMap(ClutterMap):
    def get_averaging_values(self, measurements):
        cell_measurements = [[[] for _ in self.E_vec] for _ in self.N_vec]
        average_values = np.zeros((self.N_idx, self.E_idx))
        for n_idx in range(self.N_idx):
            for e_idx in range(self.E_idx):
                center_point = np.array([self.N_vec[n_idx], self.E_vec[e_idx]])+np.array([self.resolution, self.resolution])
                norms = [np.linalg.norm(center_point-z.value, np.inf) for z in measurements]
                average_values[n_idx, e_idx] = (2.*np.amin(norms))**2
        return average_values

    @classmethod
    def from_geometric_map(cls, other_map, resolution, averaging_length):
        return super(SpatialClutterMap, cls).from_geometric_map(other_map, resolution, averaging_length, SpatialClutterCell)

class TemporalClutterMap(ClutterMap):
    def get_averaging_values(self, measurements):
        timestamp = measurements[0].timestamp # Assume all have same timestamp
        cell_measurements = [[[] for _ in self.E_vec] for _ in self.N_vec]
        averagor_values = np.zeros((self.N_idx, self.E_idx))*np.nan
        for measurement in measurements:
            N_idx, E_idx = self.pt2idx(measurement[0], measurement[1])
            cell_measurements[N_idx][E_idx].append(measurement)
        for n_idx in range(self.N_idx):
            for e_idx in range(self.E_idx):
                num_measurements = len(cell_measurements[n_idx][e_idx])
                if num_measurements > 0:
                    latest_timestamp = self.grid[n_idx][e_idx].latest_timestamp
                    averagor_value = (timestamp-latest_timestamp)/float(num_measurements)
                    [self.grid[n_idx][e_idx].add_measurement(averagor_value) for n in range(num_measurements)]
                    self.grid[n_idx][e_idx].latest_timestamp = timestamp
        return averagor_values


    @classmethod
    def from_geometric_map(cls, other_map, resolution, averaging_length, t0=0):
        temporalMap = super(TemporalClutterMap, cls).from_geometric_map(other_map, resolution, averaging_length, TemporalClutterCell)
        for row in temporalMap.grid:
            for cell in row:
                cell.latest_timestamp = t0
        return temporalMap

def plot_pair_of_clutter_map(true_map, estimated_map, ax_list, im_args, diff_args):
    min_density = 5e-5 #TODO
    max_density = 1e-3 #TODO
    im_args['vmin'] = min_density
    im_args['vmax'] = max_density
    true_map.plot_density_map(ax_list[0], im_args)
    estimated_map.plot_density_map(ax_list[1], im_args)
    estimated_map.plot_difference(true_map, ax_list[2], diff_args)

def uniform_musicki_map():
    return GeometricClutterMap(0, 1000, 0, 400, 1e-4)

def nonuniform_musicki_map():
    true_map = uniform_musicki_map()
    high_density = 7e-4
    high_density_region = PolygonRegion(high_density, [(320, 0), (320, 400), (500, 400), (500, 0)])
    true_map.add_region(high_density_region)
    return true_map

def custom_map():
    density_min = 5e-5
    density_max = 1e-3
    true_clutter_map = GeometricClutterMap(0, 400, 0, 400, density_min)

    clutter_circle = CircleRegion(density_max, np.array([300, 300]), 50)
    true_clutter_map.add_region(clutter_circle)

    clutter_box = PolygonRegion(float(density_min+density_max)/2, [(300, 0), (400, 0), (400, 150), (300, 150)])
    true_clutter_map.add_region(clutter_box)

    n_transients = 5
    N_start = 100
    N_end = 200
    dist_transient = float(N_end-N_start)/n_transients
    density_resolution = float(density_max-density_min)/n_transients
    N_points = np.linspace(N_start, N_end-dist_transient, n_transients)
    E_points = N_points
    vertices = []
    densities = np.linspace(density_max-density_resolution, density_min+density_resolution, n_transients)
    for N, E in zip(N_points, E_points):
        v = [(N, 0), (N+dist_transient, 0), (0, E+dist_transient), (0, E)]
        vertices.append(v)
    [true_clutter_map.add_region(PolygonRegion(dens, v)) for (dens, v) in zip(densities, vertices)]
    true_clutter_map.add_region(PolygonRegion(density_max, [(0, 0), (N_start, 0), (0, N_start)]))
    return true_clutter_map
