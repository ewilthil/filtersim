from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches

Path = mpath.Path
_SEA_DATA = [
    (Path.MOVETO, (0, 0)),
    (Path.LINETO, (1600, 0)),
    (Path.LINETO, (1600, 1000)),
    (Path.LINETO, (0, 1000)),
    (Path.CLOSEPOLY, (0, 0)),
    ]
_LAND_DATA = [
    (Path.MOVETO, (0, 0)),
    (Path.LINETO, (0, 200)),
    (Path.CURVE4, (300, 400)),
    (Path.CURVE4, (600, 400)),
    (Path.CURVE4, (700, 0)),
    (Path.CLOSEPOLY, (0,0)),
    ]

_RADAR_LOC = (500, 200)

_COL_RED = '#e34a33'

_BOAT_DATA = lambda l, w : [
        (l/2, 0),
        (l/4, w/2),
        (-l/2, w/2),
        (-l/2, -w/2),
        (l/4, -w/2)]

def rotate_point(pt, angle):
    dcm = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
    return dcm.dot(pt)

class Boat(object):
    def __init__(self, loc, angle, length):
        self.loc = loc
        self.angle = angle
        self.length = length

    def get_points(self):
        mu_x = self.loc[0]
        mu_y = self.loc[1]
        theta = np.deg2rad(self.angle)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)
        l = self.length
        w = self.length/3
        points_origin = _BOAT_DATA(l,w)
        points = [self.loc+rotate_point(np.array(pt), theta) for pt in points_origin]
        return points

    def get_path(self):
        boat_data = []
        points = self.get_points()
        for k, pt in enumerate(points):
            if k == 0:
                boat_data.append((Path.MOVETO, pt))
            else:
                boat_data.append((Path.LINETO, pt))
        boat_data.append((Path.CLOSEPOLY, points[0]))
        codes, verts = zip(*boat_data)
        path = mpath.Path(verts, codes)
        return path

    def draw(self, ax, color):
        path = self.get_path()
        patch = mpatches.PathPatch(path, facecolor=color)
        ax.add_patch(patch)

boat_north = Boat([500, 800], -30, 70)
boat_east = Boat([1200, 400], 100, 70)
boat_self = Boat(_RADAR_LOC, 45, 70)

def draw_sea(ax):
    codes, verts = zip(*_SEA_DATA)
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor='#bdd7e7')
    ax.add_patch(patch)

def draw_land(ax):
    codes, verts = zip(*_LAND_DATA)
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor='#74c476')
    ax.add_patch(patch)

def draw_radar(ax):
    radar_color = '#f7f7f7'
    angle = np.deg2rad(45)
    length = 90
    width = length/3
    pole_length = length
    back_patch = mpatches.Ellipse(_RADAR_LOC, width, height, angle, color=radar_color)
    front_patch = mpatches.Ellipse(_RADAR_LOC, width, height, angle, color=radar_color)
    ax.add_patch(patch)

def adjust_size_and_ticks(ax):
    ax.set_aspect('equal')
    ax.set_xlim(0, 1600)
    ax.set_ylim(0, 1000)
    ax.set_yticks([])
    ax.set_xticks([])

if __name__ == "__main__":
    land_fig, land_ax = plt.subplots()
    boat_fig, boat_ax = plt.subplots()
    for ax in [land_ax, boat_ax]:
        draw_sea(ax)
        boat_north.draw(ax, _COL_RED)
        boat_east.draw(ax, _COL_RED)
    draw_land(land_ax)
    boat_self.draw(boat_ax, '#74c476')
    for ax in [land_ax, boat_ax]:
        adjust_size_and_ticks(ax)
    plt.show()
