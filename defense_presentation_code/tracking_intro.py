from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.animation as manimation

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
_RADAR_ANGLE = 45
_RADAR_TRANSFORM = mtransforms.Affine2D().rotate_deg(_RADAR_ANGLE).translate(*_RADAR_LOC)

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

def pi2pi(ang):
    while ang < -np.pi:
        ang = ang+2*np.pi
    while ang > np.pi:
        ang = ang-2*np.pi
    return ang

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

def locs2ang(start_loc, end_loc):
    return np.arctan2(end_loc[1]-start_loc[1], end_loc[0]-start_loc[0])*180.0/np.pi

_NORTH_LOC = np.array([500, 800])
_EAST_LOC = np.array([1200, 400])
_CLUTTER_LOC = [[#Scan 1
        np.array([1000, 200]),
        np.array([1300, 900]),
        np.array([1200, 350])],
        [# Scan 2
        np.array([900, 900]),
        np.array([100, 800]),
        np.array([400, 820])],
        [ # Scan 3
        np.array([450, 810]),
        np.array([1200, 370]),
        np.array([600, 800])],
        [ # Scan 4
        np.array([470, 800]),
        np.array([1210, 400]),
        np.array([700, 300])]
        ]
_CLUTTER_ANG = []
for lst in _CLUTTER_LOC:
    _CLUTTER_ANG.append([locs2ang(loc, _RADAR_LOC) for loc in lst])
_NORTH_ANG = locs2ang(_NORTH_LOC, _RADAR_LOC)
_EAST_ANG = locs2ang(_EAST_LOC, _RADAR_LOC)
boat_north = Boat(_NORTH_LOC, -30, 70)
boat_east = Boat(_EAST_LOC, 100, 70)
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

def draw_radar(ax, transformation=mtransforms.Affine2D(), scale=1):
    dish = mpatches.Wedge([0, 0], 3*scale, 90, 270, ec='k', fc='white', zorder=4)
    antenna = mpatches.Circle(np.array([3, 0])*scale, 0.33*scale, ec='k', fc='k',zorder=4)
    for element in [dish, antenna]:
        c = ax.add_patch(element)
        c.set_transform(transformation+ax.transData)
    rod = mlines.Line2D(np.array([0, 3])*scale, np.array([0, 0]), color='k')
    for element in [rod]:
        c = ax.add_artist(element)
        c.set_transform(transformation+ax.transData)

def adjust_size_and_ticks(ax):
    ax.set_position([0.01, 0.01, 0.99, 0.99])
    ax.set_aspect('equal')
    ax.set_xlim(0, 1600)
    ax.set_ylim(0, 1000)
    ax.set_yticks([])
    ax.set_xticks([])

def plot_full_scan(ax, loc, scale, percentage):
    handles_all = []
    for n in range(1,5):
        radii = mpatches.Circle(loc, radius=scale*n*(1+percentage), ec='k', fc='none')
        h = ax.add_artist(radii)
        handles_all.append(h)
    return handles_all

def plot_partial_scan(ax, loc, scale, start_ang, end_ang, percentage):
    handles_all = []
    for n in range(1,5):
        radii = mpatches.Wedge(loc, scale*n*(1+percentage), start_ang, end_ang, 1.0/scale, ec='k', fc='none')
        h = ax.add_artist(radii)
        handles_all.append(h)
    return handles_all

def radar_full_scan_movie():
    fig, ax = plt.subplots(figsize=(16,10))
    draw_sea(ax)
    draw_land(ax)
    boat_north.draw(ax, _COL_RED)
    boat_east.draw(ax, _COL_RED)
    draw_radar(ax, _RADAR_TRANSFORM, scale=10)
    FFMpegWriter = manimation.writers['ffmpeg']
    writer = FFMpegWriter(fps=24)
    with writer.saving(fig, "radar_full_scan.mp4", 300):
        for current_percentage in np.linspace(0,3, 25):
            current_handles = plot_full_scan(ax, _RADAR_LOC, 30, current_percentage)
            adjust_size_and_ticks(ax)
            writer.grab_frame()
            for handle in current_handles:
                handle.remove()

def radar_partial_scan_movie():
    fig, ax = plt.subplots(figsize=(16,10))
    draw_sea(ax)
    draw_land(ax)
    boat_north.draw(ax, _COL_RED)
    boat_east.draw(ax, _COL_RED)
    draw_radar(ax, _RADAR_TRANSFORM, scale=10)
    FFMpegWriter = manimation.writers['ffmpeg']
    writer = FFMpegWriter(fps=24)
    with writer.saving(fig, "radar_partial_scan.mp4", 300):
        for current_percentage in np.linspace(0, 3, 25):
            current_handles_north = plot_partial_scan(ax, _NORTH_LOC, 10, _NORTH_ANG-15, _NORTH_ANG+15, current_percentage)
            current_handles_east = plot_partial_scan(ax, _EAST_LOC, 10, _EAST_ANG-15, _EAST_ANG+15, current_percentage)
            adjust_size_and_ticks(ax)
            writer.grab_frame()
            for handle in current_handles_north+current_handles_east:
                handle.remove()

def radar_rotation_movie():
    fig, ax = plt.subplots(figsize=(16,10))
    draw_sea(ax)
    draw_land(ax)
    boat_north.draw(ax, _COL_RED)
    boat_east.draw(ax, _COL_RED)
    draw_radar(ax, _RADAR_TRANSFORM, scale=10)
    FFMpegWriter = manimation.writers['ffmpeg']
    writer = FFMpegWriter(fps=24)
    with writer.saving(fig, "radar_rotation_scan.mp4", 300):
        for angle in np.linspace(-np.pi, np.pi, 50):
            removable_handles = []
            ang = _EAST_ANG
            loc = _EAST_LOC
            if angle > pi2pi(np.deg2rad(180-ang)):
                h = ax.plot(loc[0], loc[1], 'ko')
                removable_handles.append(h[0])
            ang = _NORTH_ANG
            loc = _NORTH_LOC
            if angle > pi2pi(np.deg2rad(360-ang)):
                h = ax.plot(loc[0], loc[1], 'ko')
                removable_handles.append(h[0])
            x_data = np.array([_RADAR_LOC[0], _RADAR_LOC[0]+2000*np.cos(angle)])
            y_data = np.array([_RADAR_LOC[1], _RADAR_LOC[1]+2000*np.sin(angle)])
            h = ax.plot(x_data, y_data, 'k-')
            removable_handles.append(h[0])
            adjust_size_and_ticks(ax)
            writer.grab_frame()
            [h.remove() for h in removable_handles]

def radar_rotation_clutter_movie():
    fig, ax = plt.subplots(figsize=(16,10))
    draw_sea(ax)
    draw_land(ax)
    boat_north.draw(ax, _COL_RED)
    boat_east.draw(ax, _COL_RED)
    draw_radar(ax, _RADAR_TRANSFORM, scale=10)
    FFMpegWriter = manimation.writers['ffmpeg']
    writer = FFMpegWriter(fps=24)
    with writer.saving(fig, "radar_rotation_clutter_scan.mp4", 300):
        for angle in np.linspace(-np.pi, np.pi, 50):
            removable_handles = []
            ang_all =  _CLUTTER_ANG[0]
            loc_all = _CLUTTER_LOC[0]
            for ang, loc in zip(ang_all, loc_all):
                if angle > pi2pi(-np.deg2rad(180-ang)):
                    h = ax.plot(loc[0], loc[1], 'ko')
                    removable_handles.append(h[0])

            x_data = np.array([_RADAR_LOC[0], _RADAR_LOC[0]+2000*np.cos(angle)])
            y_data = np.array([_RADAR_LOC[1], _RADAR_LOC[1]+2000*np.sin(angle)])
            h = ax.plot(x_data, y_data, 'k-')
            removable_handles.append(h[0])
            adjust_size_and_ticks(ax)
            writer.grab_frame()
            [h.remove() for h in removable_handles]

def radar_rotation_clutter_full_movie():
    fig, ax = plt.subplots(figsize=(16,10))
    draw_sea(ax)
    draw_land(ax)
    #boat_north.draw(ax, _COL_RED)
    #boat_east.draw(ax, _COL_RED)
    draw_radar(ax, _RADAR_TRANSFORM, scale=10)
    FFMpegWriter = manimation.writers['ffmpeg']
    writer = FFMpegWriter(fps=24)
    with writer.saving(fig, "radar_rotation_clutter_full.mp4", 300):
        # Scan 1
        for angle in np.linspace(-np.pi, np.pi, 50):
            removable_handles = []
            ang_all =  _CLUTTER_ANG[0]
            loc_all = _CLUTTER_LOC[0]
            for ang, loc in zip(ang_all, loc_all):
                if angle > pi2pi(-np.deg2rad(180-ang)):
                    h = ax.plot(loc[0], loc[1], 'ko')
                    removable_handles.append(h[0])
            x_data = np.array([_RADAR_LOC[0], _RADAR_LOC[0]+2000*np.cos(angle)])
            y_data = np.array([_RADAR_LOC[1], _RADAR_LOC[1]+2000*np.sin(angle)])
            h = ax.plot(x_data, y_data, 'k-')
            removable_handles.append(h[0])
            adjust_size_and_ticks(ax)
            writer.grab_frame()
            [h.remove() for h in removable_handles]

        # Scan 2
        for angle in np.linspace(-np.pi, np.pi, 50):
            removable_handles = []
            ang_persistent = _CLUTTER_ANG[0]
            loc_persistent = _CLUTTER_LOC[0]
            for ang, loc in zip(ang_persistent, loc_persistent):
                h = ax.plot(loc[0], loc[1], 'ko')
                removable_handles.append(h[0])
            ang_all =  _CLUTTER_ANG[1]
            loc_all = _CLUTTER_LOC[1]
            for ang, loc in zip(ang_all, loc_all):
                if angle > pi2pi(-np.deg2rad(180-ang)):
                    h = ax.plot(loc[0], loc[1], 'ko')
                    removable_handles.append(h[0])
            x_data = np.array([_RADAR_LOC[0], _RADAR_LOC[0]+2000*np.cos(angle)])
            y_data = np.array([_RADAR_LOC[1], _RADAR_LOC[1]+2000*np.sin(angle)])
            h = ax.plot(x_data, y_data, 'k-')
            removable_handles.append(h[0])
            adjust_size_and_ticks(ax)
            writer.grab_frame()
            [h.remove() for h in removable_handles]

        # Scan 3
        for angle in np.linspace(-np.pi, np.pi, 50):
            removable_handles = []
            ang_persistent = _CLUTTER_ANG[0]+_CLUTTER_ANG[1]
            loc_persistent = _CLUTTER_LOC[0]+_CLUTTER_LOC[1]
            for ang, loc in zip(ang_persistent, loc_persistent):
                h = ax.plot(loc[0], loc[1], 'ko')
                removable_handles.append(h[0])
            ang_all =  _CLUTTER_ANG[2]
            loc_all = _CLUTTER_LOC[2]
            for ang, loc in zip(ang_all, loc_all):
                if angle > pi2pi(-np.deg2rad(180-ang)):
                    h = ax.plot(loc[0], loc[1], 'ko')
                    removable_handles.append(h[0])
            x_data = np.array([_RADAR_LOC[0], _RADAR_LOC[0]+2000*np.cos(angle)])
            y_data = np.array([_RADAR_LOC[1], _RADAR_LOC[1]+2000*np.sin(angle)])
            h = ax.plot(x_data, y_data, 'k-')
            removable_handles.append(h[0])
            adjust_size_and_ticks(ax)
            writer.grab_frame()
            [h.remove() for h in removable_handles]

        # Scan 4
        for angle in np.linspace(-np.pi, np.pi, 50):
            removable_handles = []
            ang_persistent = _CLUTTER_ANG[0]+_CLUTTER_ANG[1]+_CLUTTER_ANG[2]
            loc_persistent = _CLUTTER_LOC[0]+_CLUTTER_LOC[1]+_CLUTTER_LOC[2]
            for ang, loc in zip(ang_persistent, loc_persistent):
                h = ax.plot(loc[0], loc[1], 'ko')
                removable_handles.append(h[0])
            ang_all =  _CLUTTER_ANG[3]
            loc_all = _CLUTTER_LOC[3]
            for ang, loc in zip(ang_all, loc_all):
                if angle > pi2pi(-np.deg2rad(180-ang)):
                    h = ax.plot(loc[0], loc[1], 'ko')
                    removable_handles.append(h[0])
            x_data = np.array([_RADAR_LOC[0], _RADAR_LOC[0]+2000*np.cos(angle)])
            y_data = np.array([_RADAR_LOC[1], _RADAR_LOC[1]+2000*np.sin(angle)])
            h = ax.plot(x_data, y_data, 'k-')
            removable_handles.append(h[0])
            adjust_size_and_ticks(ax)
            writer.grab_frame()
            [h.remove() for h in removable_handles]
if __name__ == "__main__":
    #land_fig, land_ax = plt.subplots()
    #boat_fig, boat_ax = plt.subplots()
    #for ax in [land_ax, boat_ax]:
    #    draw_sea(ax)
    #    boat_north.draw(ax, _COL_RED)
    #    boat_east.draw(ax, _COL_RED)
    #    plot_full_scan(ax, _RADAR_LOC, 50, 0)
    #draw_sea(land_ax)
    #draw_land(land_ax)
    #boat_north.draw(land_ax, _COL_RED)
    #boat_east.draw(land_ax, _COL_RED)
    #draw_radar(land_ax, _RADAR_TRANSFORM, scale=10)
    #boat_self.draw(boat_ax, '#74c476')
    #for ax in [land_ax, boat_ax]:
    #    adjust_size_and_ticks(ax)

    radar_full_scan_movie()
    radar_partial_scan_movie()
    radar_rotation_movie()
    radar_rotation_clutter_movie()
    radar_rotation_clutter_full_movie()
    plt.show()
