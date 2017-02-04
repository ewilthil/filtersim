import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.widgets import Slider
import ipdb

def plot_with_gradient(time, measurements_all, ax=None, cmap=get_cmap('Greens')):
    # measurements is assumed to be a list of list of measurements. The outer list is the same length as the time vector
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    interval = (time-time[0])/(1.*time[-1]-time[0])
    for index, timestamp in enumerate(time):
        color = cmap(interval[index])
        [ax.plot(z.value[1], z.value[0], 'o', color=color) for z in measurements_all[index]]
    return fig, ax

def plot_trajectories_from_estimates(est_list, pos_ax=None):
    if pos_ax is None:
        pos_fig, pos_ax = plt.subplots()
    else:
        pos_fig = pos_ax.get_figure()
    N = len(est_list)
    posterior = np.zeros((4, N))
    for n, est in enumerate(est_list):
        posterior[:,n] = est.est_posterior
    pos_ax.plot(posterior[2,:], posterior[0,:])
    return pos_fig, pos_ax


def plot_estimate_list(est_list, pos_ax=None):
    if pos_ax is None:
        pos_fig, pos_ax = plt.subplots()
    else:
        pos_fig = pos_ax.get_figure()
    N = len(est_list)
    state = np.zeros((4, N))
    measurements_all = [[] for _ in range(N)]
    for n, est in enumerate(est_list):
        state[:,n] = est.est_posterior
        measurements_all[n] = est.measurements
    line = pos_ax.plot(state[2,:], state[0,:])
    for n, measurements in enumerate(measurements_all):
        for z in measurements:
            pos_ax.plot(z.value[1], z.value[0], 'wo',ms=20,mfc=(0,0,0,0))
            pos_ax.text(z.value[1], z.value[0], n)
            #ipdb.set_trace()
        #[pos_ax.plot(z.value[1], z.value[0], 'o') for z in measurements]
    return pos_fig, pos_ax

def get_estlist_limits(estimates):
    N = len(estimates)
    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    for est in estimates:
        state = est.est_posterior
        if state[2] < xmin:
            xmin = state[2]
        if state[2] > xmax:
            xmax = state[2]
        if state[0] < ymin:
            ymin = state[0]
        if state[0] > ymax:
            ymax = state[0]
    return np.array([xmin, xmax]), np.array([ymin, ymax])

def plot_measurements(measurement_list, ax):
    for measurements in measurement_list:
        for z in measurements:
            ax.plot(z.value[1], z.value[0], 'wo')

def animate_track(estimates, measurements, true_track=None):
    xlimits, ylimits = get_estlist_limits(estimates)
    fig = plt.figure()
    pos_ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    pos_ax.set_xlim(xlimits)
    pos_ax.set_ylim(ylimits)
    N = len(estimates)
    slideax = fig.add_axes([0.03, 0.03, 0.6, 0.03])
    slider = Slider(slideax, 'Timestep', 0, N, 0,valfmt='%d')
    plot_trajectories_from_estimates(estimates[:1], pos_ax)
    def update(n_current):
        pos_ax.clear()
        n_current = int(np.floor(slider.val))
        plot_trajectories_from_estimates(estimates[:n_current+1], pos_ax)
        plot_measurements([measurements[n_current+1]], pos_ax)
        pos_ax.set_xlim(xlimits)
        pos_ax.set_ylim(ylimits)
        fig.canvas.draw_idle()
    slider.on_changed(update)
