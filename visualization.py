import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.widgets import Slider
import matplotlib.animation as manimation
from autopy.plotting import get_ellipse
import ipdb

default_measurement_args = {
        'c' : 'w',
        'marker' : 'o',
        'ms' : 6,
}

default_estimate_args = {
        'c' : 'r',
        }

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


def plot_estimate_list(est_list, pos_ax=None, estimate_kws=default_estimate_args, measurement_kws=default_measurement_args):
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
    line = pos_ax.plot(state[2,:], state[0,:], **estimate_kws)
    for n, measurements in enumerate(measurements_all):
        for z in measurements:
            pos_ax.plot(z.value[1], z.value[0], **measurement_kws)
            #ipdb.set_trace()
        #[pos_ax.plot(z.value[1], z.value[0], 'o') for z in measurements]
    return pos_fig, pos_ax

def plot_total_velocity(est_list, ax, args=default_estimate_args):
    N = len(est_list)
    velocity = np.zeros(N)
    time = np.zeros(N)
    for n, est in enumerate(est_list):
        velocity[n] = np.sqrt(est.est_posterior[1]**2+est.est_posterior[3]**2)
        time[n] = est.timestamp
    ax.plot(time, velocity, **args)

def plot_course(est_list, ax, args=default_estimate_args):
    N = len(est_list)
    course = np.zeros(N)
    time = np.zeros(N)
    for n, est in enumerate(est_list):
        course[n] = np.arctan2(est.est_posterior[3], est.est_posterior[1])
        time[n] = est.timestamp
    ax.plot(time, np.rad2deg(course), **args)

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

def plot_measurements(measurement_list, ax, kwargs=default_measurement_args):
    for measurements in measurement_list:
        if isinstance(measurements, list):
            for z in measurements:
                ax.plot(z.value[1], z.value[0], **kwargs)
        else:
            ax.plot(measurements.value[1], measurements.value[0], **kwargs)

def plot_estimate_with_covariance(estimate, ax, color, kwargs):
    H_pos = np.array([[0, 0, 1, 0],[1, 0, 0, 0]])
    pos_est = H_pos.dot(estimate.est_posterior)
    pos_cov = H_pos.dot(estimate.cov_posterior).dot(H_pos.T)
    kwargs['c'] = color
    ax.plot(pos_est[0], pos_est[1], **kwargs)
    ellipse = get_ellipse(pos_est, pos_cov,alpha=1)
    ellipse.set_facecolor('none')
    ellipse.set_edgecolor(color)
    ellipse.set_linewidth(1)
    ax.add_artist(ellipse)
    

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

def est_dict_to_list(est_dict, t):
    all_estimates = []
    for est_list in est_dict.values():
        valid_estimates = []
        for est, status in est_list:
            if est.timestamp <= t:
                valid_estimates.append(est)
        all_estimates.append(valid_estimates)
    return all_estimates

def create_movie(est_dict, measurements_all, timestamps, fps, fname, limits=None):
    if limits is None:
        limits = find_limits(est_dict, measurement_all)
    fig, ax = plt.subplots()
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie')
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    with writer.saving(fig, fname, 300):
        for timestamp, measurements in zip(timestamps, measurements_all):
            ax.cla()
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
            estimates = est_dict_to_list(est_dict, timestamp)
            plot_measurements(measurements, ax)
            ass_meas = {'c' : 'k', 'ms' : 2, 'marker' : 'o'}
            [plot_estimate_list(est_list, ax, measurement_kws=ass_meas) for est_list in estimates]
            writer.grab_frame()
