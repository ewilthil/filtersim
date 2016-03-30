import numpy as np
import matplotlib.pyplot as plt
import brewer2mpl
from autopy.sylte import load_pkl
from scipy.stats import chi2
#plt.style.use('filter')

errs = load_pkl('track_err_100N.pkl')
def get_bounds(time_vec, percentile=0.95):
    lower_lim = (1-percentile)/2
    upper_lim = 1-lower_lim
    UB = chi2(df=2*errs[0].N_mc).ppf(upper_lim)/errs[0].N_mc*np.ones_like(time_vec)
    LB = chi2(df=2*errs[0].N_mc).ppf(lower_lim)/errs[0].N_mc*np.ones_like(time_vec)
    return UB, LB
with plt.style.context(('filter')):
    UB, LB = get_bounds(errs[0].time)
    line_color = '0.4'
    mark_every = 10
    marker_size=5
    ground_truth_style = {'label' : 'Ground truth pose', 'marker' : 'o', 'markevery' : mark_every, 'ms' : marker_size}
    schmidt_style = {'label' : 'Schmidt', 'marker' : 's', 'markevery' : mark_every, 'ms' : marker_size}
    uncomp_style = {'label' : 'Uncompensated', 'marker' : 'v', 'markevery' : mark_every, 'ms' : marker_size}
    conv_style = {'label' : 'Converted measurement', 'marker' : 'h', 'markevery' : mark_every, 'ms' : marker_size}
    bound_args = {'color' : 'k', 'lw' : 2}
    markers = (ground_truth_style, uncomp_style, schmidt_style, conv_style)



    con_fig, con_ax = plt.subplots(1,1)
    for j, err in enumerate(errs):
        con_ax.plot(err.time, np.mean(err.NEES, axis=0), **markers[j])
    con_ax.plot(errs[0].time, UB, label='95% bounds', **bound_args)
    con_ax.plot(errs[0].time, LB, **bound_args)
    con_ax.set_title('Consistency results')
    con_ax.set_xlabel('Time [s]')
    con_ax.set_ylabel('NEES')
    con_ax.legend()
    plt.savefig('figs/NEES-straight-line.pdf')


    pos_fig, pos_ax = plt.subplots(1,1)
    for j, err in enumerate(errs):
        pos_ax.plot(err.time, np.mean(err.RMSE_pos, axis=0), **markers[j])
    pos_ax.set_title('Position RMSE')
    pos_ax.set_xlabel('Time [s]')
    pos_ax.set_ylabel('RMSE [m]')
    pos_ax.legend()
    plt.savefig('figs/RMSE-pos-straight.pdf')





    vel_fig, vel_ax = plt.subplots(1,1)
    for j, err in enumerate(errs):
        vel_ax.plot(err.time, np.mean(err.RMSE_vel, axis=0), **markers[j])
    vel_ax.set_title('Velocity RMSE')
    vel_ax.set_xlabel('Time [s]')
    vel_ax.set_ylabel('RMSE [m/s]')
    vel_ax.legend()
    plt.savefig('figs/RMSE-vel-straight.pdf')



plt.show()

