import numpy as np
import autoseapy.tracking as autotrack
from scipy.stats import chi2, multivariate_normal
import autoseapy.visualization as autovis
from ipdb import set_trace
import matplotlib.pyplot as plt

def get_rmse(true_track, est_track):
    timestamps = set([estimate.timestamp for estimate in est_track])
    partial_true_track = [state for state in true_track if state.timestamp in timestamps]
    est_states = np.array([est.est_posterior for est in est_track])
    true_states = np.array([state.est_posterior for state in partial_true_track])
    x_pos_err = true_states[:,0]-est_states[:,0]
    y_pos_err = true_states[:,2]-est_states[:,2]
    pos_rmse = np.sqrt(x_pos_err**2+y_pos_err**2)
    return [est.timestamp for est in est_track], pos_rmse

def is_true_track(true_track, est_track):
    _, pos_rmse = get_rmse(true_track, est_track)
    return np.all(pos_rmse < 20)

def run_track_manager(track_manager, measurements_all, time, true_target):
    N_true_tracks = 0
    N_false_tracks = 0
    N_total_targets = 1
    new_track_timestamps = dict()
    new_tracks_all = dict()
    for measurements, timestamp in zip(measurements_all, time):
        old_estimates, new_tracks = track_manager.step(measurements, timestamp)
        for track in new_tracks:
            idx = track[0].track_index
            new_track_timestamps[idx] = track[-1].timestamp-track[0].timestamp
            new_tracks_all[idx] = track
    time_rmse_all = []
    rmse_all = []
    true_init_time_all = []
    false_init_time_all = []
    track_detected = False
    for track in new_tracks_all.values():
        time_rmse, rmse = get_rmse(true_target, track)
        time_rmse_all.append(time_rmse)
        rmse_all.append(rmse)
        if is_true_track(true_target, track):
            if not track_detected:
                N_true_tracks += 1
                #track_detected = True
                true_init_time_all.append(new_track_timestamps[track[0].track_index])
        else:
            N_false_tracks += 1
            false_init_time_all.append(new_track_timestamps[track[0].track_index])
    return N_true_tracks, N_false_tracks, N_total_targets, time_rmse_all, rmse_all, true_init_time_all, false_init_time_all

class MCTrackInit(object):
    def __init__(self, manager, true_track, name):
        self.manager = manager
        self.true_track = true_track
        self.timestamps = np.array([track.timestamp for track in true_track])
        self.stats = {'N_targets' : 0, 'N_tracks' : 0, 'N_true_tracks' : 0, 'N_false_tracks' : 0}
        self.rmse_all = []
        self.time_rmse_all = []
        self.true_init_time_all = []
        self.false_init_time_all = []
        self.name = name

    def step(self, measurements_all):
        self.manager.reset()
        N_current_true_tracks, N_current_false_tracks, N_current_targets, time_rmse, rmse, true_init_time, false_init_time = run_track_manager(self.manager, measurements_all, self.timestamps, self.true_track)
        self.update_stats(N_current_true_tracks, N_current_false_tracks, N_current_targets)
        self.rmse_all.append(rmse)
        self.time_rmse_all.append(time_rmse)
        self.true_init_time_all.append(true_init_time)
        self.false_init_time_all.append(false_init_time)

    def plot_rmse(self, rmse_ax):
        for rmse_list, time_list in zip(self.rmse_all, self.time_rmse_all):
            for rmse, time in zip(rmse_list, time_list):
                rmse_ax.plot(time, rmse)

    def update_stats(self, N_current_true, N_current_false, N_current_targets):
        self.stats['N_targets'] += N_current_targets
        self.stats['N_tracks'] += (N_current_true+N_current_false)
        self.stats['N_true_tracks'] += np.min((N_current_true, 1))
        self.stats['N_false_tracks'] += N_current_false
    
    def get_stats(self):
        P_DT = float(self.stats['N_true_tracks'])/self.stats['N_targets']
        if self.stats['N_tracks'] > 0:
            P_FT = float(self.stats['N_false_tracks'])/self.stats['N_tracks']
        else:
            P_FT = 0.0
        return P_DT, P_FT

    def get_time_info(self):
        n_scans_true = []
        n_scans_false = []
        for true_init_times in self.true_init_time_all:
            if len(true_init_times) > 0:
                n_scans_true.append(true_init_times[0])
        for false_init_times in self.false_init_time_all:
            for init_time in false_init_times:
                n_scans_false.append(init_time)
        return n_scans_true, n_scans_false
        

    def print_stats(self):
        P_DT, P_FT = self.get_stats()
        print "{} P_DT={}".format(self.name, P_DT)
        print "{} P_FT={}".format(self.name, P_FT)

class MCManager(object):
    def __init__(self, dep_var_name, N_MC):
        self.manager = None
        self.N_MC = N_MC
        self.dep_var_name = dep_var_name
        self.dependent_variables = []
        self.prob_detect_track = []
        self.prob_false_track = []
        self.average_true_init_time = []
        self.average_false_init_time = []
    
    def update_stats(self, dependent_variable):
        self.dependent_variables.append(dependent_variable)
        P_DT, P_FT = self.manager.get_stats()
        self.prob_detect_track.append(P_DT)
        self.prob_false_track.append(P_FT)
        n_scans_true, n_scans_false = self.manager.get_time_info()
        self.average_true_init_time.append(np.mean(n_scans_true))
        self.average_false_init_time.append(np.mean(n_scans_false))

    def update_manager(self, manager):
        self.manager = manager
    
    def plot_pft_pdt(self):
        pft_sorted = []
        pdt_sorted = []
        pft_sort, pdt_sort = zip(*sorted(zip(self.prob_false_track, self.prob_detect_track)))
        for pdt, pft in zip(pdt_sort, pft_sort):
            if pdt > 0 and pft > 0:
                pft_sorted.append(pft)
                pdt_sorted.append(pdt)
        fig, ax = plt.subplots()
        ax.semilogx(pft_sorted, pdt_sorted)
        ax.set_xlabel('$P_{FT}$')
        ax.set_ylabel('$P_{DT}$')
        return fig, ax

    def plot_pft_pdt_dep_var(self):
        fig, ax = plt.subplots(nrows=2)
        dt_ax = ax[0]
        ft_ax = ax[1]
        dt_ax.semilogx(np.array(self.dependent_variables), self.prob_detect_track)
        dt_ax.set_ylabel('$P_{DT}$')
        dt_ax.set_xlabel(self.dep_var_name)
        ft_ax.semilogx(np.array(self.dependent_variables), self.prob_false_track)
        ft_ax.set_xlabel(self.dep_var_name)
        ft_ax.set_ylabel('$P_{FT}$')
        return fig, ax

    def plot_init_time(self):
        fig, ax = plt.subplots(nrows=2)
        true_ax = ax[0]
        false_ax = ax[1]
        true_ax.semilogx(dep, true_t)
        true_ax.set_xlabel(r'$\beta$')
        true_ax.set_ylabel('$T_{DT}$')
        false_ax.semilogx(dep, false_t)
        false_ax.set_xlabel(r'$\beta$')
        false_ax.set_ylabel('$T_{FT}$')
        return fig, ax

class TrackInitMonteCarlo(object):
    # Features: true_target_state is only one target for now.
    def __init__(self, true_target_state, track_manager, measurement_model, N_MC):
        self.true_target_state = true_target_state
        self.track_manager = track_manager
        self.measurement_model = measurement_model
        self.N_MC = N_MC
        self.n_targets = 1 # TODO
        self.all_true_tracks = 0
        self.all_false_tracks = 0
        self.all_targets = 0

    def step(self):
        for n in range(self.N_MC):
            current_true_tracks = 0
            current_false_tracks = 0
            self.track_manager.reset()
            true_target_state = self.true_target_state.get_state()
            time = self.true_target_state.get_time()
            measurements_all = []
            new_tracks_all = dict()
            for state in true_target_state:
                true_pos = np.array([state.est_posterior[0], state.est_posterior[2]])
                measurements = self.measurement_model.generate_measurements([true_pos], state.timestamp)
                measurements_all.append(measurements)
                _, new_tracks = self.track_manager.step(measurements, state.timestamp)
                for tracks in new_tracks:
                    new_tracks_all[tracks[0].track_index] = tracks
            found_true_track = False
            for track_idx, track in new_tracks_all.items():
                if not found_true_track and is_true_track(true_target_state, track):
                    current_true_tracks += 1
                    found_true_track = True
                else:
                    current_false_tracks += 1
            # Update statistics
            self.all_true_tracks += current_true_tracks
            self.all_false_tracks += current_false_tracks
            self.all_targets += 1
        return self.get_pdt_pft()

    def get_pdt_pft(self):
        pdt =  np.float(self.all_true_tracks)/self.all_targets
        pft = np.float(self.all_false_tracks)/(self.all_true_tracks+self.all_false_tracks)
        return pdt, pft

class TrackInitVariableParameters(object):
    def __init__(self, true_target_state, measurement_model, N_MC):
        self.true_target_state = true_target_state
        self.measurement_model = measurement_model
        self.N_MC = N_MC
        self.dependent_variables = []
        self.pdt = []
        self.pft = []

    def step(self, dep_var, manager):
        self.dependent_variables.append(dep_var)
        monte_carlo = TrackInitMonteCarlo(self.true_target_state, manager, self.measurement_model, self.N_MC)
        pdt, pft = monte_carlo.step()
        self.pdt.append(pdt)
        self.pft.append(pft)

    def plot_pdt_pft(self, pdt_ax, pft_ax):
        pdt_ax.semilogx(self.dependent_variables, self.pdt)
        pft_ax.semilogx(self.dependent_variables, self.pft)

    def plot_soc(self, soc_ax):
        soc_ax.semilogx(self.pft, self.pdt)

class ClusterAnalysis(object):
    def __init__(self):
        self.n_confirmed_clusters = 0
        self.n_H1_true = 0
        self.n_data_association_true = 0
        self.n_data_association_true_wo_misdtections = 0
        self.current_track_index = 1
        self.clusters = dict()
        self.cluster_H1_true = dict()
        self.cluster_correct_data_association = dict()

    def test_clusters(self, confirmed_clusters, terminated_clusters, true_target):
        for cluster in confirmed_clusters:
            self.clusters[self.current_track_index] = cluster
            self.n_confirmed_clusters += 1
            # The grand question is wether this should use cluster.root_measurements or cluster.measurements_all. According to VK, it should use the former
            if self.H1_true(cluster.root_measurements, true_target.measurements):
                self.cluster_H1_true[self.current_track_index] = True
                self.n_H1_true += 1
                cluster_track = cluster.get_track()
                track_measurements = set()
                for t in cluster_track:
                    [track_measurements.add(z) for z in t.measurements]
                if self.data_association_true(track_measurements, true_target.measurements):
                    self.n_data_association_true += 1
                if self.data_association_true_wo_misdetections(track_measurements, true_target.measurements):
                    self.n_data_association_true_wo_misdtections += 1
            else:
                self.cluster_H1_true[self.current_track_index] = False
            self.current_track_index += 1

    def H1_true(self, cluster_measurements, target_measurements):
        return len(cluster_measurements.intersection(target_measurements)) > 0

    def data_association_true(self, track_measurements, target_measurements):
        return track_measurements.issubset(target_measurements)

    def data_association_true_wo_misdetections(self, track_measurements, target_measurements):
        unmatched_measurements = track_measurements-target_measurements
        target_as_list = list(target_measurements)
        only_misdetections = True
        for measurement in unmatched_measurements:
            t = measurement.timestamp
            for z in target_as_list:
                if z.timestamp == t:
                    only_misdetections = z.is_zero_measurement()
        return only_misdetections

    def plot_clusters_with_track_init_hyp(self, ax):
        H1_true_tracks = dict()
        H0_true_tracks = dict()
        for cluster_id, cluster in self.clusters.items():
            track = cluster.get_track()
            measurements = cluster.get_measurements()
            color = 'g' if self.cluster_H1_true[cluster_id] else 'r'
            ms = 12 if self.cluster_H1_true[cluster_id] else 14
            for z in measurements:
                ax.plot(z.value[1], z.value[0], 'o', markeredgecolor=color,markerfacecolor='none', markersize=ms)
                ax.text(z.value[1], z.value[0], str(z.timestamp))
            if self.cluster_H1_true[cluster_id]:
                H1_true_tracks[cluster_id] = track
            else:
                H0_true_tracks[cluster_id] = track
        autovis.plot_track_pos(H1_true_tracks, ax, color='g')
        autovis.plot_track_pos(H0_true_tracks, ax, color='r')
        ax.set_xlim(-500, 500)
        ax.set_ylim(-500, 500)
        ax.set_aspect('equal')

    def print_stats(self):
        print "P(accept H1|H1 true)={}".format(self.frac2per(self.n_H1_true, self.n_confirmed_clusters))
        print "P(accept H1|H0 true)={}".format(self.frac2per(self.n_confirmed_clusters-self.n_H1_true, self.n_confirmed_clusters))
        print "P(correct Omega|H1 true and accepted)={}".format(self.frac2per(self.n_data_association_true, self.n_H1_true))
        print "P(correct Omega, except misdetections|H1 true and accepted)={}".format(self.frac2per(self.n_data_association_true_wo_misdtections, self.n_H1_true))

    @staticmethod
    def frac2per(num, denom):
        if denom == 0:
            denom = 1
            num = 0
        return float(num)/denom
