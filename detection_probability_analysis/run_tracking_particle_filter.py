from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(12345) # The seed 12346 may have bad initialization / measurements?

import autoseapy.visualization as autovis
import generate_scenario as setup
import setup_trackers
import autoseapy.track_management as automanager
import autoseapy.particle_filters as autopart
import analyse_tracks

def plot_particle(ax, particle):
    vel_arrow_scale = 0.1*setup.radar_range/setup_trackers.maximum_velocity
    #data = ax.arrow(particle.state[2], particle.state[0], vel_arrow_scale*particle.state[3], vel_arrow_scale*particle.state[1])
    ax.plot(particle.state[2], particle.state[0], '.', ms=1,color='b')

def plot_particles(ax, particles, track_file, measurements):
    autovis.plot_measurements(measurements, ax)
    autovis.plot_track_pos(track_file, ax, color='r')
    [plot_particle(ax, particle) for particle in particles]
    ax.set_xlim(-setup.radar_range, setup.radar_range)
    ax.set_ylim(-setup.radar_range, setup.radar_range)
    ax.set_aspect('equal')



if __name__ == '__main__':
    N_MC = 1
    titles =['PF tracker']





    for n_mc in range(N_MC):
        initial_distribution = autopart.UniformSampleDistribution(max_pos=setup.radar_range, max_vel=setup_trackers.maximum_velocity)
        transition_distribution = autopart.NearlyConstantVelocityDistribution(q=setup_trackers.target_process_noise_covariance)
        measurement_distribution = autopart.CartesianMeasurementDistribution(r=setup.measurement_covariance_single_axis)
        measurement_originated_distribution = autopart.UniformSampleDistribution(max_pos=25, max_vel=setup_trackers.maximum_velocity)
        pf_tracker = autopart.ParticleFilterTracker(
                initial_distribution,
                transition_distribution,
                measurement_distribution,
                measurement_originated_distribution,
                num_particles=5000,
                num_birth_particles=1000,
                num_measurement_particles=250,
                degeneracy_threshold=0.5,
                initial_empty_weight=0.9,
                detection_probability=setup.PD_high,
                survival_probability=0.99,
                birth_probability=0.01,
                clutter_density=setup.clutter_density,
                track_existence_threshold=0.9
                )

        pf_manager = automanager.ParticleFilterManager(pf_tracker)
        current_managers = [pf_manager]
        movie_writer = autovis.TrackMovie('movies/pf_tracker_{}'.format(n_mc))
        true_targets, true_detectability, measurements_all, measurement_timestamps = setup.generate_scenario([setup.munkholmen_mmsi], tmax=180)
        existence_probability = np.zeros_like(measurement_timestamps)
        misdetection_timestamps = []
        [manager.reset() for manager in current_managers]
        previous_ext_prob = 1-pf_tracker.initial_empty_weight
        for n_time, timestamp in enumerate(measurement_timestamps):
            measurements = measurements_all[n_time]
            for manager, title in zip(current_managers, titles):
                debug_data = manager.step(measurements, timestamp)
                current_ext_prob = 1-pf_tracker.empty_weight
                existence_probability[n_time] = current_ext_prob
                if len(true_targets[setup.munkholmen_mmsi][n_time].measurements) == 0:
                    print "misdetection, target existence probability is {} ({})".format(current_ext_prob, current_ext_prob-previous_ext_prob)
                else:
                    print "detection, target existence probability is {} ({})".format(current_ext_prob, current_ext_prob-previous_ext_prob)
                previous_ext_prob = 1-pf_tracker.empty_weight
            current_particles = debug_data['current_particles']
            discarded_particles = debug_data['discarded_particles']
            movie_writer.grab_frame(plot_particles, {'particles' : current_particles, 'track_file' : pf_manager.track_file, 'measurements' : measurements_all[:n_time+1]})
        movie_writer.save_movie()
        fig, ax = plt.subplots(figsize=(8,8))
        autovis.plot_measurements(measurements_all, ax)
        autovis.plot_track_pos(true_targets, ax, color='k')
        autovis.plot_track_pos(pf_manager.track_file, ax, color='r')
        ax.set_xlim(-setup.radar_range, setup.radar_range)
        ax.set_ylim(-setup.radar_range, setup.radar_range)
        ax.set_aspect('equal')
        ext_fig, ext_ax = plt.subplots()
        ext_ax.plot(measurement_timestamps, existence_probability, 'k--')
        for track_id, est_list in pf_manager.track_file.items():
            timestamps = [est.timestamp for est in est_list]
            ext_prob = [est.existence_probability for est in est_list]
            ext_ax.plot(timestamps, ext_prob, 'k-')
        for true_state in true_targets[setup.munkholmen_mmsi]:
            if len(true_state.measurements) == 0:
                ext_ax.plot([true_state.timestamp, true_state.timestamp] ,[0, 1], 'k:')
        ext_ax.set_ylim(0, 1)
        print "n_mc={}".format(n_mc)
    plt.show()
