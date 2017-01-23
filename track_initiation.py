import numpy as np
from filtersim.tracking import Estimate

class m_of_n(object):
    def __init__(self, M, N, max_vel):
        self.initiators = []
        self.min_available_track_idx = 1 # 0 is for ownship (in future stuff)
        self.preliminary_indices = dict()
        self.max_vel = max_vel
        self.M = M
        self.N = N

    def __repr__(self):
        pass

    def form_new_tracks(self, measurements):
        new_initiators = []
        new_estimates = []
        for measurement in measurements:
            associated = False
            for init in self.initiators:
                dt = measurement.timestamp-init.timestamp
                vel = (measurement.value-init.value)/(1.*dt)
                if np.linalg.norm(vel) < self.max_vel:
                    associated = True
                    est_1, est_2 = Estimate.from_measurement(init, measurement)
                    est_1.track_index = self.min_available_track_idx
                    est_2.track_index = self.min_available_track_idx
                    self.min_available_track_idx += 1
                    new_estimates.append(est_2)
            if not associated:
                new_initiators.append(measurement)
        self.initiators = new_initiators
        return new_estimates
                    

    def update_track_status(self, current_preliminary_estimates):
        confirmed_estimates = []
        preliminary_estimates = []
        for est in current_preliminary_estimates:
            if est.track_index not in self.preliminary_indices.keys():
                self.preliminary_indices[est.track_index] = (0,0)
                preliminary_estimates.append(est)
            else:
                m, n = self.preliminary_indices[est.track_index]
                if m < self.M and n < self.N:
                    preliminary_estimates.append(est)
                elif m >= self.M and n <= self.N:
                    confirmed_estimates.append(est)
        return confirmed_estimates, preliminary_estimates
