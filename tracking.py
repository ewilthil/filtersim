import numpy as np
from scipy.stats import multivariate_normal, chi2
from scipy.linalg import block_diag

H = np.array([[1, 0, 0, 0],[0, 0, 1, 0]])

class Measurement(object):
    def __init__(self, value, timestamp, covariance, measurement_matrix=H):
        self.timestamp = timestamp
        self.value = value
        self.covariance = covariance*np.identity(2)
        self.measurement_matrix = measurement_matrix

    def __repr__(self):
        return "%.2f, %.2f" % (self.value[0], self.value[1])

class MeasurementModel(object):
    def __init__(self, target_cov, clutter_density, x_lims, y_lims=None, P_D=1):
        self.measurement_covariance = target_cov
        self.measurement_matrix = H
        self.density = clutter_density
        self.detection_probability = P_D
        self.x_lims = x_lims
        if y_lims is None:
            self.y_lims = x_lims
        else:
            self.y_lims = y_lims
        self.area = np.diff(self.x_lims)*np.diff(self.y_lims)

    def generate_measurements(self, true_targets, timestamp):
        N_targets = len(true_targets)
        target_measurements = []
        for target in true_targets:
            detected = self.detection_probability > np.random.rand()
            if detected:
                noise = multivariate_normal.rvs(mean=np.zeros(2), cov=self.measurement_covariance)
                target_measurements.append(Measurement(target, timestamp, self.measurement_covariance))
        N_clutter = np.random.poisson(self.density*self.area)
        x_coords = np.random.uniform(self.x_lims[0], self.x_lims[1], N_clutter)
        y_coords = np.random.uniform(self.y_lims[0], self.y_lims[1], N_clutter)
        clutter_points = [np.hstack((x_coords[i], y_coords[i])) for i in range(N_clutter)]
        clutter_measurements = [Measurement(clutter_points[i], timestamp, self.measurement_covariance) for i in range(N_clutter)]
        return target_measurements+clutter_measurements

class Estimate(object):
    def __init__(self, t, mean, covariance, is_posterior=False, track_index=None):
        # If it is a posterior, it should be set accordingly
        self.timestamp = t
        self.measurements = []
        self.est_prior = mean
        self.cov_prior = covariance
        if is_posterior:
            self.est_posterior = mean
            self.cov_posterior = covariance
        if track_index is not None:
            self.track_index = track_index
        else:
            self.track_index = -1
    def __repr__(self):
        ID_str = "Track ID: %d" % (self.track_index)
        timestamp_str = "Timestamp: %.2f" % self.timestamp
        return ID_str+", "+timestamp_str

    def inside_gate(self, measurement):
        z = measurement.value
        H = measurement.measurement_matrix
        R = measurement.covariance
        z_hat = H.dot(self.est_prior)
        S = H.dot(self.cov_prior).dot(H.T)+R
        nu = z-z_hat
        nis = np.dot(nu.T, np.dot(np.linalg.inv(S), nu))
        inside_gate = nis.squeeze() < tracking_parameters['gamma']
        return inside_gate

    def store_measurement(self, measurement):
        self.measurements.append(measurement)

    def step_measurement(self):
        if len(self.measurements) > 0:
            self.pdaf_step()
        else:
            self.trivial_step()

    def trivial_step(self):
        self.est_posterior = self.est_prior
        self.cov_posterior = self.cov_prior

    @classmethod
    def from_estimate(cls, timestamp, old_estimate, target_model, u):
        dt = timestamp - old_estimate.timestamp
        F, B, Q = target_model.discretize_system(dt)
        mean = F.dot(old_estimate.est_posterior)+B.dot(u)
        cov = F.dot(old_estimate.cov_posterior).dot(F.T)+Q
        return cls(timestamp, mean, cov, track_index=old_estimate.track_index)

    @classmethod
    def from_measurement(cls, old_measurement, new_measurement):
        H = old_measurement.measurement_matrix
        R = old_measurement.covariance
        t1 = old_measurement.timestamp
        t2 = new_measurement.timestamp
        dt = t2-t1
        F = np.identity(4)
        F[0,1] = dt
        F[2,3] = dt
        H_s = np.vstack((H, np.dot(H,F)))
        z_s = np.hstack((old_measurement.value, new_measurement.value))
        R_s = block_diag(R, R)
        S_s = np.dot(H_s.T, np.dot(np.linalg.inv(R_s), H_s))
        S_s_inv = np.linalg.inv(S_s)
        est_x1 = np.dot(np.dot(S_s_inv, np.dot(H_s.T, np.linalg.inv(R_s))), z_s)
        est_x2 = np.dot(F, est_x1)
        cov_x1 = S_s_inv
        cov_x2 = np.dot(F, np.dot(S_s_inv, F.T))
        est_1 = cls(t1, est_x1, cov_x1, is_posterior=True)
        est_2 = cls(t2, est_x2, cov_x2, is_posterior=True)
        est_1.store_measurement(old_measurement)
        est_2.store_measurement(new_measurement)
        return est_1, est_2

class ProbabilisticDataAssociation(object):
    def __init__(self, measurement_model, P_G):
        self.P_D = measurement_model.detection_probability
        self.P_G = P_G
        self.gamma = chi2(df=2).ppf(P_G)

    def __repr__(self):
        pass

    def calculate_posterior(self, estimate):
        measurements = estimate.measurements
        N_measurements = len(measurements)
        if N_measurements is 0:
            estimate.est_posterior = estimate.est_prior
            estimate.cov_posterior = estimate.cov_prior
        else:
            H = measurements[0].measurement_matrix
            R = measurements[0].covariance
            z_hat = H.dot(estimate.est_prior)
            innovations = [z.value-z_hat for z in measurements]
            b = 2/self.gamma*N_measurements*(1-self.P_D*self.P_G)/self.P_D
            e = np.zeros(N_measurements)
            S = H.dot(estimate.cov_prior).dot(H.T)+R
            for i, nu in enumerate(innovations):
                e[i] = np.exp(-0.5*nu.dot(np.linalg.inv(S)).dot(nu))
            betas = np.hstack((e, b))
            betas = betas/(1.*np.sum(betas))
            gain = np.dot(estimate.cov_prior, np.dot(H.T, np.linalg.inv(S)))
            total_innovation = np.zeros(2)
            cov_terms = np.zeros((2,2))
            for i in range(N_measurements):
                innov = innovations[i]
                total_innovation += betas[i]*innov
                innov_vec = innov.reshape((2,1))
                cov_terms += betas[i]*np.dot(innov_vec, innov_vec.T)
            estimate.est_posterior = estimate.est_prior+np.dot(gain, total_innovation)
            total_innovation_vec = total_innovation.reshape((2,1))
            cov_terms = cov_terms-np.dot(total_innovation_vec, total_innovation_vec.T)
            soi = np.dot(gain, np.dot(cov_terms, gain.T))
            P_c = estimate.cov_prior-np.dot(gain, np.dot(S, gain.T))
            cov_posterior = betas[-1]*estimate.cov_prior+(1-betas[-1])*P_c+soi
            estimate.cov_posterior = 0.5*(cov_posterior+cov_posterior.T)

def gate_measurements(measurement_list, estimate, gate_probability):
    if len(measurement_list) > 0:
        is_gated = [False for _ in measurement_list]
        # Assume all the measurement have the same mapping and covariance
        H = measurement_list[0].measurement_matrix
        R = measurement_list[0].covariance
        S = H.dot(estimate.cov_prior).dot(H.T)+R
        gamma = chi2(df=2).ppf(gate_probability)
        z_hat = H.dot(estimate.est_prior)
        for idx, measurement in enumerate(measurement_list):
            z = measurement.value
            nu = z-z_hat
            nu_vec = nu.reshape((2,1))
            NIS = nu_vec.T.dot(np.linalg.inv(S).dot(nu_vec))
            if NIS < gamma:
                estimate.measurements.append(measurement)
                is_gated[idx] = True
        return is_gated
    else:
        return []
