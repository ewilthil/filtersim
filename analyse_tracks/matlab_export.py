from autoseapy.bag_operations import bag2raw_data
from autoseapy.tracking import sync_estimates_to_timestamp
from autoseapy.ais import known_mmsi, known_mmsi_rev
from scipy.io import savemat
bag = 'final_demo_filtered.bag'
ais_all, measurements_all = bag2raw_data(bag)
timestamps = []
for measurements in measurements_all:
    measurement = measurements.pop()
    timestamps.append(measurement.timestamp)
    measurements.add(measurement)
ais_selected = [known_mmsi[target] for target in ['TELEMETRON', 'TRESFJORD', 'TRONDHEIMSFJORD II', 'KSX_OSD1']]
data_out = dict()
for mmsi in ais_all:
    if mmsi in known_mmsi_rev.keys():
        callsign = known_mmsi_rev[mmsi].replace(' ', '_')
    else:
        callsign = "target_{}".format(mmsi)
    data_out[callsign] = []
    for t in timestamps:
        synced_estimate = sync_estimates_to_timestamp(t, ais_all[mmsi], 0.01**2)
        data_out[callsign].append(synced_estimate.est_posterior)
measurement_values = []
for measurements in measurements_all:
    measurement_values.append([z.value for z in measurements])
data_out['timestamps'] = timestamps
data_out['measurements'] = measurement_values
savemat('final_demo', data_out)
