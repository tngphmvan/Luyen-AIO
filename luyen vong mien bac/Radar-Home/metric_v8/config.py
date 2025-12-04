radar_configs = {
    'c': 3e8, 
    'startFreq': 77e9, 
    'Tr': 60e-6,                    # Sweeping frequency time
    'Idle_time': 100e-6,            # free time
    'Fs': 10e6,                     # Sampling frequency
    'Slope': 29.982e12,             # chirp slope
    'Bandwidth': 60e-6 * 29.982e12, # Transmission signal bandwidth
    'BandwidthValid': 0.767539200e9,# Effective bandwidth of the transmitted signal
    'range_size': 256,              # range size
    'azimuth_size': 181,            # azimuth size
    'elevation_size': 181,          # elevation size
    'crop_num': 3,                  # crop some indices in range domain
    'n_chirps': 128,                # number of chirps in one frame
    'min_azimuth': -90,             # min radar azimuth
    'max_azimuth': 90,              # max radar azimuth
    'min_elevation': -90,           # min radar elevation
    'max_elevation': 90,            # max radar elevation    
    'min_range': 1.0,               # min radar range
    'max_range': 25.0,              # max radar range
    'range_res': 3e8/(2*0.767539200e9), 
    'angle_res': 1
}

dimssnet_configs = {
    'n_epoch': 10,
    'batch_size': 2,
    'learning_rate': 1e-5,
    'range_size': 50,
    'azimuth_size': 181,
    'elevation_size': 181,
    'doppler_size': 181, 
    'min_azimuth': -90,             # min radar azimuth
    'max_azimuth': 90,              # max radar azimuth
    'min_elevation': -90,           # min radar elevation
    'max_elevation': 90,            # max radar elevation    
    'min_range': 1.0,               # min radar range
    'max_range': 25.0,              # max radar range
}

n_class = 2
class_table = {
    0: 'background', 
    1: 'object'
}