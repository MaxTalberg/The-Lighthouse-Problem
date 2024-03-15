import sys
import numpy as np

import configparser as cfg

def read_data(file_path):
    try:
        column1, column2 = [], []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.split()
                column1.append(float(parts[0]))
                column2.append(float(parts[1]))
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        sys.exit(1)
    except IndexError:
        print(f"Error: Incorrect file format in {file_path}.")
        sys.exit(1)

    return column1, column2

def read_and_prepare_data(file_path):
    '''
    Function to read data from a text file and return it as two numpy arrays.
    '''
    data = read_data(file_path)
    x_observed = np.array(data[0], dtype=np.float32)
    I_observed = np.array(data[1], dtype=np.float32)
    return x_observed, I_observed

def read_config(input_file):
    config = cfg.ConfigParser()
    config.read(input_file)
    
    # Extracting model parameters
    model_params = {
        'a': config.getfloat('ModelParameters', 'a', fallback=-5),
        'b': config.getfloat('ModelParameters', 'b', fallback=5),
        'c': config.getfloat('ModelParameters', 'c', fallback=0),
        'd': config.getfloat('ModelParameters', 'd', fallback=8),
    }

    # Extracting sampling parameters
    sampling_params = {
        'draws': config.getint('SamplingParameters', 'draws', fallback=30000),
        'tune': config.getint('SamplingParameters', 'tune', fallback=800),
        'chains': config.getint('SamplingParameters', 'chains', fallback=8),
        'target_accept': config.getfloat('SamplingParameters', 'target_accept', fallback=0.8)
    }
    return model_params, sampling_params
