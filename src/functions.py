import sys
import numpy as np
import arviz as az
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

def thinning(trace):

    # Compute ESS for all variables
    ess_results = az.ess(trace)

    # Find the minimum ESS across all variables
    min_ess = ess_results.to_array().min().values.item()

    # Compute total number of samples (assuming all chains have the same length)
    total_samples = len(trace.posterior.draw) * len(trace.posterior.chain)

    # Calculate tau using the minimum ESS (to be conservative)
    tau = total_samples / min_ess
    print(f"The autocorrelation time (tau) based on min ESS is approximately: {tau:.2f}")

    # Calculate thinning interval as the ceiling of tau to ensure it's an integer to be consrtvative
    thinning_interval = int(np.ceil(tau))
    print(f"Thinning interval: {thinning_interval}")

    # Thin the trace by slicing with the thinning interval using xarray's isel method
    thinned_posterior = trace.posterior.isel(draw=slice(None, None, thinning_interval))

    # Create a new InferenceData object with the thinned posterior
    thinned_trace = az.InferenceData(posterior=thinned_posterior)
    
    return thinned_trace

