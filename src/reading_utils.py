import sys
import numpy as np
import configparser as cfg


def read_data(file_path):
    """
    Read numerical data from a file and return it as two lists.

    The function expects the file to contain two columns of numerical data,
    separated by whitespace. Each row is read and split into two lists
    based on these columns.

    Parameters:
    - file_path : str
        Path to the data file.

    Returns:
    - column1 : list of float
        Data from the first column.
    - column2 : list of float
        Data from the second column.

    Raises:
    - SystemExit
        If the file is not found or the file format is incorrect.
    """
    try:
        column1, column2 = [], []
        with open(file_path, "r") as file:
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
    """
    Read data from a text file and return two numpy arrays.

    This function uses `read_data` to fetch data from the specified file
    and then converts the data into numpy arrays for further processing.

    Parameters:
    - file_path : str
        Path to the text file containing the data.

    Returns:
    - x_observed : numpy.ndarray
        Numpy array containing data from the first column of the file.
    - I_observed : numpy.ndarray
        Numpy array containing data from the second column of the file.
    """
    data = read_data(file_path)
    x_observed = np.array(data[0], dtype=np.float32)
    I_observed = np.array(data[1], dtype=np.float32)
    return x_observed, I_observed


def read_config(input_file):
    """
    Read configuration settings from a file and return model and sampling parameters.

    The function expects a configuration file in INI format with sections for
    'ModelParameters' and 'SamplingParameters'. It extracts parameters for the model
    and sampling process, providing fallback values if specific entries are missing.

    Parameters:
    - input_file : str
        Path to the configuration file.

    Returns:
    - model_params : dict
        Dictionary containing model parameters with keys 'a', 'b', 'c', and 'd'.
    - sampling_params : dict
        Dictionary containing sampling parameters with keys 'draws', 'tune',
        'chains', and 'target_accept'.
    -seed : int
        Intiger container the unique seed number
    """
    config = cfg.ConfigParser()
    config.read(input_file)

    # Extracting model parameters
    model_params = {
        "a": config.getfloat("ModelParameters", "a", fallback=-5),
        "b": config.getfloat("ModelParameters", "b", fallback=5),
        "c": config.getfloat("ModelParameters", "c", fallback=0),
        "d": config.getfloat("ModelParameters", "d", fallback=8),
    }

    # Extracting sampling parameters
    sampling_params = {
        "draws": config.getint("SamplingParameters", "draws", fallback=30000),
        "tune": config.getint("SamplingParameters", "tune", fallback=800),
        "chains": config.getint("SamplingParameters", "chains", fallback=8),
        "target_accept": config.getfloat(
            "SamplingParameters", "target_accept", fallback=0.8
        ),
    }

    # Extracting seed
    seed = config.getint("General", "seed", fallback=12042000)

    return model_params, sampling_params, seed
