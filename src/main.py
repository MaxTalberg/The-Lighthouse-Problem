import time
import warnings
import pymc3 as pm
import numpy as np
import arviz as az
import theano.tensor as tt
import matplotlib.pyplot as plt
from Coursework.src.processing_utils import read_and_prepare_data, read_config
from plotsiii import plotiii1, plotiii2
from plots import trace_plot
from sampling import define_model_x, define_model_xi, sample_model


warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered in _beta_ppf')


if __name__ == '__main__':

    # Read and prepare the data
    x_observed, I_observed = read_and_prepare_data('lighthouse_flash_data.txt')

    # Read the configuration file
    model_params, sampling_params = read_config('parameters.ini')

    # Define the models
    model_x = define_model_x(x_observed, **model_params)
    model_xi = define_model_xi(x_observed, I_observed, **model_params)

    # Sample the models
    trace_x = sample_model(model_x, **sampling_params)
    #trace_xi = sample_model(model_xi, **sampling_params)

    # Trace plot
    trace_plot(trace_x)
    #trace_plot(trace_xi)

    # Thinning

