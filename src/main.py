import time
import pymc3 as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from functions import read_data
from plotsiii import plotiii1, plotiii2
from plotsv import plotv1, plotv2


if __name__ == '__main__':
    ### part iii
    #plotiii1()
    #plotiii2()

    ### part v

    # set seed
    np.random.seed(120420)

    # read the data
    file_path = 'lighthouse_flash_data.txt'

    # Observed flashes and intensities
    x_observed = np.array(read_data(file_path)[0], dtype=np.float32)
    I_observed = np.array(read_data(file_path)[1], dtype=np.float32)

    # Initialise values for HMC MC
    num_chains = 8
    num_samples = 30000
    num_burnin = 800
    target_accept = 0.65

    # Constants for uniform priors
    a, b = -5, 5  # Bounds for alpha
    c, d = 0, 8 # Bounds for beta

    # Define the model
    with pm.Model() as model:
        # Uniform priors for alpha and beta
        alpha = pm.Uniform('alpha', lower=a, upper=b)
        beta = pm.Uniform('beta', lower=c, upper=d)

        # Cauchy likelihood of observations
        Y_obs = pm.Cauchy('Y_obs', alpha=alpha, beta=beta, observed=x_observed)

    # Smple from the model
    t_start = time.time()
    with model:
        # The default sampler is NUTS, but we'll specify it explicitly for clarity
        trace = pm.sample(draws=30000, tune=800, chains=num_chains, target_accept=0.65)
    t_end = time.time()

    # Extract the samples
    alpha_samples = trace.get_values('alpha')
    beta_samples = trace.get_values('beta')

    az.plot_trace(trace)
    plt.show()