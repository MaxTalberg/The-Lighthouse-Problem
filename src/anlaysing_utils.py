import sys
import numpy as np
import arviz as az
import pandas as pd
import configparser as cfg

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

def convergence_diagnostic(thinned_trace):

    # Compute the mean and standard deviation for each parameter
    summary_stats = az.summary(thinned_trace, round_to=2)

    # Create a DataFrame to hold the results
    diagnostic_df = pd.DataFrame({
        'mean': summary_stats['mean'],
        'sd': summary_stats['sd'],
        'tau': summary_stats['ess_bulk'] / summary_stats['ess_mean'],
        'r_hat': summary_stats['r_hat']
    })

    # Print the DataFrame as a table
    print(diagnostic_df)
