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

    # Thinned trace
    num_chains = len(thinned_trace.posterior.chain)
    num_samples_per_chain = len(thinned_trace.posterior.draw)
    total_samples_thinned = num_chains * num_samples_per_chain

    print(f"Thinned trace: {total_samples_thinned} samples in total, with {num_samples_per_chain} samples per chain across {num_chains} chains.")

    return thinned_trace

def convergence_diagnostic(thinned_trace):

    # Thinned trace
    num_chains = len(thinned_trace.posterior.chain)
    num_samples_per_chain = len(thinned_trace.posterior.draw)
    total_samples_thinned = num_chains * num_samples_per_chain

    # Uncertainty on mean

    # Compute the mean and standard deviation for each parameter
    summary_stats = az.summary(thinned_trace, round_to=2)

    # Create a DataFrame to hold the results
    diagnostic_df = pd.DataFrame({
        r'$\mu': summary_stats['mean'],
        r'SE_$\mu$s': summary_stats['sd'] / np.sqrt(num_samples_per_chain),
        r'$\sigma$': summary_stats['sd'],
        r'$\SE_{\sigma}$': summary_stats['sd'] / np.sqrt(2 * (num_samples_per_chain-1)),
        'tau': summary_stats['ess_bulk'] / summary_stats['ess_mean'],
        'r_hat': summary_stats['r_hat']
    })

    # Print the DataFrame as a table
    print(diagnostic_df)
