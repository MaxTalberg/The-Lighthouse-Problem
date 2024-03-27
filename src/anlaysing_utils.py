import numpy as np
import arviz as az
import pandas as pd


def cauchy(x, alpha, beta):
    """
    Calculate the Cauchy distribution probability density function (PDF) value.

    Parameters
    ----------
    x : float or array_like
        The points at which the PDF is to be computed.
    alpha : float
        Location parameter of the distribution, which dictates the
        "peak" of the distribution.
    beta : float
        Scale parameter, which dictates the "width" of the distribution.

    Returns
    -------
    float or ndarray
        The PDF values of the Cauchy distribution at the given x points.
    """
    return beta / np.pi * 1 / (beta**2 + (x - alpha) ** 2)


def trigonometric(theta, alpha, beta):
    """
    Compute a trigonometric function based on the tangent of an angle,
    modified by scaling and shifting.

    Parameters
    ----------
    theta : float or array_like
        The angles in radians for which the function is computed.
    alpha : float
        Shift parameter that vertically shifts the function.
    beta : float
        Scale parameter that scales the function vertically.

    Returns
    -------
    float or ndarray
        The computed values of the trigonometric function.
    """
    return beta * np.tan(theta) + alpha


def mean_mle_analysis(seed):
    """
    Perform maximum likelihood estimation (MLE) analysis using trigonometric data
    and the Cauchy distribution.

    This function generates random data based on a trigonometric function, analyses
    it using the Cauchy distribution and calculates the mean and mode of the
    generated data. It also prepares data for histogram representation and the
    true distribution curve.

    Parameters
    ----------
    seed : int, optional
        The random seed for reproducibility.

    Returns
    -------
    x : ndarray
        Random data generated from the trigonometric function.
    x_true : ndarray
        Data points for the x-axis of the true Cauchy distribution curve.
    y_true : ndarray
        Probability density values of the true Cauchy distribution.
    mean : float
        Mean of the generated trigonometric data.
    mode : float
        Mode of the generated trigonometric data.
    bins_number : int
        Recommended number of bins for histogram plotting.

    """
    # Set the seed for reproducibility
    np.random.seed(seed)

    # Set the parameters
    alpha = 0
    beta = 1

    # Generate data for histogram
    theta = np.random.uniform(-np.pi / 2, np.pi / 2, 100000)
    x = trigonometric(theta, alpha, beta)

    # Generate data for the true distribution
    x_true = np.linspace(-20, 20, 1000)
    y_true = cauchy(x_true, alpha, beta)

    # Calculate mean and mode
    mean = np.mean(x)
    mode = np.median(x)

    bins_number = 200

    return x, x_true, y_true, mean, mode, bins_number


def thinning(trace):
    """
    Apply thinning to the provided trace to reduce autocorrelation.

    This function calculates the effective sample size (ESS) for all variables
    in the trace, determines the minimum ESS and computes the thinning interval
    based on the autocorrelation time (tau). It then thins the trace accordingly
    and returns the thinned trace as an InferenceData object.

    Parameters
    ----------
    trace : arviz.InferenceData
        The MCMC trace to be thinned, encapsulated in an ArviZ InferenceData object.

    Returns
    -------
    thinned_trace : arviz.InferenceData
        The thinned trace, encapsulated in an ArviZ InferenceData object.

    """
    # Compute ESS for all variables
    ess_results = az.ess(trace)

    # Find the minimum ESS across all variables
    min_ess = ess_results.to_array().min().values.item()

    # Compute total number of samples (assuming all chains have the same length)
    total_samples = len(trace.posterior.draw) * len(trace.posterior.chain)

    # Calculate tau using the minimum ESS (to be conservative)
    tau = total_samples / min_ess
    print(
        f"The autocorrelation time (tau) based on min ESS is approximately: {tau:.2f}"
    )

    # Calculate thinning interval as the ceiling of tau to be consrtvative
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

    print(
        f"Thinned trace: {total_samples_thinned} samples in total, ",
        "with {num_samples_per_chain} samples per chain across {num_chains} chains.",
    )

    return thinned_trace


def convergence_diagnostic(thinned_trace):
    """
    Evaluate convergence diagnostics for a thinned MCMC trace.

    This function calculates convergence diagnostics for each parameter in the
    thinned trace, including the mean, standard error of the mean,
    standard deviation, standard error of the standard deviation,
    autocorrelation time (tau) and the Gelman-Rubin statistic (r_hat).
    The diagnostics are organised and displayed in a pandas DataFrame.

    Parameters
    ----------
    thinned_trace : arviz.InferenceData
        The thinned MCMC trace for which convergence diagnostics are to be
        calculated, encapsulated in an ArviZ InferenceData object.

    Returns
    -------
    diagnostic_df : pandas.DataFrame
        A DataFrame containing the convergence diagnostics for each parameter in
        the thinned trace, including mean, standard error of the mean (SE_mean),
        standard deviation (sd), standard error of the standard deviation (SE_sd),
        autocorrelation time (tau) and the Gelman-Rubin statistic (r_hat).

    Notes
    -----
    The function prints the DataFrame as a table for quick inspection.
    """

    # Thinned trace details
    num_chains = len(thinned_trace.posterior.chain)
    num_samples_per_chain = len(thinned_trace.posterior.draw)
    total_samples_thinned = num_chains * num_samples_per_chain

    # Compute the mean and standard deviation for each parameter
    summary_stats = az.summary(thinned_trace, round_to=2)

    # Create a DataFrame to hold the results
    diagnostic_df = pd.DataFrame(
        {
            "mean": summary_stats["mean"],
            "SE_mean": summary_stats["sd"] / np.sqrt(total_samples_thinned),
            "sd": summary_stats["sd"],
            "SE_sd": summary_stats["sd"] / np.sqrt(2 * total_samples_thinned),
            "tau": summary_stats["ess_bulk"] / summary_stats["ess_mean"],
            "r_hat": summary_stats["r_hat"],
        }
    )

    # Print the DataFrame as a table
    print(diagnostic_df)


def appendix_data(trace):
    """
    Print a summary of the MCMC trace data.

    This function uses ArviZ's summary function to compute and display a
    summary of the MCMC trace data, including the mean, standard deviation
    and the effective sample size for each parameter, among other statistics.

    Parameters
    ----------
    trace : arviz.InferenceData
        The MCMC trace data encapsulated in an ArviZ InferenceData object.

    Notes
    -----
    The summary is printed directly to the console.
    """
    # Print summary statistics of the trace
    print(az.summary(trace, round_to=2))
