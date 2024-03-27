import corner
import numpy as np
import arviz as az
from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter
from reading_utils import read_config


# Read the configuration file
model_params, sampling_params, seed = read_config("parameters.ini")
params = model_params["a"], model_params["b"], model_params["c"], model_params["d"]
a, b, c, d = params


def plot_cauchy(cauchy):
    """
    Plot several Cauchy distributions with different parameters.

    This function plots three Cauchy distributions over a range of x values,
    each with different alpha (location) and beta (scale) parameters. The purpose
    is to illustrate the effect of these parameters on the shape of the distribution.

    Parameters:
    - cauchy : function
        The Cauchy distribution function, which should accept x, alpha, and beta as
        arguments.

    Notes:
    - The x range for plotting is fixed between -10 and 10.
    - The function uses Matplotlib for plotting and displays the plot directly.
    """
    # generate points for cauchy distributions
    x = np.linspace(-10, 10, 1000)
    y1 = cauchy(x, alpha=0, beta=0.5)
    y2 = cauchy(x, alpha=0, beta=1)
    y3 = cauchy(x, alpha=-2, beta=2)

    plt.plot(x, y1, label=r"$\alpha=0, \beta=0.5$")
    plt.plot(x, y2, label=r"$\alpha=0, \beta=1$")
    plt.plot(x, y3, label=r"$\alpha=-2, \beta=2$")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("P(x)")
    plt.xlim(-5, 5)

    plt.tight_layout()
    plt.show()


def plot_cauchy_analysis(x, x_true, y_true, mean, mode, bins_number):
    """
    Plot the results of Cauchy distribution analysis including a histogram
    of sampled data, the true PDF, and lines indicating
    the sample mean and median.

    Parameters:
    - x : array_like
        Sampled data points used to create the histogram.
    - x_true : array_like
        Data points for the x-axis of the true Cauchy distribution curve.
    - y_true : array_like
        Probability density values of the true Cauchy distribution for
        each x_true point.
    - mean : float
        The calculated mean of the sampled data.
    - mode : float
        The calculated mode (approximated here as the median) of the sampled data.
    - bins_number : int
        The number of bins to use in the histogram.

    Notes:
    - The histogram range is fixed between -20 and 20, with density normalised.
    - The x-axis is limited between -10 and 10 for clearer visualisation.
    - The function uses Matplotlib for plotting and displays the plot directly.
    """
    # Create the histogram with the new number of bins
    _, _, _ = plt.hist(
        x, bins=bins_number, density=True, color="blue", alpha=0.7, range=(-20, 20)
    )

    # Add the Cauchy distribution PDF
    plt.plot(x_true, y_true, "r", label="True PDF")

    # Indicate the mean and median (mode)
    plt.axvline(
        mean, color="magenta", linestyle="dashed", linewidth=1.5, label="Sample Mean"
    )
    plt.axvline(
        mode, color="orange", linestyle="dashed", linewidth=1.5, label="Sample Median"
    )

    # Limit the x-axis to better visualise the peak
    plt.xlim(-10, 10)

    # Add a legend to the plot
    plt.legend()

    plt.xlabel("x")
    plt.ylabel("P(x)")

    plt.tight_layout()
    plt.show()


def trace_plot(trace, figsize=(12, 8)):
    """
    Generate a trace plot for each variable in the provided MCMC trace.

    This function plots the sampling paths (traces) for each variable in the
    given trace, allowing for the visualisation of the sampling behavior over
    iterations. It's useful for diagnosing the mixing and convergence of the
    chains.

    Parameters:
    - trace : arviz.InferenceData
        The trace data from which to generate the trace plots, encapsulated in an
        ArviZ InferenceData object.
    - figsize : tuple, optional
        The size of the figure to create. Default is (12, 8).

    Notes:
    - The function assumes the possibility of multiple chains and combines them for
    plotting.
    - The y-axis labels are set based on the index of the variable in the trace.
    - The x-axis represents the iteration number.
    - The function uses Matplotlib for plotting and displays the plot directly.
    """
    # Extract variable names directly from the InferenceData object
    var_names = list(trace.posterior.data_vars)

    n_vars = len(var_names)
    _, axes = plt.subplots(n_vars, 1, sharex=True, figsize=figsize, squeeze=False)

    for i, var_name in enumerate(var_names):
        # Extract samples and combine chains
        samples = trace.posterior[var_name].values

        # Plot trace for each chain
        for chain_samples in samples:
            axes[i, 0].plot(chain_samples, lw=0.15, alpha=0.7)

        if i == 0:
            axes[i, 0].set_ylabel(r"alpha, $\alpha$")
        elif i == 1:
            axes[i, 0].set_ylabel(r"beta, $\beta$")
        elif i == 2:
            axes[i, 0].set_ylabel(rf"$I_{0}$")

    # Set common labels
    axes[-1, 0].set_xlabel("Iteration")

    plt.tight_layout()
    plt.show()


def joint_posterior_x(trace):
    """
    Plot the joint distribution of 'alpha' and 'beta' samples from a trace,
    including marginal histograms.

    This function creates a hexbin scatter plot of the joint distribution of
    'alpha' and 'beta' parameters, with marginal histograms for each parameter.
    It's useful for visualising the relationship between the
    two parameters and their individual distributions.

    Parameters:
    - trace : arviz.InferenceData
        The MCMC trace data, encapsulated in an ArviZ InferenceData object,
        from which 'alpha' and 'beta' samples are extracted.

    Notes:
    - The function automatically determines axis limits based on the data.
    - Histograms are normalised to represent density.
    - The function uses Matplotlib for plotting and displays the plot directly.
    """
    # Unpack trace
    alpha_samples = trace.posterior["alpha"].values.flatten()
    beta_samples = trace.posterior["beta"].values.flatten()

    nullfmt = NullFormatter()

    # Define axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # Start figure
    _ = plt.figure(figsize=(9.5, 9))

    # Scatter plot
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # No labels for histograms
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # Scatter plot with hexbin
    _ = axScatter.hexbin(
        alpha_samples, beta_samples, gridsize=100, cmap="inferno", bins="log"
    )

    # Set axis labels
    axScatter.set_xlabel(r"alpha, $\alpha$")
    axScatter.set_ylabel(r"beta, $\beta$")

    # Automatically determine nice limits
    axScatter.set_xlim((alpha_samples.min(), alpha_samples.max()))
    axScatter.set_ylim((beta_samples.min(), beta_samples.max()))

    # Marginal histograms
    axHistx.hist(alpha_samples, bins=50, density=True, alpha=0.6)
    axHisty.hist(
        beta_samples, bins=50, orientation="horizontal", density=True, alpha=0.6
    )

    # Set histogram limits to match the scatter plot
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    plt.tight_layout()
    plt.show()


def joint_posterior_xi(trace):
    """
    Creates a series of 2D hexbin plots for each pair of variables in the trace,
    along with marginal histograms.

    This function visualises the joint distributions between pairs of variables
    ('alpha', 'beta', and 'I0') from the trace using hexbin plots.
    It also includes histograms for the marginal distributions of each variable.

    Parameters:
    - trace : arviz.InferenceData
        The MCMC trace data, encapsulated in an ArviZ InferenceData object.

    Notes:
    - The function uses Matplotlib for plotting and displays the plot directly.
    - Histograms for each variable are plotted along the diagonal of the subplot grid.
    - Empty subplots are hidden for aesthetic reasons.
    """
    # unpack trace
    alpha_samples = trace.posterior["alpha"].values.flatten()
    beta_samples = trace.posterior["beta"].values.flatten()
    I0_samples = trace.posterior["I0"].values.flatten()

    # Start figure
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))

    # Plot alpha vs beta
    axes[1, 0].hexbin(
        alpha_samples, beta_samples, gridsize=50, cmap="inferno", bins="log"
    )
    axes[1, 0].set_xlabel(r"alpha, $\alpha$")
    axes[1, 0].set_ylabel(r"beta, $\beta$")

    # Plot alpha vs I0
    axes[2, 0].hexbin(
        alpha_samples, I0_samples, gridsize=50, cmap="inferno", bins="log"
    )
    axes[2, 0].set_xlabel(r"alpha, $\alpha$")
    axes[2, 0].set_ylabel("I0")

    # Plot beta vs I0
    axes[2, 1].hexbin(beta_samples, I0_samples, gridsize=50, cmap="inferno", bins="log")
    axes[2, 1].set_xlabel(r"beta, $\beta$")
    axes[2, 1].set_ylabel(rf"$I_{0}$")

    # Histograms for alpha, beta, and I0
    axes[0, 0].hist(alpha_samples, bins=50, orientation="vertical", alpha=0.6)

    axes[1, 1].hist(beta_samples, bins=50, orientation="vertical", alpha=0.6)

    axes[2, 2].hist(I0_samples, bins=50, orientation="vertical", alpha=0.6)

    # Hide the empty subplots
    axes[0, 1].axis("off")
    axes[0, 2].axis("off")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()


def marginal_posterior(trace, bins=50, figsize=(12, 8)):
    """
    Plot the marginal posterior distributions for each variable in the trace,
    including mean and standard deviation markers.

    This function generates histograms for the marginal posterior distributions
    of each variable in the trace. It also marks the mean and standard deviation
    (mean ± sd) on the histograms to provide summary statistics visually.

    Parameters:
    - trace : arviz.InferenceData
        The MCMC trace data, encapsulated in an ArviZ InferenceData object.
    - bins : int, optional
        The number of bins to use for the histograms. Default is 50.
    - figsize : tuple, optional
        The size of the figure to create. Default is (12, 8).

    Notes:
    - The function uses Matplotlib for plotting and displays the plot directly.
    - Summary statistics are obtained using ArviZ's summary function.
    """
    var_names = list(trace.posterior.data_vars)

    # Obtain summary statistics using ArviZ
    summary = az.summary(trace, var_names=var_names, round_to=2)

    # Create subplots
    fig, axes = plt.subplots(len(var_names), 1, figsize=figsize, squeeze=False)

    for i, var_name in enumerate(var_names):
        # Extract samples
        samples = trace.posterior[var_name].values.flatten()

        # Extract summary statistics
        mean = summary.loc[var_name, "mean"]
        sd = summary.loc[var_name, "sd"]

        # Plot histogram
        axes[i, 0].hist(samples, bins=bins, density=True, alpha=0.6)
        axes[i, 0].axvline(mean, color="r", linestyle="-")
        axes[i, 0].axvline(mean + sd, color="r", linestyle="--")
        axes[i, 0].axvline(mean - sd, color="r", linestyle="--")
        axes[i, 0].set_xlabel(var_name)
        if i == 0:
            axes[i, 0].text(
                mean + 2 * sd,
                plt.ylim()[1] * 0.5,
                rf"$\alpha$ = {mean:.2f} $\pm$ {sd:.2f}",
                horizontalalignment="center",
                color="r",
            )
        elif i == 1:
            axes[i, 0].text(
                mean + 2 * sd,
                plt.ylim()[1] * 0.5,
                rf"$\beta$ = {mean:.2f} $\pm$ {sd:.2f}",
                horizontalalignment="center",
                color="r",
            )
        elif i == 2:
            axes[i, 0].text(
                mean + 2 * sd,
                plt.ylim()[1] * 0.5,
                rf"$I_{0}$ = {mean:.2f} $\pm$ {sd:.2f}",
                horizontalalignment="center",
                color="r",
            )

    plt.tight_layout()
    plt.show()


def plot_geweke(trace, intervals=15):
    """
    Plot the Geweke diagnostic for each variable in the MCMC trace.

    The Geweke diagnostic is a convergence diagnostic that compares
    the mean and variance of segments from the beginning and end of a single chain.
    This function plots the Geweke z-scores for each variable
    across specified intervals, helping to assess the chain's convergence.

    Parameters:
    - trace : arviz.InferenceData
        The MCMC trace data, encapsulated in an ArviZ InferenceData object.
    - intervals : int, optional
        The number of intervals to divide the trace into for the Geweke diagnostic.
        Default is 15.

    Notes:
    - The function plots z-scores and marks the ±2 standard deviation range with
    horizontal lines.
    - Z-scores within ±2 suggest that the segment means are within 2 standard deviations
    of each other,
      indicating convergence.
    - The function uses Matplotlib for plotting and displays the plot directly.
    """
    var_names = list(trace.posterior.data_vars)

    # Determine the number of subplots needed
    n_vars = len(var_names)

    # Create subplots
    fig, axes = plt.subplots(
        n_vars, 1, figsize=(10, 3 * n_vars), sharex=True, squeeze=False
    )

    for i, var_name in enumerate(var_names):
        # Calculate Geweke diagnostic for each variable
        var_samples = trace.posterior[
            var_name
        ].values.flatten()  # Flatten in case of multidimensional variables
        geweke_results = az.geweke(var_samples, intervals=intervals)

        # Extract iterations and z-scores
        iterations, z_scores = geweke_results[:, 0], geweke_results[:, 1]

        # Plot Geweke diagnostic
        axes[i, 0].scatter(iterations, z_scores, alpha=0.6)
        axes[i, 0].axhline(y=2, color="r", linestyle="--", label=r"2 $\sigma$")
        axes[i, 0].axhline(y=-2, color="r", linestyle="--")
        axes[i, 0].set_ylabel("Z-score")
        axes[i, 0].legend(loc="upper right")
        if i == 0:
            axes[i, 0].set_title(r"alpha, $\alpha$")
        elif i == 1:
            axes[i, 0].set_title(r"beta, $\beta$")
        elif i == 2:
            axes[i, 0].set_title(rf"$I_{0}$")

    # Set common xlabel
    axes[-1, 0].set_xlabel("Iteration")

    plt.tight_layout()
    plt.show()


def plotting_x(trace):
    """
    Plot a comprehensive analysis of the MCMC trace for 'x' variables,
    including joint distributions, marginal posteriors, and convergence diagnostics.

    This high-level function orchestrates the plotting of several key diagnostic
    and exploratory plots for assessing the MCMC sampling results.
    It includes the joint distribution of the 'alpha' and 'beta' parameters,
    marginal posterior distributions for all variables
    and Geweke convergence diagnostics.

    Parameters:
    - trace : arviz.InferenceData
        The MCMC trace data, encapsulated in an ArviZ InferenceData object.

    Notes:
    - This function calls `joint_posterior_x`, `marginal_posterior`, and `plot_geweke`
    to generate the plots.
    - The function uses Matplotlib for plotting and displays the plots directly.
    """
    joint_posterior_x(trace)
    marginal_posterior(trace)
    plot_geweke(trace)


def plotting_xi(trace):
    """
    Plot a comprehensive analysis of the MCMC trace for 'x' variables
    and intensities, including joint distributions, marginal posteriors
    and convergence diagnostics.

    This function orchestrates the plotting of several key diagnostic
    and exploratory plots for assessing the MCMC sampling results related
    to flash location and intensity variables. It visualises the
    joint distributions for pairs of parameters, marginal posterior
    distributions for all variables and Geweke convergence diagnostics.

    Parameters:
    - trace : arviz.InferenceData
        The MCMC trace data, encapsulated in an ArviZ InferenceData object.

    Notes:
    - The function calls `joint_posterior_xi`, `marginal_posterior`, and `plot_geweke
    sequentially to generate the plots.
    - Plots are displayed directly using Matplotlib.
    """
    joint_posterior_xi(trace)
    marginal_posterior(trace)
    plot_geweke(trace)


def appendix_plots(trace):
    """
    Generate and display plots for the appendix, including trace plots and corner plots.

    This function first displays trace plots for each variable in the trace,
    allowing for the visualisation of the sampling paths and their distributions.
    Following this, it creates corner plots (also known as pair plots) that show the
    multidimensional relationships between variables, including scatter plots for
    joint distributions and histograms for marginal distributions.

    Parameters:
    - trace : arviz.InferenceData
        The MCMC trace data, encapsulated in an ArviZ InferenceData object.

    Notes:
    - Summary statistics are calculated using ArviZ's summary function, and these
    are optionallyy used as truths in the corner plot.
    - The function uses ArviZ for trace plots and the `corner` module for corner plots,
    displaying them directly.
    """
    summary_stats = az.summary(trace, round_to=2)

    # Trace
    az.plot_trace(trace)
    plt.show()

    # Corner plot
    corner.corner(trace, truths=summary_stats["mean"])
    plt.show()
