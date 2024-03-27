import corner
import numpy as np
import arviz as az
import tensorflow_probability as tfp
from reading_utils import read_config
from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter

# plotting functions
def plot_cauchy(cauchy):
    # plot cauchy distributions
    x = np.linspace(-10, 10, 1000)
    y1 = cauchy(x, alpha=0, beta=0.5)
    y2 = cauchy(x, alpha=0, beta=1)
    y3 = cauchy(x, alpha=-2, beta=2)

    plt.plot(x, y1, label=r'$\alpha=0, \beta=0.5$')
    plt.plot(x, y2, label=r'$\alpha=0, \beta=1$')
    plt.plot(x, y3, label=r'$\alpha=-2, \beta=2$')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('P(x)')
    plt.xlim(-5, 5)
    plt.show()

def plot_cauchy_analysis(x, x_true, y_true, mean, mode, bins_number
):
    # Create the histogram with the new number of bins
    n, bins, patches = plt.hist(x, bins=bins_number, density=True, color='blue', alpha=0.7, range=(-20, 20))

    # Add the Cauchy distribution PDF
    plt.plot(x_true, y_true, 'r', label='True PDF')

    # Indicate the mean and median (mode)
    plt.axvline(mean, color='magenta', linestyle='dashed', linewidth=1.5, label='Sample Mean')
    plt.axvline(mode, color='orange', linestyle='dashed', linewidth=1.5, label='Sample Median')

    # Limit the x-axis to better visualize the peak
    plt.xlim(-10, 10)

    # Add a legend to the plot
    plt.legend()

    plt.xlabel('x')
    plt.ylabel('P(x)')

    # Show the plot
    plt.show()

# Read the configuration file
model_params, sampling_params = read_config('parameters.ini')
a, b, c, d = model_params['a'], model_params['b'], model_params['c'], model_params['d']

def trace_plot(trace, figsize=(12, 8)):

    # Extract variable names directly from the InferenceData object
    var_names = list(trace.posterior.data_vars)

    n_vars = len(var_names)
    fig, axes = plt.subplots(n_vars, 1, sharex=True, figsize=figsize, squeeze=False)

    for i, var_name in enumerate(var_names):
        # Extract samples, combining chains if there are multiple
        samples = trace.posterior[var_name].values

        # Plot trace for each chain
        for chain_samples in samples:
            axes[i, 0].plot(chain_samples, lw=0.15, alpha=0.7)

        if i == 0:
            axes[i, 0].set_ylabel(r'alpha, $\alpha$')
        elif i == 1:
            axes[i, 0].set_ylabel(r'beta, $\beta$')
        elif i == 2:
            axes[i, 0].set_ylabel(fr'$I_{0}$')

    # Set common labels
    axes[-1, 0].set_xlabel('Iteration')

    plt.tight_layout()
    plt.show()

def joint_posterior_x(trace):
    """
    Plots the joint distribution of alpha and beta samples from a trace,
    with marginal histograms.
    """
    # Unpack trace
    alpha_samples = trace.posterior['alpha'].values.flatten()
    beta_samples = trace.posterior['beta'].values.flatten()

    nullfmt = NullFormatter()

    # Define axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # Start figure
    fig = plt.figure(figsize=(9.5, 9))

    # Scatter plot
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # No labels for histograms
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # Scatter plot with hexbin
    hb = axScatter.hexbin(alpha_samples, beta_samples, gridsize=100, cmap='inferno', bins='log')

    # Set axis labels
    axScatter.set_xlabel(r'alpha, $\alpha$')
    axScatter.set_ylabel(r'beta, $\beta$')

    # Automatically determine nice limits
    axScatter.set_xlim((alpha_samples.min(), alpha_samples.max()))
    axScatter.set_ylim((beta_samples.min(), beta_samples.max()))

    # Marginal histograms
    axHistx.hist(alpha_samples, bins=50, density=True, alpha=0.6)
    axHisty.hist(beta_samples, bins=50, orientation='horizontal', density=True, alpha=0.6)

    # Set histogram limits to match the scatter plot
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    plt.show()

def joint_posterior_xi(trace):
    """
    Creates a series of 2D plots for each pair of variables.
    """
    # unpack trace
    alpha_samples = trace.posterior['alpha'].values.flatten()
    beta_samples = trace.posterior['beta'].values.flatten()
    I0_samples = trace.posterior['I0'].values.flatten()

    # Start figure
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))  # Adjust size as needed

    # Plot alpha vs beta
    axes[1, 0].hexbin(alpha_samples, beta_samples, gridsize=50, cmap='inferno', bins='log')
    axes[1, 0].set_xlabel(r'alpha, $\alpha$')
    axes[1, 0].set_ylabel(r'beta, $\beta$')

    # Plot alpha vs I0
    axes[2, 0].hexbin(alpha_samples, I0_samples, gridsize=50, cmap='inferno', bins='log')
    axes[2, 0].set_xlabel(r'alpha, $\alpha$')
    axes[2, 0].set_ylabel('I0')

    # Plot beta vs I0
    axes[2, 1].hexbin(beta_samples, I0_samples, gridsize=50, cmap='inferno', bins='log')
    axes[2, 1].set_xlabel(r'beta, $\beta$')
    axes[2, 1].set_ylabel(fr'$I_{0}$')

    # Histograms for alpha, beta, and I0
    axes[0, 0].hist(alpha_samples, bins=50, orientation='vertical', alpha=0.6)
 
    axes[1, 1].hist(beta_samples, bins=50, orientation='vertical', alpha=0.6)
  
    axes[2, 2].hist(I0_samples, bins=50, orientation='vertical', alpha=0.6)
 
    # Hide the empty subplots
    axes[0, 1].axis('off')
    axes[0, 2].axis('off')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

def marginal_posterior(trace, bins=50, figsize=(12, 8)):
    
    var_names = list(trace.posterior.data_vars)

    # Obtain summary statistics using ArviZ
    summary = az.summary(trace, var_names=var_names, round_to=2)
    
    # Create subplots
    fig, axes = plt.subplots(len(var_names), 1, figsize=figsize, squeeze=False)
    
    for i, var_name in enumerate(var_names):
        # Extract samples
        samples = trace.posterior[var_name].values.flatten()
        
        # Extract summary statistics
        mean = summary.loc[var_name, 'mean']
        sd = summary.loc[var_name, 'sd']
        
        # Plot histogram
        axes[i, 0].hist(samples, bins=bins, density=True, alpha=0.6)
        axes[i, 0].axvline(mean, color='r', linestyle='-')
        axes[i, 0].axvline(mean + sd, color='r', linestyle='--')
        axes[i, 0].axvline(mean - sd, color='r', linestyle='--')
        axes[i, 0].set_xlabel(var_name)
        if i == 0:
            axes[i, 0].text(mean + 2*sd, plt.ylim()[1] * 0.5, fr'$\alpha$ = {mean:.2f} $\pm$ {sd:.2f}', horizontalalignment='center', color='r')
        elif i == 1:
            axes[i, 0].text(mean + 2*sd, plt.ylim()[1] * 0.5, fr'$\beta$ = {mean:.2f} $\pm$ {sd:.2f}', horizontalalignment='center', color='r')
        elif i == 2:
            axes[i, 0].text(mean + 2*sd, plt.ylim()[1] * 0.5, fr'$I_{0}$ = {mean:.2f} $\pm$ {sd:.2f}', horizontalalignment='center', color='r')

    plt.tight_layout()
    plt.show()

def plot_geweke(trace, intervals=15):

    var_names = list(trace.posterior.data_vars)
    
    # Determine the number of subplots needed
    n_vars = len(var_names)
    
    # Create subplots
    fig, axes = plt.subplots(n_vars, 1, figsize=(10, 3 * n_vars), sharex=True, squeeze=False)
    
    for i, var_name in enumerate(var_names):
        # Calculate Geweke diagnostic for each variable
        var_samples = trace.posterior[var_name].values.flatten()  # Flatten in case of multidimensional variables
        geweke_results = az.geweke(var_samples, intervals=intervals)
        
        # Extract iterations and z-scores
        iterations, z_scores = geweke_results[:, 0], geweke_results[:, 1]
        
        # Plot Geweke diagnostic
        axes[i, 0].scatter(iterations, z_scores, alpha=0.6)
        axes[i, 0].axhline(y=2, color='r', linestyle='--', label=r'2 $\sigma$')
        axes[i, 0].axhline(y=-2, color='r', linestyle='--')
        axes[i, 0].set_ylabel('Z-score')
        axes[i, 0].legend(loc='upper right')
        if i == 0:
            axes[i, 0].set_title(r'alpha, $\alpha$')
        elif i == 1:
            axes[i, 0].set_title(r'beta, $\beta$')
        elif i == 2:
            axes[i, 0].set_title(fr'$I_{0}$')
        
    # Set common xlabel
    axes[-1, 0].set_xlabel('Iteration')

    plt.tight_layout()
    plt.show()

def plotting_x(trace):
    """
    Function to plot all the Flash location plots
    """
    joint_posterior_x(trace)
    marginal_posterior(trace)
    plot_geweke(trace)

def plotting_xi(trace):
    """
    Function to plot all the Flash location and Intensity plots
    """
    joint_posterior_xi(trace)
    marginal_posterior(trace)
    plot_geweke(trace)
    
def appendix_plots(trace):
    """
    Function to plot all the appendix plots
    """

    summary_stats = az.summary(trace, round_to=2)

    # Trace
    az.plot_trace(trace)
    plt.show()
    corner.corner(trace, truths=summary_stats['mean'])
    plt.show()

