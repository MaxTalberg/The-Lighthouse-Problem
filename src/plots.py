import numpy as np
import arviz as az
import tensorflow_probability as tfp
from Coursework.src.processing_utils import read_config
from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter

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

        axes[i, 0].set_title(var_name)

    # Set common labels
    axes[-1, 0].set_xlabel('Iteration')

    plt.tight_layout()
    plt.show()

def joint_posterior_x(trace):
    """
    """
    # unpack trace
    alpha_samples = trace.get_values('alpha', 'beta')
    beta_samples = trace.get_values('beta')

    nullfmt = NullFormatter()

    # Definitions for the axes
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

    # No labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # Create a hexbin plot with a color bar
    hb = axScatter.hexbin(alpha_samples, beta_samples, gridsize=50, cmap='inferno', bins='log')

    # Set axis labels
    axScatter.set_xlabel(r'alpha, $\alpha$')
    axScatter.set_ylabel(r'beta, $\beta$') 

    # Determine nice limits by hand
    binwidth = 0.25
    xymax = np.max([np.max(np.abs(alpha_samples)), np.max(np.abs(beta_samples))])
    lim = (int(xymax/binwidth) + 1) * binwidth

    axScatter.set_xlim((a, b))
    axScatter.set_ylim((c, d))

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, bins=50)
    axHisty.hist(y, bins=50, orientation='horizontal')

    # Set limits for the histograms
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

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
        axes[i, 0].text(mean + 2*sd, plt.ylim()[1] * 0.5, fr'$\mu_{var_name}$ = {mean:.2f} $\pm$ {sd:.2f}', horizontalalignment='center', color='r')
        
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
        axes[i, 0].scatter(iterations, z_scores, color='blue', edgecolor='k')
        axes[i, 0].axhline(y=2, color='r', linestyle='--', label=r'2 $\sigma$')
        axes[i, 0].axhline(y=-2, color='r', linestyle='--')
        axes[i, 0].set_ylabel('Z-score')
        axes[i, 0].set_title(f'Geweke Plot for {var_name}')
        axes[i, 0].grid(True)
        axes[i, 0].legend(loc='upper right')
        
    # Set common xlabel
    axes[-1, 0].set_xlabel('Iteration')

    plt.tight_layout()
    plt.show()