import numpy as np
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter


tdf = tfp.distributions


def plotv1(samples, xrange, yrange):
    '''
    Plot part v of the coursework.
    '''
    x, y = samples
    a, b = xrange
    c, d = yrange

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
    hb = axScatter.hexbin(x, y, gridsize=50, cmap='inferno', bins='log')

    # Set axis labels
    axScatter.set_xlabel(r'alpha, $\alpha$')
    axScatter.set_ylabel(r'beta, $\beta$') 

    # Determine nice limits by hand
    binwidth = 0.25
    xymax = np.max([np.max(np.abs(x)), np.max(np.abs(y))])
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

def plotv2(samples, xrange, yrange):

    a, b = xrange
    c, d = yrange

    # Assuming alpha_samples and beta_samples are defined and contain your MCMC samples
    alpha_samples = samples[0]
    beta_samples = samples[1]

    # Calculate mean, median, and standard deviation
    mean_alpha = np.mean(alpha_samples)
    median_alpha = np.median(alpha_samples)
    std_alpha = np.std(alpha_samples)

    mean_beta = np.mean(beta_samples)
    median_beta = np.median(beta_samples)
    std_beta = np.std(beta_samples)

    # Create histograms for alpha and beta
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot for alpha
    axes[0].hist(alpha_samples, bins=50, density=True)
    axes[0].axvline(mean_alpha, color='r', linestyle='-')
    axes[0].axvline(mean_alpha + std_alpha, color='r', linestyle='--')
    axes[0].axvline(mean_alpha - std_alpha, color='r', linestyle='--')
    axes[0].set_xlim([a, b])
    axes[0].set_xlabel(r'$\alpha$')
    axes[0].text(mean_alpha+2*std_alpha+0.1, plt.ylim()[1] * 0.5, fr'$\mu_\alpha$ = {mean_alpha:.2f} $\pm$ {std_alpha:.2f}', horizontalalignment='center', color='r')


    # Plot prior for alpha
    x_alpha = np.linspace(-10, 10, 100)
    y_alpha = tdf.Uniform(-10, 10).prob(x_alpha).numpy()  # Convert to NumPy array for Matplotlib
    axes[0].plot(x_alpha, y_alpha, label='Unifrom Prior')
    axes[0].legend()

    # Plot for beta
    axes[1].hist(beta_samples, bins=50, density=True)
    axes[1].axvline(mean_beta, color='r', linestyle='-')
    axes[1].axvline(mean_beta + std_beta, color='r', linestyle='--')
    axes[1].axvline(mean_beta - std_beta, color='r', linestyle='--')
    axes[1].set_xlim([c, d])
    axes[1].set_xlabel(r'$\beta$')
    axes[1].text(mean_beta+2*std_beta, plt.ylim()[1] * 0.6, fr'$\mu_\beta$ = {mean_beta:.2f} $\pm$ {std_beta:.2f}', horizontalalignment='center', color='r')


    # Plot prior for beta
    x_beta = np.linspace(0, 10, 100)
    y_beta = tdf.Uniform(0, 10).prob(x_beta).numpy()  # Convert to NumPy array for Matplotlib
    axes[1].plot(x_beta, y_beta, label='Uniform Prior')


    plt.show()
