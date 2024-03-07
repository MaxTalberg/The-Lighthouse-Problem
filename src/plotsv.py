import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter

def plotv(samples):
    '''
    Plot part v of the coursework.
    '''
    x, y = samples

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

    axScatter.set_xlim((-6, 6))
    axScatter.set_ylim((0, 8))

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, bins=50)
    axHisty.hist(y, bins=50, orientation='horizontal')

    # Set limits for the histograms
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    plt.show()
