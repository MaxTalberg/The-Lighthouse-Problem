import numpy as np
from matplotlib import pyplot as plt


# plotting functions
def plotiii1(cauchy):
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

def plotiii2(x, x_true, y_true, mean, mode, bins_number
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