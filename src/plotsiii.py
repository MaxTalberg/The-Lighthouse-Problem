import numpy as np
from matplotlib import pyplot as plt

# set seed
np.random.seed(120420)

# cauchy distribution
def cauchy(x, alpha, beta):
    return beta / np.pi * 1 / (beta**2 + (x - alpha)**2)

# trigonometric function
def trigonometric(theta, alpha, beta):
    return beta * np.tan(theta) + alpha

# Set the parameters
alpha = 0
beta = 1

# Generate data for histogram
theta = np.random.uniform(-np.pi/2, np.pi/2, 100000)
x = trigonometric(theta, alpha, beta)

# Generate data for the true distribution
x_true = np.linspace(-20, 20, 1000)
y_true = cauchy(x_true, alpha, beta)

# Calculate mean and mode
mean = np.mean(x)
mode = np.median(x)

bins_number = 200

# plotting functions
def plotiii1():
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

def plotiii2():
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