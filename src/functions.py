from corner import corner
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from src.plotsiii import plotiii1, plotiii2
from src.plotsv import plotv1, plotv2

tdf = tfp.distributions

# joint log probability function for flash locations
def joint_log_prob_x(x, alpha, beta):
    # likelihood
    likelihood = tdf.Cauchy(loc=alpha, scale=beta).log_prob(x)
    log_likelihood = tf.reduce_sum(likelihood)

    # prior
    log_prior_alpha = tf.where(
        tf.logical_and(alpha > a, alpha < b),
        tf.math.log(1/(b-a)),
        -np.inf
    
    )
    log_prior_beta = tf.where(
        tf.logical_and(beta > c, beta < d),
        tf.math.log(1/(d-c)),
        -np.inf
    )

    return log_prior_alpha + log_prior_beta + log_likelihood

# joint log probability function for flash locations
def joint_log_prob_I(x, I, alpha, beta, I0):

    # likelihood
    d = tf.sqrt(beta**2 + (x - alpha)**2)
    mu = tf.math.log(I0) - 2*tf.math.log(d)
    likelihood = tdf.LogNormal(loc = mu, scale = 1).log_prob(I)
    log_likelihood = tf.reduce_sum(likelihood)

    # prior
    log_uniform_prior = tf.where(
        tf.logical_and(I0 >e, I0 < f),
        -tf.math.log(I0) - tf.math.log(tf.math.log(f) - tf.math.log(e)),
        -np.inf
    
    )

    return log_uniform_prior + log_likelihood
