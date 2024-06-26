import pymc3 as pm
import theano.tensor as tt
import numpy as np


def define_model_x(x_observed, a, b, c, d):
    """
    Defines a Bayesian model for x_observed data with uniform priors for alpha and beta.

    Parameters:
    - x_observed: Observed data for x (flash locations).
    - a, b: Lower and upper bounds for the uniform prior of alpha.
    - c, d: Lower and upper bounds for the uniform prior of beta.

    Returns:
    - PyMC3 model object.
    """
    with pm.Model() as model:
        # Priors
        alpha = pm.Uniform("alpha", lower=a, upper=b)
        beta = pm.Uniform("beta", lower=c, upper=d)

        # Likelihood
        pm.Cauchy("x_likelihood", alpha=alpha, beta=beta, observed=x_observed)
    return model


def define_model_xi(x_observed, I_observed, a, b, c, d):
    """
    Defines a Bayesian model for x_observed and I_observed data with uniform priors for
    alpha and beta and a LogNormal prior for I0.

    Parameters:
    - x_observed: Observed data for x (flash locations).
    - I_observed: Observed data for I (intensities).
    - a, b: Lower and upper bounds for the uniform prior of alpha.
    - c: Upper bound for the uniform prior of beta (assuming lower bound is 0).

    Returns:
    - PyMC3 model object.
    """
    with pm.Model() as model:
        # Priors
        alpha = pm.Uniform("alpha", lower=a, upper=b)
        beta = pm.Uniform("beta", lower=c, upper=d)
        I0 = pm.Pareto("I0", alpha=2, m=0.01)

        # Likelihoods
        pm.Cauchy("x_likelihood", alpha=alpha, beta=beta, observed=x_observed)
        d = tt.sqrt(beta**2 + (x_observed - alpha) ** 2)
        mu = tt.log(I0) - 2 * tt.log(d)
        pm.Lognormal("I_likelihood", mu=mu, sigma=1, observed=I_observed)
    return model


def sample_model(model, seed, draws, tune, chains, target_accept):
    """
    Samples from a given PyMC3 model using the No-U-Turn Sampler (NUTS).

    Parameters:
    - model: A PyMC3 model object to be sampled from.
    - seed: The random seed to use for reproducibility.
    - draws: The number of samples to draw from the posterior distribution.
    - tune: The number of iterations to tune the sampler.
    - chains: The number of independent chains to run.
    - target_accept: The target acceptance probability for the NUTS sampler.

    Returns:
    - A PyMC3 Trace object containing the samples.
    """
    np.random.seed(seed)

    with model:
        step = pm.NUTS(target_accept=target_accept)
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            step=step,
            return_inferencedata=True,
        )
    return trace
