import warnings
import numpy as np
from anlaysing_utils import thinning, convergence_diagnostic, mean_mle_analysis ,appendix_data, cauchy
from reading_utils import read_and_prepare_data, read_config
from plotting_utils import plot_cauchy, plot_cauchy_analysis, trace_plot, plotting_x, plotting_xi, appendix_plots
from sampling_utils import define_model_x, define_model_xi, sample_model


warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered in _beta_ppf')


def main(appendix = False):
    # Read the configuration file
    model_params, sampling_params, seed = read_config('parameters.ini')

    # Initialise random seed
    np.random.seed(seed)

    ## iii)
    # Cauchy MLE and mean flash location analysis
    analysis_results = mean_mle_analysis()
    plot_cauchy(cauchy)
    plot_cauchy_analysis(*analysis_results)

    # Read and prepare the data
    x_observed, I_observed = read_and_prepare_data('lighthouse_flash_data.txt')

    # Define models
    model_x = define_model_x(x_observed, **model_params)
    model_xi = define_model_xi(x_observed, I_observed, **model_params)

    ## v)  Flash Locations
    trace_x = sample_model(model_x, **sampling_params)  # Sampling
    trace_plot(trace_x)
    thinned_trace_x = thinning(trace_x)
    convergence_diagnostic(thinned_trace_x)
    plotting_x(thinned_trace_x)

    ## vii) Flash Locations and Intensities
    trace_xi = sample_model(model_xi, **sampling_params) # Sampling
    trace_plot(trace_xi)
    thinned_trace_xi = thinning(trace_xi)
    convergence_diagnostic(thinned_trace_xi)
    plotting_xi(thinned_trace_xi)

    if appendix:
        appendix_data(trace_x)
        appendix_data(thinned_trace_x)
        appendix_data(trace_xi)
        appendix_data(thinned_trace_xi)

        appendix_plots(trace_x)
        appendix_plots(thinned_trace_x)
        appendix_plots(trace_xi)
        appendix_plots(thinned_trace_xi)

if __name__ == '__main__':
    main(appendix=True)
