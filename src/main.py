import warnings
import numpy as np
from anlaysing_utils import thinning, convergence_diagnostic, mean_mle_analysis ,appendix_data, cauchy
from reading_utils import read_and_prepare_data, read_config
from plotting_utils import plot_cauchy, plot_cauchy_analysis, trace_plot, plotting_x, plotting_xi, appendix_plots
from sampling_utils import define_model_x, define_model_xi, sample_model


warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered in _beta_ppf')


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(12042000)

    # question iii
    analysis_results = mean_mle_analysis()
    plot_cauchy(cauchy)
    plot_cauchy_analysis(*analysis_results)

    # Read and prepare the data
    x_observed, I_observed = read_and_prepare_data('lighthouse_flash_data.txt')

    # Read the configuration file
    model_params, sampling_params = read_config('parameters.ini')

    # Define the models
    model_x = define_model_x(x_observed, **model_params)
    model_xi = define_model_xi(x_observed, I_observed, **model_params)

    ##  Flash Locations
    # Run sampling
    trace_x = sample_model(model_x, **sampling_params)

    # Trace plot
    trace_plot(trace_x)

    # Thinning and convergence diagnostics
    thinned_trace_x = thinning(trace_x)
    convergence_diagnostic(thinned_trace_x)

    # Plots
    plotting_x(thinned_trace_x)

    ##  Flash Locations and Intensities
    # Run sampling
    trace_xi = sample_model(model_xi, **sampling_params)

    # Trace plot
    trace_plot(trace_xi)

    # Thinning and convergence diagnostics
    thinned_trace_xi = thinning(trace_xi)
    convergence_diagnostic(thinned_trace_xi)

    # Plots
    plotting_xi(thinned_trace_xi)
    

    '''
    # Appendix data
    appendix_data(trace_x)
    appendix_data(thinned_trace_x)
    appendix_data(trace_xi)
    appendix_data(thinned_trace_xi)

    # Appendix plots
    appendix_plots(trace_x)
    appendix_plots(thinned_trace_x)
    appendix_plots(trace_xi)
    appendix_plots(thinned_trace_xi)
    '''