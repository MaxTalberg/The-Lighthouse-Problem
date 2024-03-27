import warnings

from reading_utils import read_and_prepare_data, read_config
from sampling_utils import define_model_x, define_model_xi, sample_model
from anlaysing_utils import (
    thinning,
    convergence_diagnostic,
    mean_mle_analysis,
    appendix_data,
    cauchy,
)
from plotting_utils import (
    plot_cauchy,
    plot_cauchy_analysis,
    trace_plot,
    plotting_x,
    plotting_xi,
    appendix_plots,
)


warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="overflow encountered in _beta_ppf"
)


def main(appendix=False):
    # Read the configuration file
    model_params, sampling_params, seed = read_config("parameters.ini")

    ## iii)
    # Cauchy MLE and mean flash location analysis
    analysis_results = mean_mle_analysis(seed)
    plot_cauchy(cauchy)
    plot_cauchy_analysis(*analysis_results)

    # Read and prepare the data
    x_observed, I_observed = read_and_prepare_data("lighthouse_flash_data.txt")

    # Define models
    model_x = define_model_x(x_observed, **model_params)
    model_xi = define_model_xi(x_observed, I_observed, **model_params)

    ## v)  Flash Locations
    trace_x = sample_model(model_x, seed, **sampling_params)
    trace_plot(trace_x)
    thinned_trace_x = thinning(trace_x)
    convergence_diagnostic(thinned_trace_x)
    plotting_x(thinned_trace_x)

    ## vii) Flash Locations and Intensities
    trace_xi = sample_model(model_xi, seed, **sampling_params)
    trace_plot(trace_xi)
    thinned_trace_xi = thinning(trace_xi)
    convergence_diagnostic(thinned_trace_xi)
    plotting_xi(thinned_trace_xi)

    if appendix:
        print("Appendix data")
        appendix_data(trace_x)
        appendix_data(thinned_trace_x)
        appendix_data(trace_xi)
        appendix_data(thinned_trace_xi)

        appendix_plots(trace_x)
        appendix_plots(thinned_trace_x)
        appendix_plots(trace_xi)
        appendix_plots(thinned_trace_xi)


if __name__ == "__main__":
    main(appendix=True)
