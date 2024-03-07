import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tdf = tfp.distributions

# Read the data
def read_data(file_path):
    '''
    Function to read data from a text file and return it as two lists.
    '''
    # initialise lists to store the data
    column1 = []  
    column2 = []  
    
    with open(file_path, 'r') as file:  # Open the file for reading
        for line in file:  # Iterate over each line in the file
            parts = line.split()  # Split the line by whitespace
            
            # Append the parts to their respective column lists
            column1.append(float(parts[0]))
            column2.append(float(parts[1]))

    return column1, column2

# set seed
np.random.seed(120420)
tf.random.set_seed(120420)

# Set the parameters
file_path = 'src/lighthouse_flash_data.txt'

# observed flashes
x_observed = np.array(read_data(file_path)[0])

# constants for the uniform prior
a = -10
b = 10
c = 10

# joint log probability function
def joint_log_prob(x, alpha, beta):
    '''
    Generate the joint log probability function for the likelihood and priors.
    '''
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
        tf.logical_and(beta > 0, beta < c),
        tf.math.log(1/c),
        -np.inf
    )

    return log_prior_alpha + log_prior_beta + log_likelihood

# Define the unnormalized posterior function
def unnormalized_posterior(x_observed, alpha, beta):
    '''
    Define the unnormalized posterior function.
    '''
    return joint_log_prob(x_observed, alpha, beta)

# hmc kernel
hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
  target_log_prob_fn=unnormalized_posterior,
  step_size=np.float64(.01),
  num_leapfrog_steps=200) 

# run hmc
@tf.function
def run_chain(num_results=30000, num_burnin_steps=3000):
  samples, kernel_results = tfp.mcmc.sample_chain(
      num_results=num_results,
      num_burnin_steps=num_burnin_steps,
      current_state=[np.mean(x_observed), 1.0],
      kernel=hmc_kernel,
      trace_fn=lambda _, pkr: pkr.is_accepted
      )
  return samples, kernel_results