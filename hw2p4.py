
from math import exp, pi, sqrt, isclose
from scipy.stats import chi2
from scipy.special import gamma
import numpy as np


SIGMA = 4
ALPHA = 2
BETA = 0.5

def calculate_rayleigh(x) -> float:
    return x/(pow(SIGMA, 2)) * exp(-1 * pow(x, 2)/(2*pow(SIGMA, 2)))

def gamma_dist(x) -> float:
    return pow(BETA, ALPHA)/gamma(ALPHA) * pow(x, ALPHA-1) * exp(-1 * BETA * x)

def run_MCMC_sampler(target_func, num_samples):
    markov_chain = []
    num_rejections = 0
    # initial value is sampled from chi-squared(Expectation(X)), where E(X) = sigma * sqrt(pi/2)
    rayleigh_expectation = SIGMA * sqrt(pi/2)
    markov_chain.append(np.random.chisquare(rayleigh_expectation))
    for i in range(1, num_samples):
        X = markov_chain[i-1]
        X_prime = np.random.chisquare(X)
        # q(X_prime | X) = chi_squared(X_prime, x), where x = degrees of freedom
        P_x = target_func(X)
        P_x_prime = target_func(X_prime)
        Q_x = chi2.pdf(X, X_prime)
        Q_x_prime = chi2.pdf(X_prime, X)
        A_x = min(1, P_x_prime*Q_x/(P_x * Q_x_prime))
        if np.random.uniform() <= A_x:
            markov_chain.append(X_prime)
        else:
            markov_chain.append(X)
            num_rejections += 1

    return markov_chain, num_rejections

import matplotlib.pyplot as plt
   
def plot_sampler_with_stats(target_func, num_samples):
    # run sampler
    markov_chain, num_rejections = run_MCMC_sampler(target_func, num_samples)

    # print mean, variances, and reject rate
    mean = sum(markov_chain)/len(markov_chain)
    variance = sum([pow(x_i - mean, 2) for x_i in markov_chain])/len(markov_chain)
    reject_rate = num_rejections/num_samples
    print('Mean : ' + str(mean))
    print("Variance : " + str(variance))
    print("Rejection Rate : " + str(reject_rate))

    # plot histogram
    plt.hist(markov_chain, bins=50, density=True, edgecolor="black")
    x = np.linspace(0, 20, num_samples)
    y = [target_func(_) for _ in x]
    plt.plot(x,y,'r')
    plt.show()

plot_sampler_with_stats(calculate_rayleigh, num_samples=10000)
plot_sampler_with_stats(gamma_dist, num_samples=10000)