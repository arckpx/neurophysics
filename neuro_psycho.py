import numpy as np
from scipy.stats import truncnorm
from scipy.optimize import curve_fit
from scipy.integrate import simps
import matplotlib.pyplot as plt


def get_truncnorm_mean(lower, upper, mu, sigma):
    a = (lower - mu) / sigma
    b = (upper - mu) / sigma
    return truncnorm.stats(a, b, loc=mu, scale=sigma, moments='m')


def get_truncnorm_samples(lower, upper, mu, sigma, size):
    a = (lower - mu) / sigma
    b = (upper - mu) / sigma
    return truncnorm(a, b, loc=mu, scale=sigma).rvs(size)


def normalize(vector):
    return (vector - min(vector)) / (max(vector) - min(vector))


def generate_stimuli(n_corr, n_test, n_time, expbase=100, sigmin=0.001, sigmax=100):
    # nonlinear sigmas
    nonlin = expbase ** normalize(np.linspace(sigmin, sigmax, n_corr))
    sigmas = np.flip(normalize(nonlin) * (sigmax - sigmin) + sigmin)

    corr = np.array([get_truncnorm_mean(-1, 1, 1, sigma) for sigma in sigmas]) * 100  # (n_corr)
    answers = 2 * np.random.randint(2, size=(n_corr, n_test)) - 1  # (n_corr, n_test)
    input_template = np.array([np.reshape(get_truncnorm_samples(-1, 1, 1, sigma, n_test * n_time), (n_test, n_time)) for sigma in sigmas])  # input if answer = +1
    inputs = input_template * np.expand_dims(answers, -1)  # (n_corr, n_test, n_time)

    return corr, inputs, answers


def generate_neurons(size, thres_sigma=1):
    # thres_sigma: neural activation threshold variance (decrease to improve neurons)
    return get_truncnorm_samples(-1, 1, 0, thres_sigma, size)


def get_responses(inputs, neurons):
    # responses shape: (n_corr, n_neuron, n_test, n_time)
    return np.expand_dims(inputs, 1) > np.expand_dims(np.expand_dims(np.expand_dims(neurons, 0), -1), - 1)


def calc_neuropsycho(responses, answers):
    neuro_responses = np.sign(np.mean(responses, axis=-1) - 0.5)  # mean over time
    neuro_responses[neuro_responses == 0] = 2 * np.random.randint(2, size=neuro_responses[neuro_responses == 0].shape) - 1  # handle zeroes
    neuro_correctness = neuro_responses.astype(int) == np.expand_dims(answers, 1)
    neurometric = np.mean(neuro_correctness, axis=-1)  # mean over test

    psycho_responses = np.sign(np.mean(neuro_responses, axis=1))  # mean over neurons
    psycho_responses[psycho_responses == 0] = 2 * np.random.randint(2, size=psycho_responses[psycho_responses == 0].shape) - 1  # handle zeroes
    psycho_correctness = psycho_responses.astype(int) == answers
    psychometric = np.mean(psycho_correctness, axis=-1)  # mean over test
    psychometric_errors = np.std(psycho_correctness, axis=-1) / np.sqrt(n_test)  # std over test

    # neurometric shape: (n_corr, n_neuron), psychometric shape: (n_corr)
    return neurometric, psychometric, psychometric_errors


def logistic_fit(x, y, p0=None):
    def func(x, a, b, c):
        return 0.5 * (c / (1 + np.multiply(a, np.exp(np.multiply(-b, x))))) + 0.5

    if p0 is None:
        popt, pcov = curve_fit(func, x, y)
    else:
        popt, pcov = curve_fit(func, x, y, p0=p0)

    return func(x, *popt)


def plot_psychometric(x, y, yerr, yfit):
    plt.figure()
    plt.errorbar(x, y, yerr=yerr, fmt='kx', linewidth=1, capsize=2)
    plt.xscale('log')
    plt.semilogx(x, yfit, 'r-', linewidth=1)
    plt.xlabel('Correlation (%)')
    plt.ylabel('Proportion Correct')
    plt.xlim(0.001, 100)
    plt.ylim(0.4, 1.05)
    plt.grid()


def plot_ip(x, y):
    plt.figure()
    ips_mean = np.mean(y, axis=1)
    flag = True
    for ips_trp in np.transpose(y):
        if flag:
            plt.plot(x, ips_trp, 'b.', markersize=5, label='Individual Curves')
        else:
            plt.plot(x, ips_trp, 'b.', markersize=5)
        flag = False
    plt.plot(x, ips_mean, 'kx', markersize=10, label='Mean')
    plt.xlabel('Number of Neurons')
    plt.ylabel('Integrated Performance')
    plt.grid()
    plt.legend()


# config
n_neuron_list = [10, 20, 30, 40, 50, 75, 100, 150]
n_experiment = 20  # number of experiments for each number of neurons
n_corr = 30  # number of correlation points (x-resoltion)
n_test, n_time = 500, 200  # increase to decrease noise in neuronal performances

# main
ips_list = []
for n_neuron in n_neuron_list:
    print('Initiating experiments with n_neuron = {}'.format(n_neuron))
    ips = []
    for ei in range(n_experiment):
        print('Run experiment {} of {}'.format(ei + 1, n_experiment))
        corr, inputs, answers = generate_stimuli(n_corr, n_test, n_time)
        neurons = generate_neurons(n_neuron)
        responses = get_responses(inputs, neurons)
        neurometric, psychometric, psychometric_errors = calc_neuropsycho(responses, answers)
        psychometric_fit = logistic_fit(corr, psychometric, p0=[50, 0.5, 1])
        # plot_psychometric(corr, psychometric, psychometric_errors, psychometric_fit)

        ips.append(simps(psychometric_fit, x=corr))
        if ei == 0:
            corr_global = corr
    ips_list.append(ips)

plot_ip(n_neuron_list, ips_list)
plt.show()
