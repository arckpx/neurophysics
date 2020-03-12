import numpy as np
import matplotlib.pyplot as plt

dt = 1  # ms
time = np.arange(-10_000, 10_000, dt)
n_time = len(time)
firing_rate = np.ones(n_time)
firing_rate[5_000: 15_000] = 9 * np.cos(time[5_000:15_000] / 10_000 * np.pi) + 1
# Firing Rate:
# plt.plot(time, firing_rate)

n_trial = 50
neural_input = np.random.rand(n_trial, n_time)
threshold = np.ones(n_time) - firing_rate * (0.001 * dt)
spikes = np.array([(x > threshold) for x in neural_input])
# Peristimulus Time Histogram:
# psth = np.sum(spikes, axis=0)
# plt.plot(time, psth)

# Interspike Interval (mean, stdev, var, cv)
data = [[], []]
for i in range(n_trial):
    ranges = [spikes[i, 1:5_000], spikes[i, 5_000:15_000]]
    for j in range(2):
        isi = np.diff(np.where(ranges[j])) * dt
        if isi.size <= 1:  # NaN handling
            data[j].append([0, 0, 0, 0])
        else:
            isi_mean = np.mean(isi)
            isi_std = np.std(isi)
            data[j].append([isi_mean, isi_std, np.var(isi), isi_std / isi_mean])
data = np.array(data)

# 0 to 5 seconds
x0 = data[0, :, 0]  # mean
y0 = data[0, :, 1]  # std
p0 = np.polyfit(x0, y0, 1)  # linear fit
xy0 = np.linspace(0, np.max(data[0, :, 0:2]), num=2)  # y=x line
linreg0 = np.polyval(p0, xy0)

# 5 to 15 seconds
x1 = data[1, :, 0]  # mean
y1 = data[1, :, 1]  # std
p1 = np.polyfit(x1, y1, 1)  # linear fit
xy1 = np.linspace(1, np.max(data[1, :, 0:2]), num=2)  # y=x line
linreg1 = np.polyval(p1, xy1)


def plot_isi(x, y, xy, linreg, title='plot'):
    # isi_mean should approximately be equal to isi_std
    # linreg should approximately follow y=x
    plt.figure()
    plt.plot(x, y, 'kx', label='Data')
    plt.plot(xy, linreg, 'b-', linewidth=1, label='Linear Regression')
    plt.plot(xy, xy, 'r-', linewidth=1, label=r'$y=x$')
    plt.xlabel('Mean of ISI')
    plt.ylabel('Std of ISI')
    plt.title(title)
    plt.grid()
    plt.legend()


plot_isi(x0, y0, xy0, linreg0, title=r'From $t=0$s to $t=5$s')
plot_isi(x1, y1, xy1, linreg1, title=r'From $t=5$s to $t=15$s')

# Autocorrelation
spikes_vector = np.reshape(spikes, np.multiply(spikes.shape[0], spikes.shape[1]))


def autocorr(x, half_domain_size):
    n_corr = 2 * len(x) + 1
    ft = np.fft.fft(x, n_corr)
    ift = np.fft.ifft(np.multiply(ft, np.conj(ft)))
    shifted = np.fft.fftshift(ift)
    corr = np.divide(shifted, (shifted[len(x)])).real

    lags = np.arange(-len(spikes_vector), len(spikes_vector))
    center = np.where(lags == 0)[0][0]

    corr[center] = np.mean(list(corr[:center]) + list(corr[center + 1:]))

    return lags[center - half_domain_size:center + half_domain_size], corr[center - half_domain_size:center + half_domain_size]


def plot_autocorr(x, y):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel('Lags in ms')
    plt.ylabel('Autocorrelation')


lags, corr = autocorr(spikes_vector, 100_000)
plot_autocorr(lags, corr)

plt.show()
