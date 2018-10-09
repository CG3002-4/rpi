import matplotlib.pyplot as plt
from scipy import fftpack


def plot_data(data, title):
    """Plot xyz axis data"""
    plt.title(title)
    plt.plot(range(len(data)), data[:, 0], 'r-', label='x')
#    plt.plot(range(len(data)), data[:, 1], 'g-', label='y')
#    plt.plot(range(len(data)), data[:, 2], 'b-', label='z')


def plot_freq_spec(data, title):
    """Plot frequency spectrum of xyz axis data"""
    plt.title(title)

    def plot_freq_spec(axis, line, label):
        n = len(axis)
        fft = fftpack.fft(axis) / n
        fft = fft[range(int(n / 2))]
        plt.plot(range(int(n / 2)), abs(fft), line, label=label)
    plot_freq_spec(data[:, 0], 'r-', label='x')
    plot_freq_spec(data[:, 1], 'g-', label='y')
    plot_freq_spec(data[:, 2], 'b-', label='z')
