#!/usr/bin/env python
"""A module with plotting tools for signal visualization."""

from typing import Tuple
import numpy as np
from numpy.fft import fftshift, fft, fftfreq
from matplotlib import pyplot as plt

def frequency_units(data: np.ndarray) -> Tuple[str, float]:
    """Given a dataset, return the most appropriate frequency unit
        to use for plotting.

    Args:
        data (np.ndarray): Dataset to analyze

    Returns:
        Tuple[str, float]: (unit_str, scale_factor)
    """
    max_val = np.amax(np.abs(data))
    if max_val > 10**12:
        ret = ('THz', 10**12)
    elif max_val > 10**9:
        ret = ('GHz', 10**9)
    elif max_val > 10**6:
        ret = ('MHz', 10**6)
    elif max_val > 10**3:
        ret = ('kHz', 10**3)
    else:
        ret = ('Hz', 1)

    return ret


def time_units(data: np.ndarray) -> Tuple[str, float]:
    """Given a dataset, return the most appropriate time unit
        to use for plotting.

    Args:
        data (np.ndarray): Dataset to analyze

    Returns:
        Tuple[str, float]: (unit_str, scale_factor)
    """
    max_val = np.amax(np.abs(data))
    if max_val > 1:
        ret = ('s', 1)
    elif max_val > 1e-3:
        ret = ('ms', 1e-3)
    elif max_val > 1e-6:
        ret = ('us', 1e-6)
    elif max_val > 1e-9:
        ret = ('ns', 1e-9)
    elif max_val > 1e-12:
        ret = ('ps', 1e-12)
    else:
        ret = ('fs', 1e-15)

    return ret


def inst_freq(iq: np.ndarray, fs: float = 1,
              grid: bool = False, show: bool = False):
    """Plot instantaneous frequency vs time. If show is False, this function
        returns a Line2D object that can be used matplotlib. If show is True,
        this function shows the plot and returns None.

    Args:
        iq (np.ndarray): Complex data to process and plot
        fs (float, optional): Sample rate of the data. Defaults to 1.
        grid (bool, optional): If True, plot with a grid. Defaults to False.
        show (bool, optional): If True, plot will show and this function will
            return None. Defaults to False.

    Returns:
        matplotlib.lines.Line2D: The plot as a Line2D object
    """
    f = np.diff(np.unwrap(np.angle(iq))) * fs / (2 * np.pi)
    t = np.arange(len(f)) / fs
    units_f, scale_f = frequency_units(f)
    units_t, scale_t = time_units(t)

    p = plt.plot(t / scale_t, f / scale_f)
    plt.title('Frequency vs Time')
    plt.xlabel(f'Time ({units_t})')
    plt.ylabel(f'Frequency ({units_f})')
    if grid:
        plt.grid()
    if show:
        plt.show()
    else:
        return p


def power_spectrum(iq: np.ndarray, fs: float = 1, nfft: int = None,
                   nci: bool = False, log_scale: bool = True,
                   normalize: bool = False,
                   grid: bool = False, show: bool = False):
    """Plot the power spectrum of a complex dataset.

    Args:
        iq (np.ndarray): Complex data to process and plot.
        fs (float, optional): Sample rate of the data. Defaults to 1.
        nfft (int, optional): FFT size to use. If not specified it will use
            the length of the iq data. Defaults to 0.
        nci (bool, optional): If True, non-coherent integrations of size nfft will be used.
             Defaults to False.
        log_scale (bool, optional): If True, will plot the power spectrum in dB.
            Defaults to True.
        normalize (bool, optional): If True, normalizes the plot to 1 (or 0dB).
            Defaults to False.
        grid (bool, optional): If True, plots with a grid. Defaults to False.
        show (bool, optional): If True, shows the plot and returns None. Defaults to False.

    Returns:
        matplotlib.lines.Line2D: The plot as a Line2D object
    """

    if not nfft:
        nfft = len(iq)
    if nci:
        nframes = len(iq) // nfft
        nsamps = nframes * nfft
        x = np.reshape(iq[:nsamps], (nframes, nfft))
        X = np.abs(fftshift(fft(x, n=nfft, axis=1) / nfft, axes=1)) ** 2
        X = np.sum(X, axis=0)
    else:
        X = np.abs(fftshift(fft(iq, n=nfft) / nfft))

    if normalize:
        X = X / np.amax(X)

    if log_scale:
        X = 10 * np.log10(X)

    f = fftshift(fftfreq(nfft, d=1/fs))
    units, scale = frequency_units(f)

    yunit = ' (dB)' if log_scale else ''

    p = plt.plot(f / scale, X)
    plt.title('Power Spectrum')
    plt.xlabel(f'Frequency {units}')
    plt.ylabel('Magnitude' + yunit)
    if grid:
        plt.grid()
    if show:
        plt.show()
    else:
        return p


def time_domain(iq: np.ndarray, fs: float = 1, log_scale: bool = True,
                grid: bool = False, show: bool = False):
    """Plot power vs time of an complex signal.

    Args:
        iq (np.ndarray): Complex data to process and plot.
        fs (float, optional): Sample rate of the data. Defaults to 1.
        log_scale (bool, optional): If True, show the data on a log scale. Defaults to True.
        grid (bool, optional): If True, plot with a grid. Defaults to False.
        show (bool, optional): If True, show the plot and return None. Defaults to False.

    Returns:
        matplotlib.lines.Line2D: The plot as a Line2D object
    """

    power = np.abs(iq) ** 2
    if log_scale:
        power = 10 * np.log10(power)

    yunit = ' (dB)' if log_scale else ''
    t = np.arange(len(power)) / fs
    units, scale = ('samples', 1) if fs == 1 else time_units(t)

    p = plt.plot(t / scale, power)
    plt.title('Power vs Time')
    plt.xlabel(f'Time ({units})')
    plt.ylabel('Squared Magnitude' + yunit)
    if grid:
        plt.grid()
    if show:
        plt.show()
    else:
        return p
