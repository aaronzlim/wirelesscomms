#!/usr/bin/env python

import numpy as np
from numpy import random
from wirelesscomms import digital_formatting as dfmt

def awgn(x: np.ndarray, snr: float, measured: bool = False) -> np.ndarray:
    """Send a signal through an additive white gaussian noise channel.

    Args:
        x (np.ndarray): Input signal
        snr (float): Required signal to noise ratio in dB
        measured (bool): If True, measures the power of x to calculate the noise power.
            If False, adds noise with variance 10 ** (-snr/10).

    Returns:
        np.ndarray: noisy signal
    """
    if not len(x):
        return x

    if measured:
        signal_power = 10 * np.log10(np.mean(np.abs(x) ** 2)) # RMS power
        noise_power = 10 ** ((signal_power - snr)/10)         # RMS power
        if isinstance(x[0], dfmt.COMPLEX_TYPES):
            noise = np.sqrt(noise_power/2) * (random.randn(*np.shape(x)) + 1j * random.randn(*np.shape(x)))
        else:
            noise = np.sqrt(noise_power) * random.randn(*np.shape(x))

    else:
        noise = (10 ** (-snr/20)) * random.randn(*np.shape(x)) # variance = 10 ** (-snr/10)

    return x + noise
