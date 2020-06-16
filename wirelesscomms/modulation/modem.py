#!/usr/bin/env python

from abc import ABC, abstractmethod

class Modem(ABC):
    """Abstract Modem"""

    @abstractmethod
    def modulate(self, data):
        """Abstract modulation method

        :param data: Digital data to modulate
        :return: Waveform modulated with data
        """

    @abstractmethod
    def demodulate(self, waveform):
        """Abstract demodulation method

        :param waveform: Waveform to demodulate
        :return: Digital data
        """

class DigitalModem(Modem):
    """Abstract Digital Modem"""

    def __init__(self, bit_rate, samples_per_symbol, symbol_mapping):
        self.bit_rate = bit_rate
        self.samples_per_symbol = samples_per_symbol
        self.symbol_mapping = symbol_mapping

    @property
    @abstractmethod
    def bit_rate(self):
        """Abstract property bit rate."""

    @property
    @abstractmethod
    def samples_per_symbol(self):
        """Abstract property samples per symbol."""

    @property
    @abstractmethod
    def symbol_mapping(self):
        """Abstract symbol mapping."""

    @property
    @abstractmethod
    def mod_order(self):
        """Abstract modulation order."""

    @abstractmethod
    def modulate(self, data):
        """Abstract modulation method.

        Args:
            symbols: Symbol data to modulate

        Returns:
            array-like: Waveform modulated with data
        """

    @abstractmethod
    def demodulate(self, waveform):
        """Abstract demodulation method.

        Args:
        waveform: Waveform to demodulate

        Returns:
            array-like: Demodulated data
        """


class AnalogModem(Modem):

    @abstractmethod
    def modulate(self, data):
        """Abstract modulation method

        :param data: Digital data to modulate
        :return: Waveform modulated with data
        """

    @abstractmethod
    def demodulate(self, waveform):
        """Abstract demodulation method

        :param waveform: Waveform to demodulate
        :return: Digital data
        """
