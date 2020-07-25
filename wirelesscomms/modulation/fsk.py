#!/usr/bin/env python

from typing import Dict, Union
import numpy as np
from scipy import signal
from wirelesscomms.modulation.modem import DigitalModem
from wirelesscomms import digital_formatting as dfmt

# Disable certain pylint warnings
# pylint: disable=arguments-differ

###### Type Hints #####
RealNumber = Union[int, float]
NumpyArray = np.ndarray
RealNumpyArray = np.ndarray
ComplexNumpyArray = np.ndarray
Binary = Union[str, bytes, bytearray]
DigitalData = Union[Binary, RealNumpyArray]

##### Public module variables #####
SUPPORTED_MODULATION_ORDERS = (2, 4, 8, 16)

##### Private module variables #####
_bits_per_byte = 8
_bits_per_nibble = 4
_nibbles_per_byte = 2
_default_symbol_mapping = {'0' : 1, '1' : -1}
# using a dictionary to return bits/symb is 33x faster than using np.log2(mod_order)
_bits_per_symbol_dict = {order : int(np.log2(order)) for order in SUPPORTED_MODULATION_ORDERS}

class FskModem(DigitalModem):

    def __init__(self, \
                 bit_rate: RealNumber = 9600,
                 samples_per_symbol: int = 10,
                 symbol_mapping: Dict[str, int] = None,
                 freq_sep: float = None,
                 gaussian: bool = False,
                 time_bw_product: float = 1.0,
                 filter_len: int = 1):
        super().__init__(bit_rate, samples_per_symbol, symbol_mapping)
        self.gaussian = gaussian
        self.time_bw_product = time_bw_product
        self.filter_len = filter_len
        if freq_sep is not None:
            self.freq_sep = freq_sep
        else:
            self.use_msk(gaussian=self.gaussian)

    def __call__(self, data):
        if isinstance(data, (str, bytes, bytearray)):
            return self.modulate(data)

        if isinstance(data, NumpyArray): # ambiguous type (mod or demod?)
            return self.modulate(data) if dfmt.is_binary(data) else self.demodulate(data)

        raise TypeError('Expecting type str, bytes, bytearray, or NumpyArray. ' + \
                        f'Given {data.__class__.__name__}')

    def __repr__(self):
        return f'{self.__class__.__name__}(' + \
               f'bit_rate={self._bit_rate}, samples_per_symbol={self._samples_per_symbol}, ' + \
               f'symbol_mapping={self._symbol_mapping}, ' + \
               f'gaussian={self._gaussian}, time_bw_product={self._time_bw_product})'

    def __str__(self):
        return f'{self.__class__.__name__}(' + \
               f'bit_rate={self._bit_rate}, samples_per_symbol={self._samples_per_symbol}, ' + \
               f'symbol_mapping={self._symbol_mapping}, ' + \
               f'gaussian={self._gaussian}, time_bw_product={self._time_bw_product})'

    @property
    def bit_rate(self) -> float:
        """float: Bit rate in bits per second."""
        return self._bit_rate

    @bit_rate.setter
    def bit_rate(self, rb: RealNumber):
        if not isinstance(rb, (int, float)):
            raise TypeError(f'Expecting type int or float. Given {rb.__class__.__name__}')
        self._bit_rate = float(rb)

    @property
    def freq_sep(self) -> float:
        """float: Frequency separation between adjacent FSK tones."""
        return self._freq_sep

    @freq_sep.setter
    def freq_sep(self, freq_sep: float):
        if not isinstance(freq_sep, dfmt.REAL_NUMBER_TYPES):
            raise TypeError(f'Expecting type float. Given {freq_sep.__class__.__name__}.')
        if freq_sep <= 0:
            raise ValueError('Frequency separation must be a positive value greater than zero.')
        self._freq_sep = float(freq_sep)

    @property
    def samples_per_symbol(self) -> int:
        """int: Samples per symbol."""
        return self._samples_per_symbol

    @samples_per_symbol.setter
    def samples_per_symbol(self, sps):
        if not isinstance(sps, (int, float)):
            raise TypeError(f'Expecting type int. Given {sps.__class__.__name__}.')
        if sps % 1 != 0:
            raise ValueError(f'Samples per symbol must be a whole number. Given {sps}.')
        self._samples_per_symbol = int(sps)

    @property
    def symbol_mapping(self) -> Dict[str, int]:
        """Dict[str, float]: Mapping of binary digits to modulation symbols.

        Note:
            Symbols must be in the range 0 to mod_order-1.

        Example:
            >>> import numpy as np
            >>> mapping = {'00': 0, '01': 1, '10': 3, '11': 2}
            >>> Rb = 19200 # bits per second
            >>> sps = 10 # samples per symbol
            >>> modem = fsk.FskModem(bit_rate=Rb, samples_per_symbol=sps, symbol_mapping=mapping)
            >>> # frequency separation defaults to symbol rate
            >>> freqs = np.arange(-(modem.mod_order-1)/2, (modem.mod_order-1)/2) * modem.freq_sep
            >>> freqs
            ...
            -14400.0
            -4800.0
            4800.0
            14400.0
        """
        return self._symbol_mapping

    @symbol_mapping.setter
    def symbol_mapping(self, mapping: Dict[str, int]) -> None:
        if mapping is None:
            mapping = _default_symbol_mapping

        elif not isinstance(mapping, dict):
            raise TypeError(f'Expecting type {Dict[str, int].__str__().replace("typing.", "")}.' + \
                            f' Given {mapping.__class__.__name__}.')

        if len(mapping) not in SUPPORTED_MODULATION_ORDERS:
            raise ValueError('Number of symbols is not supported. This module supports ' + \
                             f'modulation orders {SUPPORTED_MODULATION_ORDERS}')

        bad_types = [(k.__class__.__name__, v.__class__.__name__) for k, v in mapping.items() \
                    if not (isinstance(k, str) and isinstance(v, (int, float)))]
        if bad_types:
            raise TypeError(f'Expecting type {Dict[str, int].__str__().replace("typing.", "")}.' + \
                            f' Found items {bad_types}')

        bad_keys = [k for k in mapping.keys() if [b for b in k if b not in ('0', '1')]]
        if bad_keys:
            raise ValueError('All symbol map keys must be binary strings without the literal ' + \
                            f'"0b" (e.g. "00", "01", "10", "11"). Given key(s) {bad_keys}.')

        self._symbol_mapping = mapping
        self._mod_order = len(mapping)
        self._bits_per_symbol = _bits_per_symbol_dict[self._mod_order]
        self._symbol_rate = self._bit_rate / self._bits_per_symbol

    @property
    def sample_rate(self) -> float:
        """float: Sample rate in samples per second."""
        return self._samples_per_symbol * self._bit_rate / self._bits_per_symbol

    @property
    def mod_order(self) -> int:
        """int: Modulation order (e.g. 2, 4, 8, 16)."""
        return self._mod_order

    @property
    def symbol_rate(self) -> float:
        """float: Symbol rate in symbols per second."""
        return self._symbol_rate

    @property
    def bits_per_symbol(self) -> int:
        """int: Number of bits per symbol."""
        return self._bits_per_symbol

    @property
    def gaussian(self) -> bool:
        """bool: If True, uses a pre-modulation gaussian lowpass filter."""
        return self._gaussian

    @gaussian.setter
    def gaussian(self, tf: bool) -> None:
        if not isinstance(tf, bool):
            raise TypeError(f'Expecting type bool. Given {tf.__class__.__name__}')
        self._gaussian = tf

    @property
    def time_bw_product(self) -> float:
        """float: Symbol Time - Filter 3dB Bandwidth product."""
        return self._time_bw_product

    @time_bw_product.setter
    def time_bw_product(self, tbp: RealNumber):
        if not isinstance(tbp, (int, float)):
            raise TypeError(f'Expected type int or float. Given {tbp.__class__.__name__}')
        self._time_bw_product = float(tbp)

    @property
    def filter_len(self) -> int:
        """int: Length of the gaussian filter used for GFSK"""
        return self._filter_len

    @filter_len.setter
    def filter_len(self, flen: int):
        if not isinstance(flen, (int, float)):
            raise TypeError(f'Expecting type int. Given {flen.__class__.__name__}')
        if flen % 1 != 0:
            raise ValueError(f'Expecting a whole number. Given {flen}.')
        self._filter_len = int(flen)

    @property
    def freq_dev(self) -> float:
        return self.freq_sep / 2

    def modulate(self, data: DigitalData) -> ComplexNumpyArray:
        """Maps data to symbols then modulates those symbols using (G)FSK.

        See also: fsk.modulate, fsk.map_to_symbols
        """
        symbols = map_to_symbols(data, self.symbol_mapping)
        return modulate(symbols=symbols,
                        mod_order=self.mod_order,
                        freq_sep=self.freq_sep,
                        sps=self.samples_per_symbol,
                        sample_rate=self.sample_rate,
                        gaussian=self.gaussian,
                        tbw=self.time_bw_product,
                        filter_len=self.filter_len)

    def demodulate(self, iq: NumpyArray,
                   coherent: bool = False, initial_phase: float = 0.0) -> str:
        """[summary]

        Args:
            iq (NumpyArray): IQ data to demodulate.
            coherent (bool, optional): If True, uses a coherent demodulator, otherwise
                uses a non-coherent demodulator. Defaults to False.
            initial_phase (float, optional): Initial phase used in coherent
                demodulator. Defaults to 0.0.

        Returns:
            str: Binary string
        """

        symbols = demodulate(iq=iq,
                             sample_rate=self.sample_rate,
                             sps=self.samples_per_symbol,
                             mod_order=self.mod_order,
                             freq_sep=self.freq_sep,
                             coherent=coherent,
                             initial_phase=initial_phase)
        demapper = {v: k for k, v in self.symbol_mapping.items()}
        binary_arr = [demapper[s] for s in symbols]
        return ''.join(binary_arr)

    def use_msk(self, gaussian: bool = False, coherent: bool = False):
        """Auto-set modem for MSK

        Sets the frequency separation to half the symbol rate if coherent is True,
        otherwise set frequency separation to the symbol rate (default).
        Set gaussian to True to use GMSK.

        The coherent argument refers to the intended demodulation method. The smallest
        useable frequency separation is half the symbol rate for a coherent demodulator
        and the symbol rate for a non-coherent demodulator.
        """
        self.freq_sep = self.symbol_rate/2 if coherent else self.symbol_rate
        self.gaussian = gaussian


def map_to_symbols(data: DigitalData, mapping: dict) -> list:
    """Map bits to (G)FSK symbols

    Note:
        DigitalData = Union[str, bytes, bytearray, np.ndarray]

    Args:
        data (DigitalData): Data to map
        mapping (dict): Data mapping (e.g. {'0': -1, '1': 1})

    Returns:
        list: Symbols mapped from the given bits
    """
    binary_str = dfmt.to_binary_str(data, literal=False)
    bits_per_symbol = int(np.log2(len(mapping)))
    chunks = dfmt.chunk(binary_str, bits_per_symbol)
    return [mapping.get(chunk, None) for chunk in chunks]


def gaussian_window(tbw: float, Tsym: float, sps: int, nsym: int = 1) -> np.ndarray:
    """Generate filter coefficients for a gaussian lowpass filter used for GFSK

    Args:
        tbw (float): Symbol Time - Filter 3dB Bandwidth product.
        Tsym (float): Symbol period in seconds
        sps (int): Samples per symbol
        nsym (int, optional): Filter length in number of symbols. Defaults to 1.

    Returns:
        np.ndarray: [description]
    """
    B = tbw/Tsym # bandwidth of the filter
    t = np.arange(start=-nsym*Tsym, stop=nsym*Tsym + Tsym/sps, step=Tsym/sps)
    h = B*np.sqrt(2*np.pi/(np.log(2)))*np.exp(-1 * ((t*np.pi*B)**2) / np.log(2))
    return h / np.sum(h)


def modulate(symbols: list, mod_order: int, freq_sep: float, sps: int, sample_rate: float = 1,
             gaussian: bool = True, tbw: float = 1, filter_len: int = 1) -> np.ndarray:
    """Modulate a symbol stream using (G)FSK modulation.

    Args:
        syms (list, np.ndarray): List of symbols normalized to the symbol rate.
        sps (int): Number of samples per symbol.
        tbw (float): Symbol Time - Filter Bandwidth product
        gaussian (bool, optional): If True, passes symbols through a gaussian filter
            before modulating. Defaults to True.
        filter_len (int, optional): Length of the gaussian filter in number of symbols.
            Defaults to 1.

    Returns:
        np.ndarray: Complex (G)FSK modulated waveform
    """
    freq_map = np.arange(start=-(mod_order-1)/2, stop=(mod_order-1)/2 + 1) * freq_sep
    c = np.repeat([freq_map[s] for s in symbols], sps) # upsample by samples per symbol

    if gaussian:
        g = gaussian_window(tbw=tbw, Tsym=sps/sample_rate, sps=sps, nsym=filter_len) # Gaussian LPF
        c = np.convolve(c, g, mode='same') # perform filtering

    # integrate to get phase information
    phi_t = signal.lfilter(b=[1], a=[1, -1], x=c) * 2 * np.pi / sample_rate
    return (np.cos(phi_t) + 1j*np.sin(phi_t)) / np.sqrt(2) # I - jQ


def demodulate(iq: np.ndarray, sample_rate: float, sps: int, mod_order: int, freq_sep: float,
               coherent: bool = False, initial_phase: float = 0.0) -> np.ndarray:
    """Demodulate a (G)FSK signal

    Warning:
        Non-coherent demodulation has not yet been implemented

    Args:
        iq (np.ndarray): IQ data to demodulate.
        sample_rate (float): Sample rate of the IQ data.
        sps (int): Samples per symbol.
        mod_order (int): Modulation order (2, 4, 8, 16, etc.)
        freq_sep (float): Spacing between adjacent FSK tones in Hz.
        coherent (bool, optional): If True, uses coherent demodulation, otherwise uses non-coherent
            demodulation. Defaults to False.
        initial_phase (float, optional): Initial phase used for coherent demodulation.
            Defaults to 0.

    Returns:
        np.ndarray: Array of symbols from the demodulation of given IQ data.

    See Also: modulate
    """

    if coherent: # coherent demodulation
        # TODO Implement coherent demodulation
        raise NotImplementedError('Coherent demodulation has not been implemented yet.')
    else: # non-coherent demodulation
        # create basis frequencies
        freqs = np.arange(-(mod_order-1)/2, (mod_order-1)/2 + 1) * freq_sep
        bases = np.zeros((mod_order, len(iq)), dtype=np.dtype(np.complex128))
        t = np.arange(len(iq)) / sample_rate
        for idx, f in enumerate(freqs):
            bases[idx, :] = np.exp(-2j * np.pi * f * t)

        # create IQ channels for each basis frequency
        iq_channels = np.reshape(np.tile(iq, mod_order), (mod_order, len(iq)))
        iq_channels = np.multiply(iq_channels, bases) # correlate
        # integrate and dump
        integrator_output = np.zeros((mod_order, len(iq)//sps))
        for idx, chan in enumerate(iq_channels):
            integrator_output[idx, :] = np.abs(np.convolve(chan, np.ones(sps), mode='valid'))[::sps]
        # hard decision
        return np.argmax(integrator_output, axis=0) # symbols in range 0 to mod_order-1
