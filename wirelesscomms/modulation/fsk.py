#!/usr/bin/env python

from typing import Dict, Union
import numpy as np
from scipy import signal
from wirelesscomms.modulation.modem import DigitalModem
from wirelesscomms.tools import digital_formatting as dfmt

###### Type Hints #####
RealNumber = Union[int, float]
NumpyArray = np.ndarray
RealNumpyArray = np.ndarray
ComplexNumpyArray = np.ndarray
Binary = Union[str, bytes, bytearray]
SymbolMapping = Dict[str, RealNumber]
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

    def __init__(self, bit_rate: RealNumber = 9600, \
                 samples_per_symbol: int = 10,
                 symbol_mapping: SymbolMapping = None, \
                 gaussian: bool = False, time_bw_product: float = 0.3, \
                 filter_len: int = 3):
        super().__init__(bit_rate, samples_per_symbol, symbol_mapping)
        self.gaussian = gaussian
        self.time_bw_product = time_bw_product
        self.filter_len = filter_len

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
    def symbol_mapping(self) -> SymbolMapping:
        """Dict[str, float]: Mapping of binary digits to modulation symbols.

        Note:
            Symbols are normalized to the symbol rate (Rs).

        Example:
            >>> mapping = {'00': -3/2, '01': -1/2, '10': 1/2, '11': 3/2}
            >>> Rb = 19200 # bits per second
            >>> sps = 10 # samples per symbol
            >>> modem = fsk.FskModem(bit_rate=Rb, samples_per_symbol=sps, symbol_mapping=mapping)
            >>> # Frequencies will be:
            >>> for sym in mapping.values():
            ...     print(sym * modem.symbol_rate)
            ...
            -14400.0
            -4800.0
            4800.0
            14400.0
        """
        return self._symbol_mapping

    @symbol_mapping.setter
    def symbol_mapping(self, mapping: SymbolMapping) -> None:
        if mapping is None:
            mapping = _default_symbol_mapping

        elif not isinstance(mapping, dict):
            raise TypeError(f'Expecting type {SymbolMapping.__str__().replace("typing.", "")}. ' + \
                            f'Given {mapping.__class__.__name__}.')

        if len(mapping) not in SUPPORTED_MODULATION_ORDERS:
            raise ValueError('Number of symbols is not supported. This module supports ' + \
                             f'modulation orders {SUPPORTED_MODULATION_ORDERS}')

        bad_types = [(k.__class__.__name__, v.__class__.__name__) for k, v in mapping.items() \
                    if not (isinstance(k, str) and isinstance(v, (int, float)))]
        if bad_types:
            raise TypeError(f'Expecting type {SymbolMapping.__str__().replace("typing.", "")}.' + \
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

    def modulate(self, data: DigitalData) -> ComplexNumpyArray:
        """Maps data to symbols then modulates those symbols using (G)FSK.

        See also: fsk.modulate, fsk.map_to_symbols
        """
        symbols = map_to_symbols(data, self.symbol_mapping)
        return modulate(symbols=symbols,
                        sps=self.samples_per_symbol,
                        tbw=self.time_bw_product,
                        gaussian=self.gaussian,
                        filter_len=self.filter_len)

    def demodulate(self, waveform: NumpyArray) -> bytearray:
        """Not Implemented"""
        raise NotImplementedError

    def use_msk(self):
        """Auto-set modem for MSK

        Changes symbol mapping to the required symbols for MSK. Sets gaussian to False.

        See Also: fsk.FskModem.use_gmsk
        """
        # sort binary chunks from smallest to largest symbol (frequency)
        # e.g. Given mapping {'00' : 1, '01' : -2, '10' : -1, '11' : 2}
        #      return ['01', '10', '00', '11']
        sorted_chunks = [tup[0] for tup in sorted(self._symbol_mapping.items(), key=lambda x: x[1])]
        # freqs are normalized to the symbol rate
        freqs = [float(f) for f in
                 np.arange(start=-(self._mod_order - 1), stop=self._mod_order, step=2) / 2]
        self.symbol_mapping = dict(zip(sorted_chunks, freqs))
        self.gaussian = False

    def use_gmsk(self):
        """Auto-set modem for GMSK.

        Changes symbol mapping to the required symbols for MSK, Sets gaussian to True.

        See Also: fsk.FskModem.use_msk
        """
        self.use_msk()
        self.gaussian = True


def demodulate():
    """Not Implemented"""
    raise NotImplementedError


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
    h = B*np.sqrt(2*np.pi/(np.log(2)))*np.exp(-2 * (t*np.pi*B)**2 /(np.log(2)))
    return h / np.sum(h)


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


def modulate(symbols: list, sps: int, tbw: float,
             gaussian: bool = True, filter_len: int = 1) -> np.ndarray:
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
    c = np.repeat(symbols, sps) # upsample by samples per symbol
    if gaussian:
        g = gaussian_window(tbw=tbw, Tsym=1, sps=sps, nsym=filter_len) # Gaussian LPF
        b = np.convolve(c, g, mode='same') # perform filtering
    else:
        b = c
    # integrate to get phase information
    phi_t = signal.lfilter(b=[1], a=[1, -1], x=b) * 2 * np.pi / sps
    return (np.cos(phi_t) - 1j*np.sin(phi_t)) / np.sqrt(2) # I - jQ
