#!/usr/bin/env python
"""Tools to handle the formatting of digital data."""

from typing import Union, Iterable, Generator
import numpy as np

DigitalData = Union[str, bytes, bytearray, list]

REAL_INTEGER_TYPES = (int, np.uint8, np.int8, np.uint16, np.int16, np.short, np.uint32, np.int32,
                      np.uint64, np.int64)
REAL_FLOAT_TYPES = (float, np.single, np.double, np.longdouble, np.float32, np.float64, np.float)
REAL_NUMBER_TYPES = REAL_INTEGER_TYPES + REAL_FLOAT_TYPES

COMPLEX_TYPES = (complex, np.complex, np.complex128, np.complex64,
                 np.csingle, np.cdouble, np.longdouble)

NIBBLES_PER_BYTE: int = 2
BITS_PER_NIBBLE: int = 4
BITS_PER_BYTE: int = 8

#: Set of all valid hexadecimal characters
HEX_CHARS_SET: set = set('0123456789abcdefABCDEF')
#: Set of all valid binary characters ('0', '1')
BINARY_CHARS_SET: set = set('01')

def chunk(it: Iterable, n: int) -> Generator:
    """Split an iterable into chunks of size n.

    Args:
        it (Iterable): An iterable to chunk
        n (int): Number of elements in one chunk

    Raises:
        TypeError: When n is not an integer

    Yields:
        Generator: Each chunk of size *n* in *it*
    """
    if not isinstance(n, int):
        raise TypeError(f'Expecting n to be of type int. Given {n.__class__.__name__}.')
    for idx in range(0, len(it), n):
        yield it[idx : idx + n]


def is_hex(data: str) -> bool:
    """Check if a string is a hexadecimal string.

    Args:
        data (str): The string to check

    Returns:
        bool: True if the string is a hex string, False if not

    Example:
        >>> not_hex_str = 'this is not a hex string'
        >>> hex_str = '0xdeadbeef'
        >>> is_hex(not_hex_str)
        False
        >>> is_hex(hex_str)
        True

    Note:
        This function will return true for binary strings (e.g. 0b01001000).
        To ignore binary strings you can use digital_formatting.is_binary.

    Example:
        >>> bin_str = '0b0001001010001
        >>> hex_str = '0xdeadbeef
        >>> is_hex(bin_str) and not is_bin(bin_str)
        False
        >>> is_hex(hex_str) and not is_bin(hex_str)
        True

    See Also: digital_formatting.is_binary
    """
    d = data[2:] if data[:2] in ('0x', '0X') else data
    if set(d.lower()) - HEX_CHARS_SET:
        return False
    return True


def is_binary(data: Union[str, np.ndarray]) -> bool:
    """Check if a data stream is binary.

    Args:
        data (Union[str, np.ndarray]): Data to check

    Returns:
        bool: True if data is binary (all 0 and/or 1), False if not binary

    Example:
        >>> bin_str = '0b001001010100'
        >>> not_bin_str = 'this is not a binary string.'
        >>> is_binary(bin_str)
        True
        >>> is_binary(not_bin_str)
        False

    See Also: digital_formatting.is_hex
    """
    if isinstance(data, str):
        d = data[2:] if data[:2] in ('0b', '0B') else data
        return False if set(d) - BINARY_CHARS_SET else True

    elif isinstance(data, np.ndarray):
        return True if np.logical_or(data == 0, data == 1).all() else False


def hex_to_bin(hex_str: str) -> str:
    """Convert a hex string to a binary string.
        The returned binary string will contain the prefix '0b' only
        if given a hex string with the prefix '0x'.

    Args:
        hex_str (str): Hexadecimal string to convert to binary

    Returns:
        str: Binary string converted from given hex string

    Example:
        >>> hex_str = '0xabcd'
        >>> hex_to_bin(hex_str)
        '0b1010101111001101'
        >>> hex_to_bin(hex_str[2:]) # remove '0b'
        '1010101111001101'
    """
    if not isinstance(hex_str, str):
        raise TypeError(f'Expecting type str. Given {hex_str.__class__.__name__}.')
    if hex_str[:2].lower() == '0x':
        literal = '0b'
        bin_len = len(hex_str[2:]) * BITS_PER_NIBBLE
    else:
        literal = ''
        bin_len = len(hex_str) * BITS_PER_NIBBLE
    return literal + bin(int(hex_str, 16))[2:].zfill(bin_len)


def bin_to_hex(bin_str: str) -> str:
    """Convert a binary string to a hex string.
        The returned hex string will contain the prefix '0x' only
        if given a binary string with the prefix '0b'.

    Args:
        bin_str (str): Binary string (e.g. '0b1001')

    Returns:
        str: Hexadecimal string zero-padded to len(bin_str) // 4

    Example:
        >>> bin_str = '0b1010101111001101'
        >>> bin_to_hex(bin_str)
        '0xabcd'
        >>> bin_to_hex(bin_str[2:]) # remove '0b'
        'abcd'
    """
    if not isinstance(bin_str, str):
        raise TypeError(f'Expecting type str. given {bin_str.__class__.__name__}.')
    literal = '0x' if bin_str[2:].lower() == '0b' else ''
    num_nibbles = len(bin_str) // BITS_PER_NIBBLE
    bin_str = bin_str[:num_nibbles * BITS_PER_NIBBLE] # truncate to whole number of nibbles
    return literal + hex(int(bin_str, 2))[2:].zfill(num_nibbles)


def to_binary_str(digital_data: DigitalData, literal=True) -> str:
    """Convert digital data (hex or binary) of type string, bytes, bytearray,
        or numpy array to a binary string.

    Args:
        digital_data (Union[str, bytes, bytearray, np.ndarray]): Data to convert
        literal (bool, optional): If True, prefaces the returned binary string with '0b'.
            Defaults to True.

    Returns:
        str: Binary string

    Example:
        >>> b = np.array([1, 0, 1, 1])
        >>> to_binary_str(b)
        '0b1011'
    """
    if isinstance(digital_data, str):
        if is_hex(digital_data) and not is_binary(digital_data):
            # must be hex
            bin_str = hex_to_bin(digital_data) if is_hex(digital_data) else digital_data
        else:
            bin_str = digital_data # if all 0 and 1, assume it's binary

    elif isinstance(digital_data, (bytes, bytearray)):
        bin_str = hex_to_bin(digital_data.hex())

    elif isinstance(digital_data, np.ndarray):
        if np.logical_or(digital_data == 0, digital_data == 1).all(): # check for non-binary elem's
            bin_str = ''.join([str(bit) for bit in digital_data])
        else:
            raise ValueError('Given numpy array with non-binary elements. ' + \
                             'Cannot convert to binary string.')

    elif isinstance(digital_data, list):
        if [True for bit in digital_data if bit not in (0, 1)]:
            raise ValueError('Given list with non-binary elements.')
        else:
            bin_str = ''.join([str(bit) for bit in digital_data])

    else:
        raise TypeError(f'Given unsupported data type {digital_data.__class__.__name__}.' + \
                        ' Valid types are str, bytes, bytearray, numpy.ndarray, or list.')

    # add or remove literal as necessary
    if literal and bin_str[:2] not in ('0b', '0B'):
        return '0b' + bin_str
    if not literal and bin_str[:2].lower() in ('0b', '0B'):
        return bin_str[2:]
    return bin_str


def to_binary_list(digital_data: DigitalData) -> list:
    """Convert digital data of type string, bytes, or bytearray into a list of binary values.

    Args:
        digital_data (Uniont[str, bytes, bytearray]): Data to convert

    Returns:
        list: List of binary values

    Example:
        >>> bin_str = '0b00110100'
        >>> to_binary_list(bin_str)
        [0, 0, 1, 1, 0, 1, 0, 0]
    """
    if isinstance(digital_data, str):
        if is_hex(digital_data) and not is_binary:
            digital_data = bin(int(digital_data, 16)) # convert to binary string
        if digital_data[:2] == '0b':
            digital_data = digital_data[2:] # remove literal
        bin_arr = [1 if c == '1' else 0 for c in digital_data]

    elif isinstance(digital_data, (bytes, bytearray)):
        bin_str = bin(int(digital_data.hex(), 16))[2:].zfill(NIBBLES_PER_BYTE)
        bin_arr = [1 if c == '1' else 0 for c in bin_str]

    elif isinstance(digital_data, np.ndarray):
        bin_arr = list(digital_data)

    elif isinstance(digital_data, list):
        bin_arr = digital_data # already a list

    else:
        raise TypeError(f'Unknown data type {digital_data.__class__.__name__}. Require binary' + \
                        '/hex values as a string, bytes, bytearray, or numpy array.')

    return bin_arr


def str_xor(x: str, y: str) -> str:
    if len(x) != len(y):
        raise ValueError(f'Inputs are different lengths {len(x)} and {len(y)}. Inputs must be the same length.')
