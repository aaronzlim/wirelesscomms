#!/usr/bin/env python
"""An example script to demonstrate the use of wirelesscomms.modulation.fsk.FskModem."""

import numpy as np
from numpy.random import default_rng
from matplotlib import pyplot as plt
from wirelesscomms.modulation import fsk
from wirelesscomms.tools import plots


if __name__ == '__main__':

    rng = default_rng(22) # with seed for repeatability

    # FSK Settings
    sym_map = {'0': -1, '1': 1}
    Rb = 57600 # bit rate
    samps_per_symb = 10
    tbw = 0.5 # time bandwidth product

    # Data Settings
    header = '000101'
    num_bits = 10_000 # number of bits to send

    # Derived parameters
    bits_per_symbol = np.log2(len(sym_map))
    Rs = Rb/bits_per_symbol
    fs = samps_per_symb * Rs

    msk_modem = fsk.FskModem(bit_rate=Rb,
                             samples_per_symbol=samps_per_symb,
                             symbol_mapping=sym_map)
    msk_modem.use_msk()

    gmsk_modem = fsk.FskModem(bit_rate=Rb,
                              samples_per_symbol=samps_per_symb,
                              symbol_mapping=sym_map,
                              time_bw_product=tbw)
    gmsk_modem.use_gmsk()

    # generate random data
    test_data = header + ''.join([str(bit) for bit in \
                                  rng.integers(low=0, high=2,
                                               size=num_bits - len(header),
                                               dtype=np.dtype(int))])

    # modulate the data
    wm = msk_modem.modulate(test_data)
    wg = gmsk_modem.modulate(test_data)

    # Plots
    plt.subplot(221)
    plots.power_spectrum(wm, fs=fs, nfft=2048, nci=True, normalize=True, grid=True)
    plt.title('MSK')

    plt.subplot(222)
    plots.inst_freq(wm[:10*samps_per_symb], fs=fs, grid=True)

    plt.subplot(223)
    plots.power_spectrum(wg, fs=fs, nfft=2048, nci=True, normalize=True, grid=True)
    plt.title('GMSK')

    plt.subplot(224)
    plots.inst_freq(wg[:10*samps_per_symb], fs=fs, grid=True)

    plt.tight_layout()
    plt.show()
