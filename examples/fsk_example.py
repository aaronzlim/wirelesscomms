#!/usr/bin/env python
"""An example script to demonstrate the use of wirelesscomms.modulation.fsk.FskModem."""

import numpy as np
from numpy.random import default_rng
from matplotlib import pyplot as plt
from wirelesscomms.modulation import fsk
from wirelesscomms.channels import awgn
from wirelesscomms import plots

show_plots = True

if __name__ == '__main__':

    rng = default_rng(22) # with seed for repeatability

    # FSK Settings
    sym_map = {'0': 1, '1': 0}
    Rb = 57600 # bit rate
    samps_per_symb = 10
    tbw = 1 # time bandwidth product
    filter_len = 3 # gaussian filter size in units of symbols

    # non-idealities
    delay = samps_per_symb // 11 # samples
    freq_shift = 0 # Hz
    snr_req = 15 # required SNR in dB

    # Data Settings
    header = '00100010'
    num_bits = 10_000 # number of bits to send

    # Derived parameters
    bits_per_symbol = np.log2(len(sym_map))
    Rs = Rb/bits_per_symbol
    fs = samps_per_symb * Rs

    msk_modem = fsk.FskModem(bit_rate=Rb,
                             samples_per_symbol=samps_per_symb,
                             symbol_mapping=sym_map)
    msk_modem.use_msk()
    print('\nMSK Modem\n', msk_modem)

    gmsk_modem = fsk.FskModem(bit_rate=Rb,
                              samples_per_symbol=samps_per_symb,
                              symbol_mapping=sym_map,
                              gaussian=True,
                              time_bw_product=tbw,
                              filter_len=filter_len)
    gmsk_modem.use_msk(gaussian=True)
    print('\nGMSK Modem\n', gmsk_modem)

    # generate random data
    tx_data = header + ''.join([str(bit) for bit in \
                                  rng.integers(low=0, high=2,
                                               size=num_bits - len(header),
                                               dtype=np.dtype(int))])

    # modulate the data
    wm = msk_modem.modulate(tx_data)
    wg = gmsk_modem.modulate(tx_data)

    wm = awgn(wm, snr_req, measured=True)
    wg = awgn(wg, snr_req, measured=True)

    # apply delay
    wm = np.roll(wm, delay)
    wm[:delay] = wm[delay]
    wg = np.roll(wg, delay)
    wg[:delay] = wm[delay]

    # apply frequency shift
    t = np.arange(len(wm)) / fs
    wm = np.multiply(wm, np.exp(2j * np.pi * freq_shift * t))
    wg = np.multiply(wg, np.exp(2j * np.pi * freq_shift * t))

    # demodulate the data
    rx_data_msk = msk_modem.demodulate(wm)
    rx_data_gmsk = gmsk_modem.demodulate(wg)

    err_msk = [idx for idx, v in enumerate(zip(tx_data, rx_data_msk)) if v[0] != v[1]]
    err_gmsk = [idx for idx, v in enumerate(zip(tx_data, rx_data_gmsk)) if v[0] != v[1]]
    print(f'Pbe MSK : {round(100*len(err_msk)/len(rx_data_msk), 4)}%')
    print(f'Pbe GMSK: {round(100*len(err_gmsk)/len(rx_data_gmsk), 4)}%')

    if show_plots:
        # Plots
        plt.subplot(221)
        plots.power_spectrum(wm, fs=fs, nfft=2048, nci=True, normalize=True, grid=True)
        plt.title('MSK')

        plt.subplot(222)
        plots.inst_freq(wm[:20*samps_per_symb], fs=fs, grid=True)

        plt.subplot(223)
        plots.power_spectrum(wg, fs=fs, nfft=2048, nci=True, normalize=True, grid=True)
        plt.title('GMSK')

        plt.subplot(224)
        plots.inst_freq(wg[:20*samps_per_symb], fs=fs, grid=True)

        plt.tight_layout()
        plt.show()
