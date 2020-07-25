#!/usr/bin/env python

import numpy as np
from numpy.random import default_rng
from matplotlib import pyplot as plt
from wirelesscomms.modulation.fsk import FskModem
from wirelesscomms.channels import awgn

"""A script to show the relationship between probability of error and SNR for an FSK modem."""

if __name__ == '__main__':

    # FSK Settings
    sym_map = {'0': 1, '1': 0}
    Rb = 57600 # bit rate
    sps= 4
    gaussian = False
    # these are only used if gaussian is True
    tbw = 1 # time bandwidth product
    filter_len = 3 # gaussian filter size in units of symbols

    # Channel Settings
    snr_vec = np.arange(-10, 11, 1)

    # Data Settings
    payload_length = 2048 # bits
    num_trials = 3

    # construct the modem
    modem = FskModem(bit_rate=Rb,
                     samples_per_symbol=sps,
                     symbol_mapping=sym_map,
                     gaussian=gaussian,
                     time_bw_product=tbw,
                     filter_len=filter_len)

    # random number generator
    rng = default_rng(22) # seed for repeatability


    pbe = np.zeros(len(snr_vec)) # initialize vector to hold results
    for idx, snr in enumerate(snr_vec):
        print('SNR =', snr)
        for _ in range(num_trials):
            # generate random data
            tx_data = rng.integers(0, 2, payload_length)
            # modulate the data
            tx_wfm = modem.modulate(tx_data)
            noisy_wfm = awgn(tx_wfm, snr, measured=True)
            rx_data = [int(s) for s in modem.demodulate(noisy_wfm)]
            pbe[idx] += np.sum(np.bitwise_xor(tx_data, rx_data)) / payload_length

    pbe /= num_trials # average out results

    plt.semilogy(snr_vec, pbe)
    plt.grid()
    plt.title('Probability of Bit Error vs SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate')
    plt.tight_layout()
    plt.show()
