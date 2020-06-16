#!/usr/bin/env python

import unittest
from wirelesscomms.modulation import fsk

class TestFskModem(unittest.TestCase):

    def setUp(self):
        self.fsk_modem = fsk.FskModem()

    def test_samples_per_symbol(self):
        sps_required = 17
        sps_wrong_type = '19'
        sps_negative = -20

        self.fsk_modem.samples_per_symbol = sps_required
        self.assertEqual(self.fsk_modem.samples_per_symbol, sps_required)
        with self.assertRaises(TypeError):
            self.fsk_modem.samples_per_symbol = sps_wrong_type
        with self.assertRaises(ValueError):
            self.fsk_modem.samples_per_symbol = sps_negative

    # TODO Test each property
