#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 16:22:16 2022

@author: aufort
"""

from astropy.io import fits
import matplotlib.pyplot as plt
fits_image_filename ="test_moons/ID_302327_BC03_z0.69_v60.00_m24.1_nexp8_HR_RI.fits"

hdul = fits.open(fits_image_filename)
noisy = hdul[1].data
wo_noise = hdul[4].data
noise_array =  hdul[8].data


plt.plot(noisy)
plt.plot(wo_noise)


