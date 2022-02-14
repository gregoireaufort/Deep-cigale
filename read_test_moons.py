#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 16:22:16 2022

@author: aufort
"""

from astropy.io import fits
import matplotlib.pyplot as plt
fits_image_filename ="test_moons/ID_302327_BC03_z0.69_v60.00_m24.1_nexp8_HR_RI.fits"

hdul = fits.open(fits_image_filename2)
noisy = hdul[1].data
wo_noise = hdul[4].data
noise_array =  hdul[8].data


plt.plot(noisy)
plt.plot(wo_noise)


plt.plot(noise_array)
plt.plot(wo_noise/noise_array)


fits_image_filename2 ="test_moons/ID_302327_BC03_z0.69_v60.00_m24.1_nexp1_HR_RI.fits"
hdul2 = fits.open(fits_image_filename2)
noisy2 = hdul2[1].data
wo_noise2 = hdul2[4].data
noise_array2 =  hdul2[8].data


plt.plot(noisy2)
plt.plot(wo_noise2)


plt.plot(noise_array2)
plt.plot(wo_noise2/noise_array2)


import numpy as np 
plt.plot(np.linspace(7650.0,8980.0,4096)/1.69,wo_noise)


