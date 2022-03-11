#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 17:53:50 2022

@author: aufort
"""



import SED_statistical_analysis
import scipy.stats as stats
from utils import *
import pcigale.sed_modules
from GMM import Mixture_gaussian
from pcigale.warehouse import SedWarehouse
import pandas as pd
import astropy
import numpy as np

np.random.seed(42)


A=astropy.io.fits.open("/home/aufort/Desktop/jorge/results.fits")
names = A[1].data.columns.names
best = [name for name in names if name[:4]=='best']
best_names = [x.rsplit('.', 1)[-1] for x in best]


C = pd.DataFrame.from_dict( {n:A[1].data[n].byteswap().newbyteorder() for n in best}) #FFS astropy formats
C.columns = [x.rsplit('.', 1)[-1] for x in C.columns]
C["sfr_A"]=1.0
C["normalise"] = True
C["separation_age"] = 10
C["ne"] = 100
C["emission"] = True
C["Ext_law_emission_lines"] = 1
C["Rv"] = 3.1
C["filters"] = "B_B90 & V_B90 & FUV"
C["beta_calz94"] = True
C["D4000"] = False
C["IRX"] = True
C["EW_lines"] = "500.7/1.0 & 656.3/1.0"
C["luminosity_filters"] = "FUV & V_B90"
C["colours_filters"] = "FUV-NUV & NUV-r_prime"
module_list = ['sfhdelayed', 'bc03','nebular','dustatt_modified_starburst','dl2014', 'restframe_parameters','redshifting']
warehouse = SedWarehouse()
modules_params = [list(pcigale.sed_modules.get_module(module,blank = True).parameter_list.keys()) for module in module_list]
parameter_list =[[{param: C[param].iloc[i] for param in module_params} for module_params in modules_params] for i in range( C.shape[0])] 
SEDS = [ warehouse.get_sed(module_list,i) for i in parameter_list]
    
bands = ["galex.NUV",
         "cfht.megacam.u", 
         "subaru.suprime.B", 
         "subaru.suprime.V", 
         "subaru.suprime.r",
         "subaru.suprime.i",
         "subaru.suprime.z",
         "subaru.hsc.y", 
         "UKIRT_WFCJ", 
         "cfht.wircam.H",
         "WFCAM_K",
         "IRAC1", 
         "IRAC2", 
         "IRAC3",
         "IRAC4"]
lines = ["line.Ly-alpha", "line.HeII-164.0", "line.OIII-166.5", 
         "line.CIII-190.9", "line.MgII-279.8","line.OII-372.7",
          "line.H-9", "line.NeIII-386.9", "line.HeI-388.9", " line.H-8", 
          "line.NeIII-396.8","line.H-epsilon",  "line.H-delta", "line.H-gamma", 
          "line.OIII-436.3","line.H-beta","line.OIII-495.9", 
         "line.OIII-500.7","line.HeI-587.6",  "line.OI-630.0",
          "line.NII-654.8", "line.H-alpha","line.NII-658.4", 
          "line.SII-671.6", "line.SII-673.1", "line.ArIII-713.6"]

B = SEDS[0]
photo2 = np.array([B.compute_fnu(band) for band in bands])
B.wavelength_grid/(1+B.info["universe.redshift"])

spectro = np.array([SED.fnu for SED in SEDS])
alpha = A[1].data["best.sfh.integrated"]
scaled_spectro =np.array([alpha[i]*spectro[i,:] for i in range(491)]) # pourquoi il veut pas broadcast ? 
spectro_wavelength = np.array([SED.wavelength_grid for SED in SEDS])
scaled_photo = np.array([array[194:209] for array in A[1].data])
scaled_lines = np.array([array[209:] for array in A[1].data])
photo_store = pd.DataFrame(scaled_photo, columns = bands)
spectro_store = pd.DataFrame(scaled_spectro)
wavelength_store = pd.DataFrame(spectro_wavelength)
lines_store = pd.DataFrame(scaled_lines, columns = lines)
redshift = pd.DataFrame(np.array([SED.info["universe.redshift"] for SED in SEDS]))

photo_store.to_csv("test_Jorge/photo.csv")
spectro_store.to_csv("test_Jorge/spectro.csv")
wavelength_store.to_csv("test_Jorge/wavelength_spectro.csv")
lines_store.to_csv("test_Jorge/lines.csv")
redshift.to_csv("test_Jorge/redshift.csv")
