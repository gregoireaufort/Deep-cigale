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
from pcigale import warehouse
import pandas as pd
import astropy
import numpy as np

np.random.seed(42)


A=astropy.io.fits.open("/home/aufort/Desktop/jorge/results.fits")
names = A[1].data.columns.names
best = [name for name in names if name[:4]=='best']
best_names = [x.rsplit('.', 1)[-1] for x in best]

photo = [array[63:78] for array in A[1].data]

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
module_list = ['sfhdelayed', 'bc03','nebular','dustatt_modified_starburst','dl2014', 'redshifting']
warehouse = SedWarehouse()
modules_params = [list(pcigale.sed_modules.get_module(module,blank = True).parameter_list.keys()) for module in module_list]
parameter_list =[[{param: C[param].iloc[i] for param in module_params} for module_params in modules_params] for i in range( C.shape[0])] 
SEDS = [ warehouse.get_sed(module_list,i) for i in parameter_list]
    
B = A[1].data[0]
B["best.sfh.tau_main"]

