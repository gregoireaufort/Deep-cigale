#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 17:37:49 2021

@author: aufort
"""


import SED_statistical_analysis
import importlib
import scipy.stats as stats
import pcigale.sed_modules
from GMM import GMM_fit, Mixture_t, Mixture_gaussian,multivariate_t
from TAMIS import TAMIS
import random
from pcigale.warehouse import SedWarehouse
import pcigale_filters
from utils import *

import numpy as np
np.random.seed(42)


galaxy_obs = SED_statistical_analysis.read_galaxy("/home/aufort/Desktop/photo_test.csv",
                                                  "/home/aufort/Desktop/spectro_test.csv")

bands = list(galaxy_obs["bands"])
pcigale.sed_modules.get_module('deep_bc03_pca_norm')
module_list = ['deep_sfhdelayed', 'deep_bc03_pca_norm','nebular','dustatt_modified_starburst','dl2014', 'redshifting']
path_deep = '/home/aufort/Desktop/cigale-master/params_comparison.txt'
file_store = '/home/aufort/Desktop/cigale-master/store_parameters_test.csv'
deep_modules = [pcigale.sed_modules.deep_bc03_pca_norm]
module_parameters_to_fit = {'tau_main': {"type":"log","min":2,"max" :5},
            'age_main': {"type":"unif","min":500,"max" :10000},
            'tau_burst':{"type":"unif","min":100,"max" :10000},
            'f_burst': {"type":"unif","min":0,"max" :0.05},
            'age_burst': {"type":"log","min":1,"max" :2},
    
}

module_parameters_discrete = {'sfr_A' : [1.],
                             'normalise' : [True],
                             'E_BV_lines' : [0.1],
                             'E_BV_factor' :  [0.44],
                             'uv_bump_wavelength' : [217.5],
                             'uv_bump_width' :[35.0],
                             'uv_bump_amplitude':[0.0],
                             'powerlaw_slope' : [0.0],
                             'Ext_law_emission_lines' : [1],
                             'Rv' : [3.1],
                             'imf' : [1],
                             'metallicity' : [0.02],
                             'qpah' : [2.5],
                             'umin' : [1.0],
                             'alpha' : [2],
                             'gamma' : [0.5],
                             'separation_age': [10],
                             'logU' :[-3.5, -2.5, -1.5],
                             'f_esc': [0.0],
                             'f_dust' : [0.0],
                             'lines_width' :[300.0],
                             'emission' :[True],
                             'redshift':[1],
                             'filters':["B_B90 & V_B90 & FUV"],
}
wavelength_limits = {"min" : 645,"max" : 1800}
wavelength_lines =[121.60000000000001,
 133.5,
 139.70000000000002,
 154.9,
 164.0,
 166.5,
 190.9,
 232.60000000000002,
 279.8,
 372.70000000000005,
 379.8,
 383.5,
 386.90000000000003,
 388.90000000000003,
 397.0,
 407.0,
 410.20000000000005,
 434.0,
 486.1,
 495.90000000000003,
 500.70000000000005,
 630.0,
 654.8000000000001,
 656.3000000000001,
 658.4000000000001,
 671.6,
 673.1]
nebular_params = {"lines_width" : module_parameters_discrete["lines_width"][0],"line_waves" : wavelength_lines}
CIGALE_parameters = {"module_list":module_list,
                    "path_deep" : path_deep,
                    "file_store":file_store,
                    "deep_modules":deep_modules,
                    "module_parameters_to_fit":module_parameters_to_fit,
                    "module_parameters_discrete":module_parameters_discrete,
                    "n_bins":10,
                    "wavelength_limits" : wavelength_limits,
                    "nebular" :nebular_params,
                    "bands" :bands,
                    "n_jobs" : 4}

dim_prior = len(CIGALE_parameters["module_parameters_to_fit"]) #Number of continuous parameters to fit
n_comp = 4 #arbitrary
ESS_tol = 300*dim_prior 
proposal = Mixture_gaussian
T_max = 50
n_sample = [500]*T_max

#NEED TO AUTOMATE THIS PART, USELESS TO SET UP
var0 = [3]*dim_prior
mean0 = 0
init_mean = stats.uniform.rvs(size =(n_comp,dim_prior),loc=-1,scale = 2 )
init = [init_mean,
         np.array([np.diag(var0)]*n_comp),
         np.ones((n_comp,))/n_comp]
init_theta= theta_params(init)

TAMIS_parameters = {'dim_prior' : dim_prior,
                    'n_comp' : n_comp,
                    'ESS_tol' : ESS_tol,
                    'proposal' : proposal,
                    'T_max' : T_max,
                    'n_sample' : n_sample,
                    'init_theta' : init_theta,
    
}
result = SED_statistical_analysis.fit(galaxy_obs , CIGALE_parameters, TAMIS_parameters)
