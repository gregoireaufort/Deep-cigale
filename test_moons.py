#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 16:22:16 2022

@author: aufort
"""

import SED_statistical_analysis
import scipy.stats as stats
from utils import *
import pcigale.sed_modules
from GMM import Mixture_gaussian
from pcigale.warehouse import SedWarehouse
from pcigale.data import Database
import pandas as pd
import astropy
import numpy as np

np.random.seed(42)


spec_filename ="test_moons/ID_302327_BC03_z0.69_v60.00_m24.1_nexp8_HR_RI.fits"


galaxy_obs = SED_statistical_analysis.read_galaxy_fits(None, 
                 spec_filename,
                 ident =None)
module_list = ['deep_sfhdelayed', 'deep_bc03_pca_norm','nebular','dustatt_modified_starburst','dl2014', 'redshifting']
path_deep = '/home/aufort/Desktop/cigale-master/params_comparison.txt'
file_store = 'store_parameters_'+str(galaxy_targ["id"])+'_deep.csv'
deep_modules = [pcigale.sed_modules.deep_bc03_pca_norm]
module_parameters_to_fit = {'tau_main': {"type":"unif","min":1500,"max" :3000},
            'age_main': {"type":"unif","min":1000,"max" :10000},
            'tau_burst':{"type":"unif","min":100,"max" :10000},
            'f_burst': {"type":"unif","min":0,"max" :0.2},
            'age_burst': {"type":"unif","min":10,"max" :100},
            #'E_BV_lines' : {"type":"unif","min":0,"max" :2},
    
}

module_parameters_discrete = {'sfr_A' : [1.],
                             'normalise' : [True],
                             'E_BV_lines':[galaxy_targ["best.attenuation.E_BV_lines"]],
                             'E_BV_factor' :  [0.44],
                             'uv_bump_wavelength' : [217.5],
                             'uv_bump_width' :[35.0],
                             'uv_bump_amplitude':[0.0],
                             'powerlaw_slope' : [0.0],
                             'Ext_law_emission_lines' : [1],
                             'Rv' : [3.1],
                             'imf' : [1],
                             'metallicity' : [0.02],
                             'qpah' : [galaxy_targ["best.dust.qpah"]],
                             'umin' : [galaxy_targ["best.dust.umin"]],
                             'alpha' : [2],
                             'gamma' : [0.02],
                             'separation_age': [10],
                             'logU' :[galaxy_targ["best.nebular.logU"]],
                             'f_esc': [0.0],
                             'f_dust' : [0.0],
                             'lines_width' :[300.0],
                             'emission' :[True],
                             'redshift':[galaxy_obs["redshift"]],
                             'filters':["B_B90 & V_B90 & FUV"],
}
wavelength_limits = {"min" : 645,"max" : 1800}
wavelength_lines =[121.60000000000001,133.5,139.7, 154.9, 164.0, 166.5, 190.9,232.6, 279.8, 372.7, 379.8, 383.5, 386.9, 388.9, 397.0, 407.0, 410.2, 434.0, 486.1, 495.9, 500.7, 630.0, 654.8,656.3, 658.4, 671.6, 673.1]
nebular_params = {"lines_width" : module_parameters_discrete["lines_width"][0],"line_waves" : wavelength_lines}
CIGALE_parameters = {"module_list":module_list,
                    "path_deep" : path_deep,
                    "file_store":file_store,
                    "deep_modules":deep_modules,
                    "module_parameters_to_fit":module_parameters_to_fit,
                    "module_parameters_discrete":module_parameters_discrete,
                    "n_bins":100,
                    "wavelength_limits" : wavelength_limits,
                    "nebular" :nebular_params,
                    "bands" :bands,
                    "mode" : ["spectro"],
                    "n_jobs" : 10}

dim_prior = len(CIGALE_parameters["module_parameters_to_fit"]) #Number of continuous parameters to fit
n_comp = 4 #arbitrary
ESS_tol = 300*dim_prior 
proposal = Mixture_gaussian
T_max = 30
n_sample = [1000]*T_max
alpha = 100

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
                    'alpha':alpha,
                    "verbose" : True
    
}
# result = SED_statistical_analysis.fit(galaxy_obs,
#                                       CIGALE_parameters, TAMIS_parameters)

