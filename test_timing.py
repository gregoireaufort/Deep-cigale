#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 13:30:38 2021

@author: aufort
"""

import SED_statistical_analysis
import scipy.stats as stats
from utils import *
import pcigale.sed_modules
from GMM import Mixture_gaussian, Mixture_gaussian_discrete
from pcigale.warehouse import SedWarehouse
#from pcigale.data import SimpleDatabase
#from pcigale.data import Database

import pandas as pd
import astropy
import numpy as np

np.random.seed(42)


A=astropy.io.fits.open("/home/aufort/Desktop/jorge/results.fits")
B = A[1].data
galaxy_targ = B[0]



fit_jorge = {"tau_main" : galaxy_targ["best.sfh.tau_main"],
                'age_main':galaxy_targ["best.sfh.age_main"],
                  'tau_burst':galaxy_targ["best.sfh.tau_burst"],
                  'f_burst':galaxy_targ["best.sfh.f_burst"],
                  'age_burst':galaxy_targ["best.sfh.age_burst"],
                  'E_BV_lines':galaxy_targ["best.attenuation.E_BV_lines"]}


galaxy_obs = SED_statistical_analysis.read_galaxy_fits("observations.fits", 
                 str(galaxy_targ["id"])+"_best_model.fits",
                 ident = galaxy_targ["id"])



bands = list(galaxy_obs["bands"])
pcigale.sed_modules.get_module('deep_bc03_pca_norm')
module_list = ['deep_sfhdelayed', 'deep_bc03_pca_norm','nebular','dustatt_modified_starburst','dl2014', 'redshifting']
path_deep = '/home/aufort/Bureau/cigale_bpt/cigale-cigale-BPT/pcigale/data/'
file_store = 'store_parameters_'+str(galaxy_targ["id"])+'_deep.csv'
deep_modules = [pcigale.sed_modules.deep_bc03_pca_norm]
module_parameters_to_fit = {'tau_main': {"type":"unif","min":1500,"max" :3000},
            'age_main': {"type":"unif","min":1000,"max" :10000},
            'tau_burst':{"type":"unif","min":100,"max" :10000},
            'f_burst': {"type":"unif","min":0,"max" :0.2},
            'age_burst': {"type":"unif","min":10,"max" :100},
            'E_BV_lines' : {"type":"unif","min":0,"max" :2},
    
}

module_parameters_discrete = {'sfr_A' : [1.],
                             'normalise' : [True],
                             #'E_BV_lines':[galaxy_targ["best.attenuation.E_BV_lines"]],
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
                             "zgas": [galaxy_targ["best.nebular.zgas"]],
                             "logU": [galaxy_targ["best.nebular.logU"]],
                             "ne":[100],
                             "beta_calz94":[ True],
                             "D4000":[False],
                             "IRX":[True],
                             "EW_lines":["500.7/1.0 & 656.3/1.0"],
                             "luminosity_filters":["FUV & V_B90"],
                             "colours_filters": ["FUV-NUV & NUV-r_prime"],
}
wavelength_limits = {"min" : 645,"max" : 1800}
wavelength_lines =np.array([121.60000000000001,133.5,139.7, 154.9, 164.0, 166.5, 190.9,232.6, 279.8, 372.7, 379.8, 383.5, 386.9, 388.9, 397.0, 407.0, 410.2, 434.0, 486.1, 495.9, 500.7, 630.0, 654.8,656.3, 658.4, 671.6, 673.1])
nebular_params = {"lines_width" : module_parameters_discrete["lines_width"][0],"lines_waves" : wavelength_lines}
CIGALE_parameters = {"module_list":module_list,
                    "path_deep" : path_deep,
                    "file_store":file_store,
                    "deep_modules":deep_modules,
                    "module_parameters_to_fit":module_parameters_to_fit,
                    "module_parameters_discrete":module_parameters_discrete,
                    "n_bins":20,
                    "wavelength_limits" : wavelength_limits,
                    "nebular" :nebular_params,
                    "bands" :bands,
                    "mode" : ["spectro"],
                    "n_jobs" : 10}

dim_prior = len(CIGALE_parameters["module_parameters_to_fit"]) #Number of continuous parameters to fit
n_comp = 4 #arbitrary
ESS_tol = 300*dim_prior 
proposal = Mixture_gaussian_discrete
T_max = 3
n_sample = [1000]*T_max
alpha = 300

#NEED TO AUTOMATE THIS PART, USELESS TO SET UP
var0 = [3]*dim_prior
mean0 = 0
init_mean = stats.uniform.rvs(size =(n_comp,dim_prior),loc=-1,scale = 2 )
probs = 31*[[1]]
init = [init_mean,
         np.array([np.diag(var0)]*n_comp),
         np.ones((n_comp,))/n_comp,
         probs]
init_theta= theta_params_discrete(init)

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
result = SED_statistical_analysis.fit(galaxy_obs,
                                      CIGALE_parameters, TAMIS_parameters)


# targ_covar = SED_statistical_analysis.extract_target(galaxy_obs,CIGALE_parameters)

# target_photo,target_lines, = targ_covar[0:2]
# target_spectro,covar_photo = targ_covar[2:4]
# covar_spectro, covar_lines = targ_covar[4:6]

# sample = result[0].sample

# pre_computed_constants =SED_statistical_analysis.scale_factor_pre_computation(target_photo ,
#                                                                   covar_photo,
#                                                                   target_spectro,
#                                                                   covar_spectro,
#                                                                   CIGALE_parameters["mode"])



# scaled_SED_photo,scaled_SED_spectro,scaled_SED_lines= compute_scaled_SED(sample,
#                                                                  pre_computed_constants,
#                                                                  1,
#                                                                  CIGALE_parameters,
#                                                                  warehouse)