#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 15:09:52 2022

@author: aufort
"""



import SED_statistical_analysis
import scipy.stats as stats
from utils import *
import pcigale.sed_modules
from GMM import Mixture_gaussian_discrete
from pcigale.warehouse import SedWarehouse
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
                             'metallicity' : [0.02,0.004],
                             'qpah' : [galaxy_targ["best.dust.qpah"]],
                             'umin' : [galaxy_targ["best.dust.umin"]],
                             'alpha' : [2],
                             'gamma' : [0.02],
                             'separation_age': [10],
                             'logU' :[galaxy_targ["best.nebular.logU"]],
                             'zgas':[0.007, 0.014],
                             'ne':[100],
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
                    "mode" : ["photo"],
                    "n_jobs" : 10}

dim_prior = len(CIGALE_parameters["module_parameters_to_fit"]) #Number of continuous parameters to fit
n_comp = 4 #arbitrary
ESS_tol = 300*dim_prior 
proposal = Mixture_gaussian_discrete
T_max = 10
n_sample = [800]*T_max
alpha = 100

#NEED TO AUTOMATE THIS PART, USELESS TO SET UP
var0 = [3]*dim_prior
mean0 = 0
init_mean = stats.uniform.rvs(size =(n_comp,dim_prior),loc=-1,scale = 2 )
# need to create a probability vector associated with each discrete parameter
tst = [len(module_parameters_discrete[name]) for name in module_parameters_discrete.keys()]
[[1/i]*i for i in tst]
probs = [[1/i]*i for i in tst]

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
np.random.seed(42)


# result = SED_statistical_analysis.fit(galaxy_obs , CIGALE_parameters, TAMIS_parameters)


module_list_normal = ['sfhdelayed', 'bc03','nebular','dustatt_modified_starburst','dl2014', 'redshifting']
file_store_normal = 'store_parameters_'+str(galaxy_targ["id"])+'_spectro_normal.csv'

CIGALE_parameters_normal = {"module_list":module_list_normal,
                    "path_deep" : path_deep,    
                    "file_store":file_store_normal,
                    "deep_modules":None,
                    "module_parameters_to_fit":module_parameters_to_fit,
                    "module_parameters_discrete":module_parameters_discrete,
                    "n_bins":10,
                    "wavelength_limits" : wavelength_limits,
                    "nebular" :nebular_params,
                    "bands" :bands,
                    "mode" : ["photo"],
                    "n_jobs" : 10}
np.random.seed(42)

result_normal = SED_statistical_analysis.fit(galaxy_obs , CIGALE_parameters_normal, TAMIS_parameters)



# SED_statistical_analysis.plot_result(CIGALE_parameters_normal,
#                                      line_dict_fit = fit_jorge,
#                                      title = "CIGALE Jorge 1 spectro")