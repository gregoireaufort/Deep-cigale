#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 17:23:20 2021

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


A=astropy.io.fits.open("/home/aufort/Desktop/jorge/results.fits")
B = A[1].data
galaxy_targ = B[0]



fit_jorge = {"tau_main" : galaxy_targ["best.sfh.tau_main"],
                'age_main':galaxy_targ["best.sfh.age_main"],
                  'tau_burst':galaxy_targ["best.sfh.tau_burst"],
                  'f_burst':galaxy_targ["best.sfh.f_burst"],
                  'age_burst':galaxy_targ["best.sfh.age_burst"]}


galaxy_obs = SED_statistical_analysis.read_galaxy_fits("observations.fits", 
                 str(galaxy_targ["id"])+"_best_model.fits",
                 ident = galaxy_targ["id"])



bands = list(galaxy_obs["bands"])
pcigale.sed_modules.get_module('deep_bc03_pca_norm')
module_list = ['deep_sfhdelayed', 'deep_bc03_pca_norm','nebular','dustatt_modified_starburst','dl2014', 'redshifting']
path_deep = '/home/aufort/Desktop/cigale-master/params_comparison.txt'
file_store = 'store_parameters_'+str(galaxy_targ["id"])+'_deep.csv'
deep_modules = [pcigale.sed_modules.deep_bc03_pca_norm]
module_parameters_to_fit = {'tau_main': {"type":"unif","min":1500,"max" :3000},
            'age_main': {"type":"unif","min":1000,"max" :10000},
            'tau_burst':{"type":"unif","min":100,"max" :10000},
            'f_burst': {"type":"unif","min":0,"max" :0.2},
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
                             'qpah' : [0.47, 1.12, 1.77, 2.5],
                             'umin' : [5.0, 10.0, 25.0],
                             'alpha' : [2],
                             'gamma' : [0.02],
                             'separation_age': [10],
                             'logU' :[-3.5, -2.5, -1.5],
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
                    "n_bins":10,
                    "wavelength_limits" : wavelength_limits,
                    "nebular" :nebular_params,
                    "bands" :bands,
                    "mode" : ["photo"],
                    "n_jobs" : 16}

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
result = SED_statistical_analysis.fit(galaxy_obs,
                                      CIGALE_parameters, TAMIS_parameters)



SED_statistical_analysis.plot_result(CIGALE_parameters,
                                     line_dict_fit = fit_jorge,
                                     title = "Deep jorge 1")



module_list_normal = ['sfhdelayed', 'bc03','nebular','dustatt_modified_starburst','dl2014', 'redshifting']
file_store_normal = 'store_parameters_'+str(galaxy_targ["id"])+'_normal.csv'

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
result_normal = SED_statistical_analysis.fit(galaxy_obs , CIGALE_parameters_normal, TAMIS_parameters)

SED_statistical_analysis.plot_result(CIGALE_parameters_normal,
                                     line_dict_fit = fit_jorge,
                                     title = "CIGALE Jorge 1")
# SED_statistical_analysis.analyse_results(CIGALE_parameters_normal)



galaxy_targ_2 = B[1]


fit_jorge_2 = {"tau_main" : galaxy_targ_2["best.sfh.tau_main"],
                'age_main':galaxy_targ_2["best.sfh.age_main"],
                  'tau_burst':galaxy_targ_2["best.sfh.tau_burst"],
                  'f_burst':galaxy_targ_2["best.sfh.f_burst"],
                  'age_burst':galaxy_targ_2["best.sfh.age_burst"]}


galaxy_obs_2= SED_statistical_analysis.read_galaxy_fits("observations.fits", 
                 str(galaxy_targ_2["id"])+"_best_model.fits",
                 ident = galaxy_targ_2["id"])

file_store = 'store_parameters_'+str(galaxy_targ_2["id"])+'_deep.csv'
CIGALE_parameters_2=CIGALE_parameters.copy()
CIGALE_parameters_2["file_store"]=file_store
result2 = SED_statistical_analysis.fit(galaxy_obs_2,
                                      CIGALE_parameters_2, TAMIS_parameters)


SED_statistical_analysis.plot_result(CIGALE_parameters_2,
                                     line_dict_fit = fit_jorge_2 ,
                                     title = "Deep jorge 2")


file_store_normal_2 = 'store_parameters_'+str(galaxy_targ_2["id"])+'_normal.csv'
CIGALE_parameters_normal_2 = {"module_list":module_list_normal,
                    "path_deep" : path_deep,
                    "file_store":file_store_normal_2,
                    "deep_modules":None,
                    "module_parameters_to_fit":module_parameters_to_fit,
                    "module_parameters_discrete":module_parameters_discrete,
                    "n_bins":10,
                    "wavelength_limits" : wavelength_limits,
                    "nebular" :nebular_params,
                    "bands" :bands,
                    "mode" : ["photo"],
                    "n_jobs" : 10}

result_normal_2 = SED_statistical_analysis.fit(galaxy_obs_2,
                                      CIGALE_parameters_normal_2, TAMIS_parameters)




SED_statistical_analysis.plot_result(CIGALE_parameters_normal_2,
                                     line_dict_fit = fit_jorge_2 ,
                                     title = "Normal jorge 2")

def plot_best_SED(CIGALE_parameters,obs):
    
    
    results_read = pd.read_csv(CIGALE_parameters["file_store"])
    param_frame = results_read[results_read["MAP"]==1].iloc[0].to_dict()
    modules_params = [list(pcigale.sed_modules.get_module(module,blank = True).parameter_list.keys()) for module in CIGALE_parameters['module_list']]
    parameter_list =[{param:param_frame[param] for param in module_params} for module_params in modules_params]
    db_filters = Database()
    filters = [db_filters.get_filter(name = band) for band in obs["bands"]]
    wave_to_plot = [filtr.pivot_wavelength for filtr in filters]
    warehouse = SedWarehouse(nocache = module_list)
    SED = SED_statistical_analysis.cigale(parameter_list, CIGALE_parameters,warehouse)
    
    targ_covar = SED_statistical_analysis.extract_target(obs,CIGALE_parameters)
    target_photo, target_lines, = targ_covar[0:2]
    target_spectro, covar_photo = targ_covar[2:4]
    covar_spectro, covar_lines = targ_covar[4:6]
    constants = SED_statistical_analysis.scale_factor_pre_computation(target_photo ,
                                                                     covar_photo,
                                                                     target_spectro,
                                                                     covar_spectro,
                                                                     CIGALE_parameters["mode"])
    weight_spectro = 1
    SED_photo = SED[0]
    SED_spectro = SED[1]
    SED_lines = SED[2]
    constant = SED_statistical_analysis.compute_constant(SED_photo, SED_spectro,constants,weight_spectro)
    scaled_photo = constant*SED_photo
    scaled_spectro = constant*SED_spectro
    scaled_lines = constant*SED_lines
    
    yerr = obs['photometry_err']*1.96
    plt.errorbar(x=wave_to_plot,y=obs['photometry_fluxes'],yerr=yerr)
    plt.plot(wave_to_plot,scaled_photo)
    plt.xscale("log")
    print(np.sum(((scaled_photo-galaxy_obs['photometry_fluxes'])/galaxy_obs['photometry_err'])**2))
    return None
plot_best_SED(CIGALE_parameters_normal,galaxy_obs)
plot_best_SED(CIGALE_parameters_normal,galaxy_obs)


bands_jorge = ["best."+band for band in bands]
pred_jorge = [galaxy_targ[band] for band in bands_jorge]

def plot_jorge(obs,pred_jorge):
    db_filters = Database()
    filters = [db_filters.get_filter(name = band) for band in obs["bands"]]
    wave_to_plot = [filtr.pivot_wavelength for filtr in filters]
    yerr = obs['photometry_err']*1.96
    plt.errorbar(x=wave_to_plot,y=obs['photometry_fluxes'],yerr=yerr)
    plt.plot(wave_to_plot,pred_jorge)
    plt.xscale("log")
    print(np.sum(((pred_jorge-obs['photometry_fluxes'])/obs['photometry_err'])**2))
    return None

plot_jorge(galaxy_obs,pred_jorge)
print(np.sum(((pred_jorge-galaxy_obs['photometry_fluxes'])/galaxy_obs['photometry_err'])**2))
print(np.sum(((pred_jorge-galaxy_obs['photometry_fluxes'])/galaxy_obs['photometry_err'])**2))


def plot_jorge_2(obs,params_jorge):
    results_read = pd.read_csv(CIGALE_parameters["file_store"])
    param_frame = params_jorge
    modules_params = [list(pcigale.sed_modules.get_module(module,blank = True).parameter_list.keys()) for module in CIGALE_parameters['module_list']]
    parameter_list =[{param:param_frame[param] for param in module_params} for module_params in modules_params]
    db_filters = Database()
    filters = [db_filters.get_filter(name = band) for band in obs["bands"]]
    wave_to_plot = [filtr.pivot_wavelength for filtr in filters]
    warehouse = SedWarehouse(nocache = module_list)
    SED = SED_statistical_analysis.cigale(parameter_list, CIGALE_parameters,warehouse)
    
    targ_covar = SED_statistical_analysis.extract_target(obs,CIGALE_parameters)
    target_photo, target_lines, = targ_covar[0:2]
    target_spectro, covar_photo = targ_covar[2:4]
    covar_spectro, covar_lines = targ_covar[4:6]
    constants = SED_statistical_analysis.scale_factor_pre_computation(target_photo ,
                                                                     covar_photo,
                                                                     target_spectro,
                                                                     covar_spectro,
                                                                     CIGALE_parameters["mode"])
    weight_spectro = 1
    SED_photo = SED[0]
    SED_spectro = SED[1]
    SED_lines = SED[2]
    constant = SED_statistical_analysis.compute_constant(SED_photo, SED_spectro,constants,weight_spectro)
    scaled_photo = constant*SED_photo
    scaled_spectro = constant*SED_spectro
    scaled_lines = constant*SED_lines
    
    yerr = obs['photometry_err']*1.96
    plt.errorbar(x=wave_to_plot,y=obs['photometry_fluxes'],yerr=yerr)
    plt.plot(wave_to_plot,scaled_photo)
    plt.xscale("log")
    print(np.sum(((scaled_photo-galaxy_obs['photometry_fluxes'])/galaxy_obs['photometry_err'])**2))
    return None
params= pd.read_csv(CIGALE_parameters["file_store"]).keys()[0:-2]
param_jorge = ["best."+param for param in params]
