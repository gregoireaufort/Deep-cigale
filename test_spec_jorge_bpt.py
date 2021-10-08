#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 14:56:32 2021

@author: aufort
"""




import SED_statistical_analysis
import scipy.stats as stats
from utils import *
import pcigale.sed_modules
from GMM import Mixture_gaussian
from pcigale.warehouse import SedWarehouse
from pcigale.data import SimpleDatabase
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
path_deep = '/home/aufort/Desktop/cigale-master/params_comparison.txt'
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
                             "E_BV_lines":[galaxy_targ["best.attenuation.E_BV_lines"]],
                             "ne":[100],
                             "beta_calz94":[ True],
                             "D4000":[False],
                             "IRX":[True],
                             "EW_lines":["500.7/1.0 & 656.3/1.0"],
                             "luminosity_filters":["FUV & V_B90"],
                             "colours_filters": ["FUV-NUV & NUV-r_prime"],
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



# SED_statistical_analysis.plot_result(CIGALE_parameters,
#                                      line_dict_fit = fit_jorge,
#                                      title = "Deep jorge 1")



module_list_normal = ['sfhdelayed', 'bc03','nebular','dustatt_modified_starburst','dl2014', 'redshifting']
file_store_normal = 'store_parameters_'+str(galaxy_targ["id"])+'_spectro_normal.csv'


n_bins = 20

CIGALE_parameters_normal = {"module_list":module_list_normal,
                    "path_deep" : path_deep,
                    "file_store":file_store_normal,
                    "deep_modules":None,
                    "module_parameters_to_fit":module_parameters_to_fit,
                    "module_parameters_discrete":module_parameters_discrete,
                    "n_bins":n_bins,
                    "wavelength_limits" : wavelength_limits,
                    "nebular" :nebular_params,
                    "bands" :bands,
                    "mode" : ["spectro"],
                    "n_jobs" : 10}
# result_normal = SED_statistical_analysis.fit(galaxy_obs , CIGALE_parameters_normal, TAMIS_parameters)

# SED_statistical_analysis.plot_result(CIGALE_parameters_normal,
#                                      line_dict_fit = fit_jorge,
#                                      title = "CIGALE Jorge 1 spectro")
# SED_statistical_analysis.analyse_results(CIGALE_parameters_normal)



import astropy

A=astropy.io.fits.open("/home/aufort/Desktop/jorge/results.fits")
B = A[1].data
B[0]["id"]
B[0]["best.sfh.tau_main"]
B[0]["best.sfh.age_main"]
B[0]["best.sfh.tau_burst"]
B[0]["best.sfh.f_burst"]
B[0]["best.sfh.age_burst"]



line_dict_fit = {"log(tau_main)" : np.log10(B[0]["best.sfh.tau_main"]),
                'age_main':B[0]["best.sfh.age_main"],
                  'tau_burst':B[0]["best.sfh.tau_burst"],
                  'f_burst':B[0]["best.sfh.f_burst"],
                  'log(age_burst)':np.log10(B[0]["best.sfh.age_burst"])}

# SED_statistical_analysis.plot_result(CIGALE_parameters, line_dict_fit)


module_list_normal = ['sfhdelayed', 'bc03','nebular','dustatt_modified_starburst','dl2014', 'restframe_parameters','redshifting']
file_store_normal = 'store_parameters_test_normal_spectro_bpt_correct.csv'
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
CIGALE_parameters_normal = {"module_list":module_list_normal,
                    "path_deep" : path_deep,
                    "file_store":file_store_normal,
                    "deep_modules":None,
                    "module_parameters_to_fit":module_parameters_to_fit,
                    "module_parameters_discrete":module_parameters_discrete,
                    "n_bins":20,
                    "wavelength_limits" : wavelength_limits,
                    "nebular" :nebular_params,
                    "bands" :bands,
                    "mode" : ["spectro"],
                    "n_jobs" : 10}
result_normal = SED_statistical_analysis.fit(galaxy_obs , CIGALE_parameters_normal, TAMIS_parameters)

SED_statistical_analysis.plot_result(CIGALE_parameters_normal,
                                      line_dict_fit = fit_jorge,
                                      title = "CIGALE Jorge 1 spectro")
# wave = galaxy_obs["spectroscopy_wavelength"]
# err = galaxy_obs["spectroscopy_err"]**2
# lim_wave, lim_err = SED_statistical_analysis.limit_spec( galaxy_obs["spectroscopy_wavelength"],
#                                 galaxy_obs["spectroscopy_err"],
#                                 CIGALE_parameters['wavelength_limits']["min"],
#                                 CIGALE_parameters['wavelength_limits']["max"])

# _, lim_flux = SED_statistical_analysis.limit_spec( galaxy_obs["spectroscopy_wavelength"],
#                                 galaxy_obs["spectroscopy_fluxes"],
#                                 CIGALE_parameters['wavelength_limits']["min"],
#                                 CIGALE_parameters['wavelength_limits']["max"])


# # n_bins = 200

# yerr =lim_err *1.96
# plt.plot(lim_wave,lim_flux)
# plt.fill_between(lim_wave,lim_flux - yerr ,lim_flux + yerr,alpha = 0.5)
# plt.yscale("log")


# bin_wave , bin_flux = SED_statistical_analysis.binning_flux(lim_wave, lim_flux, n_bins)
# bin_wave,bin_err = SED_statistical_analysis.binning_variances(lim_wave, lim_err**2, n_bins)


# yerr =lim_err *1.96
# plt.plot(lim_wave,lim_flux)
# plt.plot(galaxy_obs["spectroscopy_wavelength"],galaxy_obs["spectroscopy_fluxes"],)
# plt.fill_between(lim_wave,lim_flux - yerr ,lim_flux + yerr,alpha = 0.5)
# plt.errorbar(x=bin_wave,y=bin_flux,yerr=1.96*np.sqrt(bin_err), color = "red")
# plt.yscale("log")
# plt.xscale("log")


def plot_best_SED(CIGALE_parameters,obs):
    
    
    results_read = pd.read_csv(CIGALE_parameters["file_store"])
    param_frame = results_read[results_read["MAP"]==1].iloc[0].to_dict()
    print(param_frame)
    modules_params = [list(pcigale.sed_modules.get_module(module,blank = True).parameter_list.keys()) for module in CIGALE_parameters['module_list']]
    parameter_list =[{param:param_frame[param] for param in module_params} for module_params in modules_params]
    lim_wave, lim_err = SED_statistical_analysis.limit_spec( galaxy_obs["spectroscopy_wavelength"],
                                galaxy_obs["spectroscopy_err"],
                                CIGALE_parameters['wavelength_limits']["min"],
                                CIGALE_parameters['wavelength_limits']["max"])
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
    _, lim_flux = SED_statistical_analysis.limit_spec( galaxy_obs["spectroscopy_wavelength"],
                                galaxy_obs["spectroscopy_fluxes"],
                                CIGALE_parameters['wavelength_limits']["min"],
                                CIGALE_parameters['wavelength_limits']["max"])
    wave_to_plot , bin_flux = SED_statistical_analysis.binning_flux(lim_wave, 
                                                                    lim_flux, 
                                                                    n_bins,
                                                                    CIGALE_parameters['wavelength_limits']["min"],
                                                                    CIGALE_parameters['wavelength_limits']["max"])

    bin_wave,bin_err = SED_statistical_analysis.binning_variances(lim_wave,
                                                                  lim_err**2, 
                                                                  n_bins,
                                                                  CIGALE_parameters['wavelength_limits']["min"],
                                                                  CIGALE_parameters['wavelength_limits']["max"])

    weight_spectro = 1
    SED_photo = SED[0]
    SED_spectro = SED[1]
    SED_lines = SED[2]
    constant = SED_statistical_analysis.compute_constant(SED_photo, SED_spectro,constants,weight_spectro)
    scaled_photo = constant*SED_photo
    scaled_spectro = constant*SED_spectro
    scaled_lines = constant*SED_lines
    
    yerr = np.array(np.sqrt(bin_err))*1.96
    plt.errorbar(x=wave_to_plot,y=bin_flux,yerr=yerr, color = "red")
    plt.plot(wave_to_plot,scaled_spectro)
    plt.xscale("log")
    return None
plot_best_SED(CIGALE_parameters_normal,galaxy_obs)

def plot_best_no_bins(CIGALE_parameters,obs):
    results_read = pd.read_csv(CIGALE_parameters["file_store"])
    param_frame = results_read[results_read["MAP"]==1].iloc[0].to_dict()
    modules_params = [list(pcigale.sed_modules.get_module(module,blank = True).parameter_list.keys()) for module in CIGALE_parameters['module_list']]
    parameter_list =[{param:param_frame[param] for param in module_params} for module_params in modules_params]
    lim_wave, lim_err = SED_statistical_analysis.limit_spec( galaxy_obs["spectroscopy_wavelength"],
                                galaxy_obs["spectroscopy_err"],
                                CIGALE_parameters['wavelength_limits']["min"],
                                CIGALE_parameters['wavelength_limits']["max"])
        
    warehouse = SedWarehouse(nocache = module_list)
    SED = SED_statistical_analysis.cigale(parameter_list, CIGALE_parameters,warehouse)
    SED_complete = warehouse.get_sed(module_list_normal, parameter_list)  
    targ_covar = SED_statistical_analysis.extract_target(obs,CIGALE_parameters)
    target_photo, target_lines, = targ_covar[0:2]
    target_spectro, covar_photo = targ_covar[2:4]
    covar_spectro, covar_lines = targ_covar[4:6]
    constants = SED_statistical_analysis.scale_factor_pre_computation(target_photo ,
                                                                      covar_photo,
                                                                      target_spectro,
                                                                      covar_spectro,
                                                                      CIGALE_parameters["mode"])
    _, lim_flux = SED_statistical_analysis.limit_spec( galaxy_obs["spectroscopy_wavelength"],
                                galaxy_obs["spectroscopy_fluxes"],
                                CIGALE_parameters['wavelength_limits']["min"],
                                CIGALE_parameters['wavelength_limits']["max"])

    weight_spectro = 1
    SED_photo = SED[0]
    SED_spectro = SED[1]
    SED_lines = SED[2]
    constant = SED_statistical_analysis.compute_constant(SED_photo, SED_spectro,constants,weight_spectro)
    scaled_photo = constant*SED_photo
    scaled_spectro = constant*SED_spectro
    scaled_lines = constant*SED_lines
    scaled_spectro_complete = constant * SED_complete.fnu
    
    lim_wve,lim_flx =  SED_statistical_analysis.limit_spec( SED_complete.wavelength_grid,
                                                           scaled_spectro_complete,
                                CIGALE_parameters['wavelength_limits']["min"],
                                CIGALE_parameters['wavelength_limits']["max"])

    
    plt.plot(lim_wave,lim_flux, color = "red")
    plt.plot(lim_wve,lim_flx)
    return None


plot_best_no_bins(CIGALE_parameters_normal,galaxy_obs)

def plot_jorge_2(obs,params_jorge):
    param_frame = params_jorge
    modules_params = [list(pcigale.sed_modules.get_module(module,blank = True).parameter_list.keys()) for module in CIGALE_parameters_normal['module_list']]
    parameter_list =[{param:param_frame[param] for param in module_params} for module_params in modules_params]
    
    warehouse = SedWarehouse(nocache = module_list)
    SED = warehouse.get_sed(module_list_normal, parameter_list)  
    targ_covar = SED_statistical_analysis.extract_target(obs,CIGALE_parameters_normal)
    target_photo, target_lines, = targ_covar[0:2]
    target_spectro, covar_photo = targ_covar[2:4]
    covar_spectro, covar_lines = targ_covar[4:6]
    constants = SED_statistical_analysis.scale_factor_pre_computation(target_photo ,
                                                                     covar_photo,
                                                                     target_spectro,
                                                                     covar_spectro,
                                                                     CIGALE_parameters_normal["mode"])
    lim_wave, lim_flux = SED_statistical_analysis.limit_spec( galaxy_obs["spectroscopy_wavelength"],
                                galaxy_obs["spectroscopy_fluxes"],
                                CIGALE_parameters['wavelength_limits']["min"],
                                CIGALE_parameters['wavelength_limits']["max"])
    wave_to_plot , bin_flux = SED_statistical_analysis.binning_flux(lim_wave, lim_flux, n_bins)

    bin_wave,bin_err = SED_statistical_analysis.binning_variances(lim_wave, lim_err**2, n_bins)

    weight_spectro = 1
    SED_photo = SED[0]
    SED_spectro = SED[1]
    SED_lines = SED[2]
    constant = galaxy_targ["best.sfh.integrated"]
    scaled_photo = constant*SED_photo
    scaled_spectro = constant*SED_spectro
    scaled_lines = constant*SED_lines
    yerr = np.array(np.sqrt(bin_err))*1.96
    plt.errorbar(x=wave_to_plot,y=bin_flux,yerr=yerr, color = "red", alpha = 0.2)
    plt.plot(wave_to_plot,scaled_spectro)
    plt.xscale("log")
    return None
# params= pd.read_csv(CIGALE_parameters["file_store"]).keys()[0:-2]


# indices_params = np.where(["best" in name for name in galaxy_targ.array.dtype.names])
# tst = [galaxy_targ.array.dtype.names[idx] for idx in indices_params[0]]

def plot_jorge_3(obs,params_jorge):
    param_frame = params_jorge
    modules_params = [list(pcigale.sed_modules.get_module(module,blank = True).parameter_list.keys()) for module in CIGALE_parameters_normal['module_list']]
    parameter_list =[{param:param_frame[param] for param in module_params} for module_params in modules_params]
    
    warehouse = SedWarehouse(nocache = module_list)
    SED = warehouse.get_sed(module_list_normal, parameter_list)    
    targ_covar = SED_statistical_analysis.extract_target(obs,CIGALE_parameters_normal)
    target_photo, target_lines, = targ_covar[0:2]
    target_spectro, covar_photo = targ_covar[2:4]
    covar_spectro, covar_lines = targ_covar[4:6]
    #constants = SED_statistical_analysis.scale_factor_pre_computation(target_photo ,
                                                                     # covar_photo,
                                                                     # target_spectro,
                                                                     # covar_spectro,
                                                                     # CIGALE_parameters_normal["mode"])
    lim_wave, lim_flux = SED_statistical_analysis.limit_spec( galaxy_obs["spectroscopy_wavelength"],
                                galaxy_obs["spectroscopy_fluxes"],
                                CIGALE_parameters['wavelength_limits']["min"],
                                CIGALE_parameters['wavelength_limits']["max"])
    # wave_to_plot , bin_flux = SED_statistical_analysis.binning_flux(lim_wave, lim_flux, n_bins)

    # bin_wave,bin_err = SED_statistical_analysis.binning_variances(lim_wave, lim_err**2, n_bins)
    lim_wave_SED,lim_SED =SED_statistical_analysis.limit_spec( SED.wavelength_grid,
                                SED.fnu,
                                CIGALE_parameters['wavelength_limits']["min"],
                                CIGALE_parameters['wavelength_limits']["max"])

    # weight_spectro = 1
    # SED_photo = SED[0]
    SED_spectro = SED.fnu
    #SED_lines = SED[2]
    #scaled_photo = constant*SED_photo
    constant = galaxy_targ["best.sfh.integrated"]
    scaled_spectro = lim_SED*constant
    #yerr = np.array(lim_err)*1.96

    plt.plot(lim_wave_SED,scaled_spectro)
    plt.plot(lim_wave,lim_flux)
    #plt.errorbar(lim_wave,lim_flux,yerr = yerr,color = "red",alpha = 0.2)
    plt.xscale("log")
    #print(np.sum(((scaled_spectro-lim_flux)/lim_err)**2))

    return lim_flux, scaled_spectro , lim_wave, lim_wave_SED

results_read = pd.read_csv(CIGALE_parameters_normal["file_store"])
param_frame = results_read[results_read["MAP"]==1].iloc[0].to_dict()
params_jorge = param_frame.copy()
params_jorge["tau_main"] = galaxy_targ["best.sfh.tau_main"]
params_jorge["age_burst"] = galaxy_targ["best.sfh.age_burst"]
params_jorge["age_main"] = galaxy_targ["best.sfh.age_main"]
params_jorge["f_burst"] = galaxy_targ["best.sfh.f_burst"]
params_jorge["tau_burst"] = galaxy_targ["best.sfh.tau_burst"]
params_jorge["qpah"] = galaxy_targ["best.dust.qpah"]
params_jorge["alpha"] = galaxy_targ["best.dust.alpha"]
params_jorge["gamma"] = galaxy_targ["best.dust.gamma"]
params_jorge["umin"] = galaxy_targ["best.dust.umin"]
params_jorge["zgas"] = galaxy_targ["best.nebular.zgas"]
params_jorge["logU"] = galaxy_targ["best.nebular.logU"]
params_jorge["E_BV_lines"] = galaxy_targ["best.attenuation.E_BV_lines"]
params_jorge["ne"] =100
params_jorge["beta_calz94"] = True
params_jorge["D4000"] = False
params_jorge["IRX"] = True
params_jorge["EW_lines"] = "500.7/1.0 & 656.3/1.0"
params_jorge["luminosity_filters"] = "FUV & V_B90"
params_jorge["colours_filters"] = "FUV-NUV & NUV-r_prime"

plot_jorge_2(galaxy_obs,params_jorge)


res = plot_jorge_3(galaxy_obs,params_jorge)


plt.plot(res[2],res[0])
plt.plot(res[3],res[1])


diff_dist = []
for i in range(1,50):
    n_bins = 4*i
    true_binned = SED_statistical_analysis.binning_flux(res[2],res[0],n_bins,645,1800)
    other_binned = SED_statistical_analysis.binning_flux(res[3],res[1],n_bins,645,1800)
    #plt.plot(true_binned[0], true_binned[1])
    var = SED_statistical_analysis.binning_variances(res[2],np.sqrt(res[0])*0.1,n_bins,645,1800)
    plt.errorbar(true_binned[0], true_binned[1],yerr = 2*np.sqrt(var[1]))
    plt.plot(other_binned[0], other_binned[1])
    plt.yscale("log")
    plt.show()
plt.plot(diff)
plt.plot(diff_wave)

# def binning_flux_tst(wavelength, spectrum, n_bins,wave_min,wave_max):
#     """Bins the spectroscopy and the associated uncertainties.
#         We assume no correlation and constant band width
    
#     Input : 
#         wavelength - list, ordered wavelengths
#         spectrum - np.array, observed fluxes at each wavelength
#         n_bins - integer, number of bins
    
#     Output : 
#         wave_binned
#         spec_binned
#     """
#     bins = np.logspace(start = np.log10(wave_min),
#                        stop = np.log10(wave_max),
#                        num = n_bins)
#     idx = np.digitize(wavelength, bins)
#     spec_binned = [np.mean(spectrum[idx == i]) for i in range(1,n_bins+1)]
#     wave_binned = [np.mean(wavelength[idx == i]) for i in range(1,n_bins+1)]
#     return wave_binned,spec_binned

# diff = []
# diff_wave = []
# for i in range(1,50):
#     n_bins = 2*i
#     true_binned = binning_flux_tst(res[2],res[0],n_bins,645,1800)
#     other_binned = binning_flux_tst(res[3],res[1],n_bins,645,1800)
#     diff_wave.append(np.nanmean((np.array(true_binned[0])-np.array(other_binned[0])/np.array(true_binned[0]))))
#     diff.append(np.nanmean((np.array(true_binned[1])-np.array(other_binned[1]))/np.array(true_binned[1])))
# plt.plot(diff)
# plt.plot(diff_wave)



# # def var_trapz(var,wave):
# #     seq_diff = (wave[1:] -wave[:-1])**2 #consecutive differences for the step
# #     sum_middle = seq_diff[:-1] + seq_diff[1:] # consecutive sums of step
# #     res = seq_diff[0]*var[0] + seq_diff[-1]*var[-1] + np.sum(sum_middle*var[1:-1])
# #     return 0.25*res
# # def binning_variances_tst(wavelength, variances, n_bins,wave_min,wave_max):
# #     """Bins the spectroscopy and the associated uncertainties.
# #     We assume no correlation and constant bandwidth
    
# #     Input : 
# #     wavelength - list, ordered wavelengths
# #     spectrum - np.array, observed fluxes at each wavelength
# #     n_bins - integer, number of bins
    
# #     Output : 
# #     wave_binned
# #     spec_binned
# #     """
# #     bins = np.linspace(start = wave_min,
# #                        stop = wave_max,
# #                        num = n_bins)
# #     idx = np.digitize(wavelength, bins)
# #     variances_binned = [np.sum(variances[idx == i])/(np.sum(idx==i)**2) for i in range(1,n_bins+1)]
# #     wave_binned = [np.mean(wavelength[idx == i]) for i in range(1,n_bins+1)]
# #     return wave_binned,variances_binned


# diff_dist = []
# for i in range(100):
#     n_bins = 2*i
#     true_binned = binning_flux_tst(res[2],res[0],n_bins,645,1800)
#     other_binned = binning_flux_tst(res[3],res[1],n_bins,645,1800)
#     var = binning_variances_tst(res[2],np.sqrt(res[0])*0.1,n_bins,645,1800)
#     diff_dist.append(np.nanmean((np.array(true_binned[1])-np.array(other_binned[1])**2)/var))
# plt.plot(diff)
# plt.plot(diff_wave)


# plt.plot(diff_dist)
# plt.yscale("log")


    
diff_const = []
spec_true=[]
spec_simu = []
covars = []
lkhd = []
for i in range(1,101):
    n_bins = 2*i
    CIGALE_parameters_normal["n_bins"] = n_bins
    param_frame = params_jorge
    modules_params = [list(pcigale.sed_modules.get_module(module,blank = True).parameter_list.keys()) for module in CIGALE_parameters_normal['module_list']]
    parameter_list =[{param:param_frame[param] for param in module_params} for module_params in modules_params]
    warehouse = SedWarehouse(nocache = module_list)
    SED = SED_statistical_analysis.cigale(parameter_list, CIGALE_parameters_normal,warehouse)
    targ_covar = SED_statistical_analysis.extract_target(galaxy_obs,CIGALE_parameters_normal)
    target_photo, target_lines, = targ_covar[0:2]
    target_spectro, covar_photo = targ_covar[2:4]
    covar_spectro, covar_lines = targ_covar[4:6]
    constants = SED_statistical_analysis.scale_factor_pre_computation(target_photo ,
                                                                      covar_photo,
                                                                      target_spectro,
                                                                      covar_spectro,
                                                                      CIGALE_parameters_normal["mode"])
    _,covar = SED_statistical_analysis.compute_covar_spectro(galaxy_obs, CIGALE_parameters_normal) 
    weight_spectro = 1
    SED_photo = SED[0]
    SED_spectro = SED[1]
    SED_lines = SED[2]
    constant = SED_statistical_analysis.compute_constant(SED_photo, SED_spectro,constants,weight_spectro)
    diff_const.append((constant -galaxy_targ["best.sfh.integrated"])/galaxy_targ["best.sfh.integrated"])
    spec_true.append(target_spectro)
    spec_simu.append(galaxy_targ["best.sfh.integrated"]*SED_spectro)
    covars.append(np.mean(np.diag(covar)))
    lkhd.append(np.sum(((target_spectro- galaxy_targ["best.sfh.integrated"]*SED_spectro)**2) / np.diag(covar)))
    
plt.plot(diff_const)

plt.plot(spec_true[1])
plt.plot(spec_simu[1])

plt.plot(spec_true[5])
plt.plot(spec_simu[5])

plt.plot(spec_true[20])
plt.plot(spec_simu[20])

plt.plot(spec_true[80])
plt.plot(spec_simu[80])


# plt.plot((spec_true[10] -spec_simu[10]) / spec_true[10])

# plt.plot((spec_true[20] -spec_simu[20]) / spec_true[20])


# plt.plot((spec_true[80] -spec_simu[80]) / spec_true[80])

# plt.plot((spec_true[-16] -spec_simu[-16]) / spec_true[-16])

# plt.plot([np.mean(spec_true[i]) for i in range(100)])

# plt.plot(lkhd)
# plt.yscale("log")



