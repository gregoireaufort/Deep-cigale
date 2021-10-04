#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 15:59:45 2021

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



# SED_statistical_analysis.plot_result(CIGALE_parameters,
#                                      line_dict_fit = fit_jorge,
#                                      title = "Deep jorge 1")



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
                    "mode" : ["spectro"],
                    "n_jobs" : 10}
result_normal = SED_statistical_analysis.fit(galaxy_obs , CIGALE_parameters_normal, TAMIS_parameters)

SED_statistical_analysis.plot_result(CIGALE_parameters_normal,
                                     line_dict_fit = fit_jorge,
                                     title = "CIGALE Jorge 1 spectro")
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

SED_statistical_analysis.plot_result(CIGALE_parameters, line_dict_fit)


module_list_normal = ['sfhdelayed', 'bc03','nebular','dustatt_modified_starburst','dl2014', 'redshifting']
file_store_normal = 'store_parameters_test_normal_spectro.csv'
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
                    "deep_modules":deep_modules,
                    "module_parameters_to_fit":module_parameters_to_fit,
                    "module_parameters_discrete":module_parameters_discrete,
                    "n_bins":10,
                    "wavelength_limits" : wavelength_limits,
                    "nebular" :nebular_params,
                    "bands" :bands,
                    "mode" : ["spectro"],
                    "n_jobs" : 15}
result_normal = SED_statistical_analysis.fit(galaxy_obs , CIGALE_parameters_normal, TAMIS_parameters)


wave = galaxy_obs["spectroscopy_wavelength"]
err = galaxy_obs["spectroscopy_err"]**2
lim_wave, lim_err = SED_statistical_analysis.limit_spec( galaxy_obs["spectroscopy_wavelength"],
                                galaxy_obs["spectroscopy_err"],
                                CIGALE_parameters['wavelength_limits']["min"],
                                CIGALE_parameters['wavelength_limits']["max"])

_, lim_flux = SED_statistical_analysis.limit_spec( galaxy_obs["spectroscopy_wavelength"],
                                galaxy_obs["spectroscopy_fluxes"],
                                CIGALE_parameters['wavelength_limits']["min"],
                                CIGALE_parameters['wavelength_limits']["max"])


n_bins = 10

yerr =lim_err *1.96
plt.plot(lim_wave,lim_flux)
plt.fill_between(lim_wave,lim_flux - yerr ,lim_flux + yerr,alpha = 0.5)
plt.yscale("log")


bin_wave , bin_flux = SED_statistical_analysis.binning_flux(lim_wave, lim_flux, n_bins)
bin_wave,bin_err = SED_statistical_analysis.binning_variances(lim_wave, lim_err**2, n_bins)


yerr =lim_err *1.96
plt.plot(lim_wave,lim_flux)
plt.plot(galaxy_obs["spectroscopy_wavelength"],galaxy_obs["spectroscopy_fluxes"],)
plt.fill_between(lim_wave,lim_flux - yerr ,lim_flux + yerr,alpha = 0.5)
plt.errorbar(x=bin_wave,y=bin_flux,yerr=1.96*np.sqrt(bin_err), color = "red")
plt.yscale("log")
plt.xscale("log")


def plot_best_SED(CIGALE_parameters,obs):
    
    
    results_read = pd.read_csv(CIGALE_parameters["file_store"])
    param_frame = results_read[results_read["MAP"]==1].iloc[0].to_dict()
    print(param_frame)
    modules_params = [list(pcigale.sed_modules.get_module(module,blank = True).parameter_list.keys()) for module in CIGALE_parameters['module_list']]
    parameter_list =[{param:param_frame[param] for param in module_params} for module_params in modules_params]
    db_filters = Database()
    filters = [db_filters.get_filter(name = band) for band in obs["bands"]]
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
    wave_to_plot , bin_flux = SED_statistical_analysis.binning_flux(lim_wave, lim_flux, n_bins)

    bin_wave,bin_err = SED_statistical_analysis.binning_variances(lim_wave, lim_err**2, n_bins)

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





def plot_jorge_2(obs,params_jorge):
    param_frame = params_jorge
    modules_params = [list(pcigale.sed_modules.get_module(module,blank = True).parameter_list.keys()) for module in CIGALE_parameters_normal['module_list']]
    parameter_list =[{param:param_frame[param] for param in module_params} for module_params in modules_params]
    db_filters = Database()
    filters = [db_filters.get_filter(name = band) for band in obs["bands"]]
    wave_to_plot = [filtr.pivot_wavelength for filtr in filters]
    warehouse = SedWarehouse(nocache = module_list)
    SED = SED_statistical_analysis.cigale(parameter_list, CIGALE_parameters_normal,warehouse)
    
    targ_covar = SED_statistical_analysis.extract_target(obs,CIGALE_parameters_normal)
    target_photo, target_lines, = targ_covar[0:2]
    target_spectro, covar_photo = targ_covar[2:4]
    covar_spectro, covar_lines = targ_covar[4:6]
    constants = SED_statistical_analysis.scale_factor_pre_computation(target_photo ,
                                                                     covar_photo,
                                                                     target_spectro,
                                                                     covar_spectro,
                                                                     CIGALE_parameters_normal["mode"])
    _, lim_flux = SED_statistical_analysis.limit_spec( galaxy_obs["spectroscopy_wavelength"],
                                galaxy_obs["spectroscopy_fluxes"],
                                CIGALE_parameters['wavelength_limits']["min"],
                                CIGALE_parameters['wavelength_limits']["max"])
    wave_to_plot , bin_flux = SED_statistical_analysis.binning_flux(lim_wave, lim_flux, n_bins)

    bin_wave,bin_err = SED_statistical_analysis.binning_variances(lim_wave, lim_err**2, n_bins)

    weight_spectro = 1
    SED_photo = SED[0]
    SED_spectro = SED[1]
    SED_lines = SED[2]
    constant = SED_statistical_analysis.compute_constant(SED_photo, SED_spectro,constants,weight_spectro)
    scaled_photo = constant*SED_photo
    scaled_spectro = constant*SED_spectro
    scaled_lines = constant*SED_lines
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
    print(np.sumscaled_photo-galaxy_obs['photometry_fluxes'])/galaxy_obs['photometry_err'])**2))
    return None
# params= pd.read_csv(CIGALE_parameters["file_store"]).keys()[0:-2]


# indices_params = np.where(["best.sfh" in name for name in galaxy_targ.array.dtype.names])
# param_jorge = [galaxy_targ.array.dtype.names[idx] for idx in indices_params[0]][0:6]



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

plot_jorge_2(galaxy_obs,params_jorge)
