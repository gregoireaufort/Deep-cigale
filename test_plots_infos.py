#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:29:15 2022

@author: aufort
"""

import SED_statistical_analysis
from astropy.io import fits

import scipy.stats as stats
from utils import *


import numpy as np

np.random.seed(42)


spec_filename ="test_moons/ID_302327_ETC_output/ID_302327_BC03_z0.69_v60.00_m24.1_nexp8_HR_RI.fits"

hdul = fits.open(spec_filename)
galaxy_obs = SED_statistical_analysis.read_galaxy_moons(spec_filename, 
                 None,
                 ident =None)

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
                             'qpah' : [0.47,1.12,1.77,2.5],
                             'umin' : [5.0,10.0,25.0],
                             'alpha' : [2],
                             'gamma' : [0.02],
                             'separation_age': [10],
                             'logU' :[-3.5,-2.5,-1.5],
                             'zgas':[0.004,0.008,0.011,0.022,0.007, 0.014],
                             'ne':[100],
                             'f_esc': [0.0],
                             'f_dust' : [0.0],
                             'lines_width' :[300.0],
                             'emission' :[True],
                             'redshift':[galaxy_obs["redshift"]],
                             'filters':["B_B90 & V_B90 & FUV"],
}
wavelength_limits = {"min" :  galaxy_obs["spectroscopy_wavelength"][0],"max" : galaxy_obs["spectroscopy_wavelength"][-1]}


wavelength_lines =[121.60000000000001,133.5,139.7, 154.9, 164.0, 166.5, 190.9,232.6, 279.8, 372.7, 379.8, 383.5, 386.9, 388.9, 397.0, 407.0, 410.2, 434.0, 486.1, 495.9, 500.7, 630.0, 654.8,656.3, 658.4, 671.6, 673.1]
nebular_params = {"lines_width" : module_parameters_discrete["lines_width"][0],"line_waves" : wavelength_lines}

module_list = ['sfhdelayed', 'bc03','nebular','dustatt_modified_starburst','dl2014', 'redshifting']
file_store = 'store_parameters_302327_BC03_z0.69_v60.00_m24.1_nexp8_HR_RI_normal.csv'

CIGALE_parameters= {"module_list":module_list,
                    "path_deep" : None,    
                    "file_store":file_store,
                    "infos_to_save":['sfh.sfr','stellar.m_star'],
                    "deep_modules":None,
                    "module_parameters_to_fit":module_parameters_to_fit,
                    "module_parameters_discrete":module_parameters_discrete,
                    "n_bins":20,
                    "wavelength_limits" : wavelength_limits,
                    "nebular" :nebular_params,
                    "bands" :['IRAC1'], #DUMMY
                    "mode" : ["spectro"],
                    "n_jobs" : 10}


TAMIS_parameters = initialize_TAMIS(CIGALE_parameters)
np.random.seed(42)


result = SED_statistical_analysis.fit(galaxy_obs , CIGALE_parameters, TAMIS_parameters)



def compare_SED(parameter_list,obs,CIGALE_params):
    
def plot_best_SED(CIGALE_parameters,obs):
    
    
    results_read = pd.read_csv(CIGALE_parameters["file_store"])
    n_bins = CIGALE_parameters["n_bins"]
    param_frame = results_read[results_read["MAP"]==1].iloc[0].to_dict()
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
    constant = SED_statistical_analysis.compute_constant(SED_photo, SED_spectro,constants,weight_spectro)
    scaled_spectro = constant*SED_spectro
    
    yerr = np.array(np.sqrt(bin_err))*1.96
    plt.errorbar(x=wave_to_plot,y=bin_flux,yerr=yerr, color = "red")
    plt.plot(wave_to_plot,scaled_spectro)
    plt.xscale("log")
    return None
plot_best_SED(CIGALE_parameters,galaxy_obs)
