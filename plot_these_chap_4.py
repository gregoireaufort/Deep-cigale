#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 17:27:25 2022

@author: aufort
"""

import SED_statistical_analysis
from utils import *
import pandas as pd
import pcigale
from pcigale.warehouse import SedWarehouse
import matplotlib.pyplot as plt
import astropy 



def add_sim_plot(params_list,
                 galaxy_obs,
                 CIGALE_parameters,
                 warehouse,
                 color = "blue", 
                 alpha = 1):
    SED = SED_statistical_analysis.cigale(params_list, CIGALE_parameters,warehouse)
    
    SED_photo = SED[0]
    SED_spectro = SED[1]
    weight_spectro = 1
    SED_cig = warehouse.get_sed(CIGALE_parameters['module_list'],params_list)
    new_SED=  SED_cig.fnu.copy()
    new_SED[np.isinf(galaxy_obs["spectroscopy_fluxes"]/new_SED)] = np.mean(galaxy_obs["spectroscopy_fluxes"])
    scaled_spectro = np.nanmean(galaxy_obs["spectroscopy_fluxes"]/new_SED)*new_SED
    plt.loglog(SED_cig.wavelength_grid,scaled_spectro, color,alpha = 1)


def plot_posterior_predictive(CIGALE_parameters,galaxy_obs,n,title = None):
    wave_to_plot = []
    with pcigale.data.SimpleDatabase("filters") as db:
        wave = [db.get(name=fltr).pivot for fltr in CIGALE_parameters["bands"]]
        wave_to_plot = wave_to_plot + list(wave)
                
    modules_params = [list(pcigale.sed_modules.get_module(module,blank = True).parameter_list.keys()) for module in CIGALE_parameters['module_list']]
    
    warehouse = SedWarehouse()
    results_read = pd.read_csv(CIGALE_parameters["file_store"])
    idxs =np.random.choice(range(len(results_read["weights"])), p = results_read["weights"], size = n)
    norm_weights = results_read["weights"].iloc[idxs]/np.sum(results_read["weights"].iloc[idxs])
    sims_to_plot = []
    for idx in idxs:
        param_frame = results_read.iloc[idx].to_dict()
        parameter_list =[{param:param_frame[param] for param in module_params} for module_params in modules_params]
        add_sim_plot(parameter_list,
                 galaxy_obs,
                 CIGALE_parameters,
                 warehouse,
                 color = "red", 
                 alpha =  results_read["weights"].iloc[idx]/np.sum(results_read["weights"].iloc[idxs]))
    if title :
        plt.suptitle(title)
        plt.savefig(title)
    plt.loglog(galaxy_obs["spectroscopy_wavelength"],galaxy_obs["spectroscopy_fluxes"], alpha = 0.5)
    plt.loglog(wave_to_plot,galaxy_obs["photometry_fluxes"], color = "black", marker = "+",linestyle = 'None',markersize = 10)
    # plt.loglog(wave_spectro,target_spectro, color = "green", marker = "o",linestyle = 'None', alpha = 0.5, markersize = 0.5)

    plt.ylim((1e-10,1e6))
    plt.show()
    
np.random.seed(42)
folder_path ="/home/aufort/Bureau/Deep-cigale/test_Jorge/"

photo_flux = pd.read_csv("test_Jorge/photo.csv",index_col=[0])
z =  pd.read_csv("test_Jorge/redshift.csv",index_col=[0])
spec_flux =  pd.read_csv("test_Jorge/spectro.csv",index_col=[0])
spec_wavelength =  pd.read_csv("test_Jorge/wavelength_spectro.csv",index_col=[0])

ident = 0
galaxy_obs =SED_statistical_analysis.galaxy_Jorge(photo_flux,
                                      spec_flux,
                                      spec_wavelength, 
                                      z,
                                      ident, 
                                      SNR_photo=np.inf,
                                      SNR_spectro =np.inf)

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
wavelength_limits =  {"min" : 600,"max" : 1800}


#wavelength_lines =[121.60000000000001,133.5,139.7, 154.9, 164.0, 166.5, 190.9,232.6, 279.8, 372.7, 379.8, 383.5, 386.9, 388.9, 397.0, 407.0, 410.2, 434.0, 486.1, 495.9, 500.7, 630.0, 654.8,656.3, 658.4, 671.6, 673.1]
wavelength_lines=[486.1,121.6]
nebular_params = {"lines_width" : module_parameters_discrete["lines_width"][0],"line_waves" : wavelength_lines}

module_list = ['sfhdelayed', 'bc03','nebular','dustatt_modified_starburst','dl2014', 'redshifting']
title = "photo+spectro"
file_store = 'test_Jorge/complete3/'+str(ident)+"_"+title+ ".csv"

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
                    "bands" :galaxy_obs["bands"], #DUMMY
                    "mode" : ["photo","spectro"],
                    "n_jobs" : 10}

plt.loglog(galaxy_obs["spectroscopy_wavelength"],galaxy_obs["spectroscopy_fluxes"])


wave_to_plot = []
with pcigale.data.SimpleDatabase("filters") as db:
    wave = [db.get(name=fltr).pivot for fltr in CIGALE_parameters["bands"]]
    wave_to_plot = wave_to_plot + list(wave)
            

galaxy_obs =SED_statistical_analysis.galaxy_Jorge(photo_flux,
                                      spec_flux,
                                      spec_wavelength, 
                                      z,
                                      ident, 
                                      SNR_photo=5,
                                      SNR_spectro =2)
targ2 = SED_statistical_analysis.lim_target_spectro(galaxy_obs,
                                                 CIGALE_parameters)

plt.loglog(targ2[0],targ2[1]+1e-4)
plt.loglog(wave_to_plot,galaxy_obs['photometry_fluxes'], marker = "+",
           linestyle = 'None',markersize = 10) 
plt.xlabel("Wavelength (nm)")
plt.ylabel("flux")
plt.title("Observed data")

def plot_obs(CIGALE_parameters,galaxy_obs):
    f,ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    wave_to_plot = []
    err_to_plot=  []
    flux_to_plot = []
    if 'spectro' in CIGALE_parameters["mode"]:
        
        # wave,y_spec,err = limit_spec(galaxy_obs["spectroscopy_wavelength"],
        #                           galaxy_obs["spectroscopy_fluxes"],
        #                           CIGALE_parameters["wavelength_limits"]["min"],
        #                           CIGALE_parameters["wavelength_limits"]["max"],
        #                           galaxy_obs["spectroscopy_err"])
        # yerr_spec = np.array(err)*1.96
        wave_spectro,target_spectro = lim_target_spectro(galaxy_obs,
                                                 CIGALE_parameters)
        target_spectro,wave_spectro,_ = extract_lines(CIGALE_parameters,
                                                    wave_spectro,
                                                    target_spectro,
                                                    False)
        wave_spectro,target_spectro = binning_flux(wave_spectro,
                             target_spectro,
                             CIGALE_parameters['n_bins'],
                             CIGALE_parameters['wavelength_limits']["min"],
                             CIGALE_parameters['wavelength_limits']["max"])
        
        covar_spectro,covar_lines = compute_covar_spectro(galaxy_obs,
                                                          CIGALE_parameters)
        wave_to_plot = wave_to_plot + list(wave_spectro)
        err_to_plot = err_to_plot+list(np.diag(covar_spectro))
        flux_to_plot = flux_to_plot+list(target_spectro)
        ax.plot(wave_to_plot,flux_to_plot, color = "green",marker = "+",
           linestyle = 'None',markersize = 10)
    if "photo" in CIGALE_parameters["mode"]:
        with pcigale.data.SimpleDatabase("filters") as db:
            wave = [db.get(name=fltr).pivot for fltr in CIGALE_parameters["bands"]]
            wave_to_plot = wave_to_plot + list(wave)
            
        flux_to_plot = flux_to_plot +list(galaxy_obs["photometry_fluxes"])
        yerr_photo = np.array(galaxy_obs["photometry_err"])*1.96
        err_to_plot = err_to_plot+list(yerr_photo)
        ax.errorbar(x=list(wave),y=list(galaxy_obs["photometry_fluxes"]),yerr=list(yerr_photo),
                color = "blue", fmt = " ",alpha = 0.5)
        ax.plot(wave,galaxy_obs['photometry_fluxes'], marker = "+",
           linestyle = 'None',markersize = 10,color = "blue")
    wave,flux, err = zip(*sorted(zip(wave_to_plot, flux_to_plot,err_to_plot)))
    
    #
    return ax,wave_to_plot
plot_obs(CIGALE_parameters,galaxy_obs)



def plot_obs2(CIGALE_parameters,galaxy_obs):
    f,ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    wave_to_plot = []
    err_to_plot=  []
    flux_to_plot = []
    if 'spectro' in CIGALE_parameters["mode"]:
        
        # wave,y_spec,err = limit_spec(galaxy_obs["spectroscopy_wavelength"],
        #                           galaxy_obs["spectroscopy_fluxes"],
        #                           CIGALE_parameters["wavelength_limits"]["min"],
        #                           CIGALE_parameters["wavelength_limits"]["max"],
        #                           galaxy_obs["spectroscopy_err"])
        # yerr_spec = np.array(err)*1.96
        wave_spectro,target_spectro = lim_target_spectro(galaxy_obs,
                                                 CIGALE_parameters)
        target_spectro,wave_spectro,_ = extract_lines(CIGALE_parameters,
                                                    wave_spectro,
                                                    target_spectro,
                                                    False)
        wave_spectro,target_spectro = binning_flux(wave_spectro,
                             target_spectro,
                             CIGALE_parameters['n_bins'],
                             CIGALE_parameters['wavelength_limits']["min"],
                             CIGALE_parameters['wavelength_limits']["max"])
        
        covar_spectro,covar_lines = compute_covar_spectro(galaxy_obs,
                                                          CIGALE_parameters)
        wave_to_plot = wave_to_plot + list(wave_spectro)
        err_to_plot = err_to_plot+list(np.diag(covar_spectro))
        flux_to_plot = flux_to_plot+list(target_spectro)
        ax.plot(wave_to_plot,flux_to_plot, color = "green")
    if "photo" in CIGALE_parameters["mode"]:
        with pcigale.data.SimpleDatabase("filters") as db:
            wave = [db.get(name=fltr).pivot for fltr in CIGALE_parameters["bands"]]
            wave_to_plot = wave_to_plot + list(wave)
            
        flux_to_plot = flux_to_plot +list(galaxy_obs["photometry_fluxes"])
        yerr_photo = np.array(galaxy_obs["photometry_err"])*1.96
        err_to_plot = err_to_plot+list(yerr_photo)
        ax.errorbar(x=list(wave),y=list(galaxy_obs["photometry_fluxes"]),yerr=list(yerr_photo),
                color = "red", fmt = " ")
    wave,flux, err = zip(*sorted(zip(wave_to_plot, flux_to_plot,err_to_plot)))
    
    #
    return ax,wave_to_plot

plot_obs2(CIGALE_parameters,galaxy_obs)
