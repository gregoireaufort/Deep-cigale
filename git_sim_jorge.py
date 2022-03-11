#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:31:00 2022

@author: aufort
"""

import SED_statistical_analysis
import glob
import pandas as pd
from utils import *
import numpy as np

np.random.seed(42)
folder_path ="/home/aufort/Bureau/Deep-cigale/test_Jorge/"

photo_flux = pd.read_csv("test_Jorge/photo.csv",index_col=[0])

z =  pd.read_csv("test_Jorge/redshift.csv",index_col=[0])
spec_flux =  pd.read_csv("test_Jorge/spectro.csv",index_col=[0])
spec_wavelength =  pd.read_csv("test_Jorge/wavelength_spectro.csv",index_col=[0])

test =SED_statistical_analysis.galaxy_Jorge(photo_flux,
                                      spec_flux,
                                      spec_wavelength, 
                                      z,
                                      ident = 1,
                                      SNR_photo=5,
                                      SNR_spectro = 5)

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
                             'redshift':None,
                             'filters':["B_B90 & V_B90 & FUV"],
}
wavelength_limits = None


wavelength_lines =None#[121.60000000000001,133.5,139.7, 154.9, 164.0, 166.5, 190.9,232.6, 279.8, 372.7, 379.8, 383.5, 386.9, 388.9, 397.0, 407.0, 410.2, 434.0, 486.1, 495.9, 500.7, 630.0, 654.8,656.3, 658.4, 671.6, 673.1]
nebular_params = None

module_list = ['sfhdelayed', 'bc03','nebular','dustatt_modified_starburst','dl2014', 'redshifting']
file_store = None

CIGALE_parameters = {"module_list":module_list,
                    "path_deep" : None,    
                    "file_store":file_store,
                    "deep_modules":None,
                    "module_parameters_to_fit":module_parameters_to_fit,
                    "module_parameters_discrete":module_parameters_discrete,
                    "infos_to_save":['sfh.sfr','stellar.m_star'],
                    "n_bins":20,
                    "wavelength_limits" : wavelength_limits,
                    "nebular" :nebular_params,
                    "bands" :None, #DUMMY
                    "mode" : ["photo"],
                    "n_jobs" : 10}


A=astropy.io.fits.open("/home/aufort/Desktop/jorge/results.fits")
B = A[1].data
galaxy_targ = B[0]



def fit(ident,CIGALE_parameters,failed):
    galaxy_obs =SED_statistical_analysis.galaxy_Jorge(photo_flux,
                                      spec_flux,
                                      spec_wavelength, 
                                      z,
                                      ident,
                                      SNR_photo=5,
                                      SNR_spectro = 5)
        
    A=astropy.io.fits.open("/home/aufort/Desktop/jorge/results.fits")
    B = A[1].data
    galaxy_targ = B[ident]

    
    fit_jorge = {"tau_main" : galaxy_targ["best.sfh.tau_main"],
                    'age_main':galaxy_targ["best.sfh.age_main"],
                      'tau_burst':galaxy_targ["best.sfh.tau_burst"],
                      'f_burst':galaxy_targ["best.sfh.f_burst"],
                      'age_burst':galaxy_targ["best.sfh.age_burst"],
                      'E_BV_lines':galaxy_targ["best.attenuation.E_BV_lines"]}
    file_store = 'test_Jorge/'+str(ident)+"_photo"+ ".csv"
    wavelength_lines =[10]#[121.60000000000001,133.5,139.7, 154.9, 164.0, 166.5, 190.9,232.6, 279.8, 372.7, 379.8, 383.5, 386.9, 388.9, 397.0, 407.0, 410.2, 434.0, 486.1, 495.9, 500.7, 630.0, 654.8,656.3, 658.4, 671.6, 673.1]
    nebular_params = {"lines_width" : module_parameters_discrete["lines_width"][0],"line_waves" : wavelength_lines}
    
    CIGALE_parameters["module_parameters_discrete"]["redshift"] = [galaxy_obs["redshift"]]
    CIGALE_parameters["wavelength_limits"] = {"min" : 600,
                                               "max" : 1800}
    CIGALE_parameters["bands"] =galaxy_obs["bands"]
    CIGALE_parameters["nebular"] = nebular_params 
    CIGALE_parameters["file_store"] = file_store
    TAMIS_parameters = initialize_TAMIS(CIGALE_parameters)
    result = SED_statistical_analysis.fit(galaxy_obs,
                                                 CIGALE_parameters,
                                                 TAMIS_parameters)
    
    try:
        SED_statistical_analysis.plot_result(CIGALE_parameters,
                                             title = str(ident)+"_photo",
                                             line_dict_fit = fit_jorge ,
                                             savefile = 'test_Jorge/'+str(ident)+"_photo.pdf")
    except:
        failed.append(CIGALE_parameters)
        
        
        
fit(3, CIGALE_parameters, [])

















# def fit_all_folder(folder_path):
#     list_files = glob.glob(folder_path+"*") 
#     failed_plots = []
#     for file in list_files:
#         fit(file, CIGALE_parameters,failed_plots)
#     return failed_plots
# #
# to_plot_after = fit_all_folder("test_moons/ID_302327_ETC_output/")