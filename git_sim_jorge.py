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
from test_plots_infos import plot_best_SED,plot_posterior_predictive
import astropy 

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
                             'metallicity' : [0.0004,0.008,0.05,0.02,0.004],
                             'qpah' : [0.47,1.12,1.77,2.5],
                             'umin' : [5.0,10.0,25.0],
                             'alpha' : [2],
                             'gamma' : [0.02],
                             'separation_age': [10],
                             'logU' :[ -4.0,-3.5,-3.0, -2.5, -2.0,-1.5,-1.0],
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
                    "bands" :list(photo_flux.columns), #DUMMY
                    "mode" : ["spectro"],
                    "n_jobs" : 10}


A=astropy.io.fits.open("/home/aufort/Desktop/jorge/results.fits")
B = A[1].data
galaxy_targ = B[0]

def fit(ident,CIGALE_parameters,mode,failed):
    galaxy_obs =SED_statistical_analysis.galaxy_Jorge(photo_flux,
                                      spec_flux,
                                      spec_wavelength, 
                                      z,
                                      ident, 
                                      SNR_photo=5,
                                      SNR_spectro = 2)
        
    galaxy_targ = B[ident]
    if mode == ["photo"]:
        title = "photo"
    elif mode == ["spectro"]:
        title = "spectro"
    else: 
        title = "photo+spectro"
    fit_jorge = {"tau_main" : galaxy_targ["best.sfh.tau_main"],
                    'age_main':galaxy_targ["best.sfh.age_main"],
                      'tau_burst':galaxy_targ["best.sfh.tau_burst"],
                      'f_burst':galaxy_targ["best.sfh.f_burst"],
                      'age_burst':galaxy_targ["best.sfh.age_burst"],
                      'E_BV_lines':galaxy_targ["best.attenuation.E_BV_lines"]}
    fit_disc = {"metallicity" : galaxy_targ["best.stellar.metallicity"],
                    "qpah":galaxy_targ["best.dust.qpah"],
                    "logU" : galaxy_targ["best.nebular.logU"],
                    "zgas" : galaxy_targ["best.nebular.zgas"],
                    "umin" : galaxy_targ["best.dust.umin"]}
    file_store = 'test_Jorge/complete2/'+str(ident)+"_"+title+ ".csv"
    wavelength_lines =[10]#[121.60000000000001,133.5,139.7, 154.9, 164.0, 166.5, 190.9,232.6, 279.8, 372.7, 379.8, 383.5, 386.9, 388.9, 397.0, 407.0, 410.2, 434.0, 486.1, 495.9, 500.7, 630.0, 654.8,656.3, 658.4, 671.6, 673.1]
    nebular_params = {"lines_width" : module_parameters_discrete["lines_width"][0],"line_waves" : wavelength_lines}
    
    CIGALE_parameters["module_parameters_discrete"]["redshift"] = [galaxy_obs["redshift"]]
    CIGALE_parameters["wavelength_limits"] = {"min" : 600,
                                               "max" : 1800}
    CIGALE_parameters["bands"] =galaxy_obs["bands"]
    CIGALE_parameters["nebular"] = nebular_params 
    CIGALE_parameters["file_store"] = file_store
    CIGALE_parameters["mode"] = mode
    TAMIS_parameters = initialize_TAMIS(CIGALE_parameters, n_comp=4, alpha=100,
                                        T_max=40, recycle =True)
    result = SED_statistical_analysis.fit(galaxy_obs,
                                                 CIGALE_parameters,
                                                 TAMIS_parameters)
    
    try:
        SED_statistical_analysis.plot_result(CIGALE_parameters,
                                                  title = str(ident)+"_"+title,
                                                  line_dict_fit_cont = fit_jorge ,
                                                  line_dict_disc = fit_disc,
                                                  savefile = 'test_Jorge/complete2/'+str(ident)+"_"+title+".pdf")

    except:
        failed.append(CIGALE_parameters)
    return result, CIGALE_parameters
 
failed_plots = []
results = []
np.random.seed(42)

for ident in range(21,50):
    for mode in [["photo"],["spectro"],["photo","spectro"]]:
        ft = fit(ident, CIGALE_parameters,mode, failed_plots)
        # if mode ==["spectro","photo"]:
        #     results.append(ft[0])

#fit(17,CIGALE_parameters,["spectro","photo"],failed_plots)
#ESS = []    
#ESS_relatif = []
#N=[]
#for ident in range(10):
#    for mode in [["photo"],["spectro"],["spectro","photo"]]:
#        file_store = 'test_Jorge/complete/'+str(ident)+"_"+str(mode[0])+ ".csv"
#        n_sim = pd.read_csv(file_store).shape[0]
#        ESS_run = compute_ESS_file(file_store)
#        ESS.append(compute_ESS_file(file_store))
#        ESS_relatif.append(ESS_run/n_sim)
#        N.append(n_sim)
#ESS[::3]
#ESS[1::3]
#ESS[2::3]

#failed_plots


# plot_best_SED(CIGALE_parameters,galaxy_obs)
# for ident in range(10):
#     for mode in [["photo"],["spectro"],["spectro","photo"]]:
#         galaxy_obs =SED_statistical_analysis.galaxy_Jorge(photo_flux,
#                                       spec_flux,
#                                       spec_wavelength, 
#                                       z,
#                                       ident, 
#                                       SNR_photo=5,
#                                       SNR_spectro = 2)
#         galaxy_targ = B[ident]
        
#         fit_jorge = {"tau_main" : galaxy_targ["best.sfh.tau_main"],
#                         'age_main':galaxy_targ["best.sfh.age_main"],
#                           'tau_burst':galaxy_targ["best.sfh.tau_burst"],
#                           'f_burst':galaxy_targ["best.sfh.f_burst"],
#                           'age_burst':galaxy_targ["best.sfh.age_burst"],
#                           'E_BV_lines':galaxy_targ["best.attenuation.E_BV_lines"]}
#         fit_disc = {"metallicity" : galaxy_targ["best.stellar.metallicity"],
#                     "qpah":galaxy_targ["best.dust.qpah"],
#                     "logU" : galaxy_targ["best.nebular.logU"],
#                     "zgas" : galaxy_targ["best.nebular.zgas"],
#                     "umin" : galaxy_targ["best.dust.umin"]}
#         file_store = 'test_Jorge/complete/'+str(ident)+"_"+str(mode[0])+ ".csv"
#         wavelength_lines =[10]#[121.60000000000001,133.5,139.7, 154.9, 164.0, 166.5, 190.9,232.6, 279.8, 372.7, 379.8, 383.5, 386.9, 388.9, 397.0, 407.0, 410.2, 434.0, 486.1, 495.9, 500.7, 630.0, 654.8,656.3, 658.4, 671.6, 673.1]
#         nebular_params = {"lines_width" : module_parameters_discrete["lines_width"][0],"line_waves" : wavelength_lines}
        
#         CIGALE_parameters["module_parameters_discrete"]["redshift"] = [galaxy_obs["redshift"]]
#         CIGALE_parameters["wavelength_limits"] = {"min" : 600,
#                                                     "max" : 1800}
#         CIGALE_parameters["bands"] =galaxy_obs["bands"]
#         CIGALE_parameters["nebular"] = nebular_params 
#         CIGALE_parameters["file_store"] = file_store
#         CIGALE_parameters["mode"] = mode
#         plot_posterior_predictive(CIGALE_parameters,galaxy_obs,500,
#                           title = 'test_Jorge/complete/'+str(ident)+"_"+str(mode[0])+"_post_pred.pdf")

#============================REPLOTING discrete =============================#
# mode = ['photo']
# for ident in range(10):
#     for mode in [["photo"],["spectro"],["spectro","photo"]]:
#         galaxy_targ = B[ident]
        
#         fit_jorge = {"tau_main" : galaxy_targ["best.sfh.tau_main"],
#                         'age_main':galaxy_targ["best.sfh.age_main"],
#                           'tau_burst':galaxy_targ["best.sfh.tau_burst"],
#                           'f_burst':galaxy_targ["best.sfh.f_burst"],
#                           'age_burst':galaxy_targ["best.sfh.age_burst"],
#                           'E_BV_lines':galaxy_targ["best.attenuation.E_BV_lines"]}
#         fit_disc = {"metallicity" : galaxy_targ["best.stellar.metallicity"],
#                     "qpah":galaxy_targ["best.dust.qpah"],
#                     "logU" : galaxy_targ["best.nebular.logU"],
#                     "zgas" : galaxy_targ["best.nebular.zgas"],
#                     "umin" : galaxy_targ["best.dust.umin"]}
#         file_store = 'test_Jorge/complete/'+str(ident)+"_"+str(mode)+ ".csv"
#         wavelength_lines =[10]#[121.60000000000001,133.5,139.7, 154.9, 164.0, 166.5, 190.9,232.6, 279.8, 372.7, 379.8, 383.5, 386.9, 388.9, 397.0, 407.0, 410.2, 434.0, 486.1, 495.9, 500.7, 630.0, 654.8,656.3, 658.4, 671.6, 673.1]
#         nebular_params = {"lines_width" : module_parameters_discrete["lines_width"][0],"line_waves" : wavelength_lines}
        
#         CIGALE_parameters["module_parameters_discrete"]["redshift"] = [galaxy_obs["redshift"]]
#         CIGALE_parameters["wavelength_limits"] = {"min" : 600,
#                                                    "max" : 1800}
#         CIGALE_parameters["bands"] =galaxy_obs["bands"]
#         CIGALE_parameters["nebular"] = nebular_params 
#         CIGALE_parameters["file_store"] = file_store
#         CIGALE_parameters["mode"] = mode
#         SED_statistical_analysis.plot_result(CIGALE_parameters,
#                                                  title = str(ident)+"_"+str(mode),
#                                                  line_dict_fit_cont = fit_jorge ,
#                                                  line_dict_disc = fit_disc,
#                                                  savefile = 'test_Jorge/complete/'+str(ident)+"_"+str(mode)+".pdf")
