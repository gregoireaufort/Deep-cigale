#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 13:02:38 2022

@author: aufort
"""

import SED_statistical_analysis
import glob
import pandas as pd
from utils import *
import numpy as np
from test_plots_infos import plot_best_SED,plot_posterior_predictive
import astropy 



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
                    "bands" :None,#list(photo_flux.columns), #DUMMY
                    "mode" : ["spectro"],
                    "n_jobs" : 10}


A=astropy.io.fits.open("/home/aufort/Desktop/jorge/results.fits")
B = A[1].data
galaxy_targ = B[0]

error ={}
error["photo"] = pd.DataFrame()
error["spectro"] = pd.DataFrame()
error["photo+spectro"] = pd.DataFrame()
for ident in range(10):
    for mode in [["photo"],["spectro"],["spectro","photo"]]:
        galaxy_obs =None
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
                          'E_BV_lines':galaxy_targ["best.attenuation.E_BV_lines"],
                          "metallicity" : galaxy_targ["best.stellar.metallicity"],
                        "qpah":galaxy_targ["best.dust.qpah"],
                        "logU" : galaxy_targ["best.nebular.logU"],
                        "zgas" : galaxy_targ["best.nebular.zgas"],
                        "umin" : galaxy_targ["best.dust.umin"]}
        file_store = '/media/aufort/Nouveau nom/complete/'+str(ident)+"_"+str(mode)+ ".csv"
        #wavelength_lines =[10]#[121.60000000000001,133.5,139.7, 154.9, 164.0, 166.5, 190.9,232.6, 279.8, 372.7, 379.8, 383.5, 386.9, 388.9, 397.0, 407.0, 410.2, 434.0, 486.1, 495.9, 500.7, 630.0, 654.8,656.3, 658.4, 671.6, 673.1]
        #nebular_params = {"lines_width" : module_parameters_discrete["lines_width"][0],"line_waves" : wavelength_lines}
        
        #CIGALE_parameters["module_parameters_discrete"]["redshift"] = [galaxy_obs["redshift"]]
        #CIGALE_parameters["wavelength_limits"] = {"min" : 600,
        #                                            "max" : 1800}
        #CIGALE_parameters["bands"] =galaxy_obs["bands"]
        #CIGALE_parameters["nebular"] = nebular_params 
        CIGALE_parameters["file_store"] = file_store
        CIGALE_parameters["mode"] = mode
        res = SED_statistical_analysis.analyse_results(CIGALE_parameters)
        temp = {}
        for parameter in res.keys():
                   temp[parameter]= [np.abs((res[parameter]["max"] - fit_jorge[parameter]) /fit_jorge[parameter])]
        error[title] = pd.concat([error[title],pd.DataFrame.from_dict(temp)])
        
        
# [df.reset_index(drop = True,inplace= True) for df in error.values()]
# error["photo+spectro"]=error["photo+spectro"].append(error["photo"].loc[1])
error["photo+spectro"]["mode"] = "photo+spectro"
error["photo"]["mode"] = "photo"
error["spectro"]["mode"] = "spectro"

cdf = pd.concat([error["photo"],error["spectro"],error["photo+spectro"] ])
mdf = pd.melt(cdf, id_vars = ["mode"], value_vars = [*res.keys()])



import seaborn as sns
sns.boxplot(x = "variable", y ="value", hue = "mode",data = mdf)
plt.yscale("log")
plt.xticks(rotation = 45)