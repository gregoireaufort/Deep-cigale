#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 12:26:01 2021

@author: aufort
"""

from  TAMIS import TAMIS
import seaborn as sns
from astropy.table import Table
from astropy.io import fits

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import pcigale.sed_modules
from itertools import chain
import importlib
import os.path
from joblib import Parallel, delayed
from pcigale.warehouse import SedWarehouse
from scipy import constants as cst
import multiprocessing as mp
from functools import partial
from statsmodels.stats.weightstats import DescrStatsW 

def limit_spec(wavelength,spectrum,L_min,L_max):
    """extracts spectrum and associated errors between L_min and L_max
    """
    mask = (wavelength >= L_min) & (wavelength<=L_max)
    return wavelength[mask], spectrum[mask]

def rescale(sample,minimum,maximum):
    return sample*(maximum-minimum) + minimum



def extract_lines(params,wave,spec):
    """
    Integrate emission lines and separate them from the continuum

    Parameters
    ----------
    params : dictionnary
        CIGALE parameters
    spectre : array like
        observed spectrum

    Returns
    -------
    tuple
        integrated_lines : emission lines
        continuum : spectrum without the emission lines
        new_wave : wavelength of the spectrum without emission lines

    """
    width = params["nebular"]["lines_width"]
    lines_waves = params["nebular"]["line_waves"]
    limits = [(line - 3. * (line *width * 1e3 / cst.c),line + 3. *  (line *width * 1e3 / cst.c)) for line in lines_waves]
    lines = [limit_spec(wave,spec,limit[0],limit[1]) for limit in limits]
    wave_to_remove = np.unique(np.array(list(chain(*[line[0] for line in lines]))))
    integrated_lines = [np.trapz(C[1],C[0]) for C in lines]
    continuum =  np.delete(spec,np.in1d(wave, wave_to_remove))
    new_wave = np.delete(wave,np.in1d(wave, wave_to_remove))
    return continuum, new_wave, integrated_lines


def window_averaging(wavelength,spectrum):
     delta_lambda = wavelength[1:] - wavelength[:-1] 
     mid_point =(spectrum[1:] + spectrum[:-1])/2
     spec = np.average(mid_point,weights = delta_lambda)
     return spec
 
def binning_flux(wavelength, spectrum, n_bins,L_min,L_max):
    """Bins the spectroscopy
    
    Input : 
        wavelength - list, ordered wavelengths
        spectrum - np.array, observed fluxes at each wavelength
        n_bins - integer, number of bins
    
    Output : 
        wave_binned
        spec_binned
    """
    bins = np.linspace(start = L_min,
                           stop = L_max,
                           num = n_bins+1)
    idx = np.digitize(wavelength, bins)
    spec_b = [window_averaging(wavelength[idx == i],spectrum[idx == i]) for i in range(1,n_bins +1)]
    wave_b = [np.mean(wavelength[idx == i]) for i in range(1,n_bins +1)]
    return wave_b,spec_b


def var_trapz(wave,var):
    """We assume no correlation """
    seq_diff = (wave[1:] -wave[:-1]) #consecutive differences for the step
    coeff = 1/(4*(wave[-1]-wave[0])**2)
    sum_terms = np.sum(var[0:-1]*(seq_diff**2)) + np.sum(var[1:]*(seq_diff**2))
    res = coeff * sum_terms
    return res



def binning_variances(wavelength, variances, n_bins,L_min,L_max):
    """Bins the spectroscopy and the associated uncertainties.
    We assume no correlation 
    
    Input : 
    wavelength - list, ordered wavelengths
    spectrum - np.array, observed fluxes at each wavelength
    n_bins - integer, number of bins
    
    Output : 
    wave_binned
    spec_binned
    """
    bins = np.linspace(start = L_min,
                           stop = L_max,
                           num = n_bins+1)
    idx = np.digitize(wavelength, bins)
    variances_binned = [var_trapz(wavelength[idx == i],variances[idx == i]) for i in range(1,n_bins+1)]
    wave_binned = [np.mean(wavelength[idx == i]) for i in range(1,n_bins+1)]
    return wave_binned,variances_binned

def compute_covar_spectro(observed_galaxy, CIGALE_parameters):
    width = CIGALE_parameters["nebular"]["lines_width"]
    wave = observed_galaxy["spectroscopy_wavelength"]
    err = observed_galaxy["spectroscopy_err"]**2
    L_min =  CIGALE_parameters['wavelength_limits']["min"]
    L_max =CIGALE_parameters['wavelength_limits']["max"]
    lim_wave, lim_err = limit_spec(wave,
                                    err,
                                    L_min,
                                    L_max)
    
    if "lines" in CIGALE_parameters["mode"]:
        limits = [(line_wave - 3. * (line_wave *width * 1e3 / cst.c), line_wave + 3. *  (line_wave *width * 1e3 / cst.c)) for line_wave in CIGALE_parameters["nebular"]["line_waves"]]
        lines = [limit_spec(wave,err,limit[0],limit[1]) for limit in limits]
        wave_to_remove = np.array(list(chain(*[line[0] for line in lines])))
        err_to_remove = err[np.in1d(wave, wave_to_remove)]
        #covar_lines = np.array([var_trapz(C[1],C[0]) for C in lines])
        lim_err =  np.setdiff1d(lim_err,err_to_remove)
        lim_wave = np.setdiff1d(lim_wave,wave_to_remove)

    _,covar_continuum = binning_variances(lim_wave, 
                                          lim_err, 
                                          CIGALE_parameters["n_bins"],
                                          L_min,
                                          L_max)
    return np.diag(covar_continuum)


def sample_to_cigale_input(sample,
                           CIGALE_parameters = None,
                           weights_discrete = None):
    """sample from the proposal
    
    Input : 
        sample     - ndarray of shape n_sample,n_params
        CIGALE_parameters['module_parameters_to_fit'] - Dictionnary  with parameter_name as keys and
                    dictionary of type, min and max
                    example : parameters = {"tau_main":{"type":"log","min":2,"max" :3},
                                            "age_main":{"type":"unif","min":500,"max" :1000}}
        CIGALE_parameters['module_parameters_discrete']- Dictionnary  with parameter_name as keys and
                    a list of values
                    example : parameters = {"metallicity":[0.04,0.2]}
                    
    """
    continuous_parameters_list =  CIGALE_parameters['module_parameters_to_fit']
    discrete_parameters = CIGALE_parameters["module_parameters_discrete"]
    discrete_parameters_names = list(discrete_parameters.keys())
    continuous_parameter_names = list(continuous_parameters_list.keys())
    logscale_params = [key for key in continuous_parameters_list if continuous_parameters_list[key]["type"] == "log"]
    param_frame = pd.DataFrame(sample, 
                               columns = continuous_parameter_names + discrete_parameters_names) # change to continuous + discrete 
    for name in continuous_parameter_names:
        param_frame[name] = rescale(stats.norm.cdf(param_frame[name]),
                               continuous_parameters_list[name]["max"],
                               continuous_parameters_list[name]["min"])
        
        if name in logscale_params:
            param_frame[name] = 10**param_frame[name]
    


    for name in discrete_parameters.keys():
        param_frame[name] = np.array(discrete_parameters[name])[param_frame[name].astype(int)]
                                             
    modules_params = [list(pcigale.sed_modules.get_module(module,blank = True).parameter_list.keys()) for module in CIGALE_parameters['module_list']]
    parameter_list =[[{param:param_frame[param].iloc[i] for param in module_params} for module_params in modules_params] for i in range(param_frame.shape[0])] 
    
    list_deep_modules = [name for name in CIGALE_parameters['module_list'] if "deep" in name]
    deep_names = [str(module) + "." + i for module in list_deep_modules for i in pcigale.sed_modules.get_module(module,blank = True).parameter_list.keys()]
    old_names = list(chain(*[list(pcigale.sed_modules.get_module(module,blank = True).parameter_list.keys()) for module in list_deep_modules ]))
    
    new_names = dict(zip(old_names,deep_names))
    param_frame.rename(columns = new_names, inplace = True)
    for module_name in list_deep_modules:
        deep_params = param_frame.filter(like = module_name)
        file_name = CIGALE_parameters['path_deep']+module_name+'_parameters.csv'
        deep_params.to_csv(file_name, index = False, sep = " ")
    param_frame.to_csv(CIGALE_parameters['file_store'], mode ='a',index = False)
    return parameter_list


def cigale(params_input_cigale,CIGALE_parameters,warehouse):
    photo, spectro, lines = np.ones(2),np.ones(2),np.ones(2)
    SED = warehouse.get_sed(CIGALE_parameters['module_list'],params_input_cigale)
    if "photo" in CIGALE_parameters["mode"]:
        photo = np.array([SED.compute_fnu(band) for band in CIGALE_parameters['bands']])

    if "spectro" in CIGALE_parameters["mode"]:
        wavelength,spectrum = SED.wavelength_grid, SED.fnu
        L_min =  CIGALE_parameters['wavelength_limits']["min"]
        L_max =CIGALE_parameters['wavelength_limits']["max"]
        new_spec, new_wave,lines = extract_lines(CIGALE_parameters,
                                                 wavelength,
                                                 spectrum)
        lim_wave, lim_spec = limit_spec(new_wave,
                                        new_spec,
                                        L_min,
                                        L_max)
        _,spectro = binning_flux(lim_wave,
                                 lim_spec,
                                 CIGALE_parameters['n_bins'],
                                  L_min,
                                  L_max)
    if "lines" in CIGALE_parameters["mode"]:
        lines = lines
    elif "lines" not in CIGALE_parameters["mode"]:
        lines = np.ones(2)
    return  photo,np.array(spectro), lines
    
def scale_factor_pre_computation(target_photo,
                                 cov_photo,
                                 target_spectro,
                                 cov_spectro,
                                 mode):
    """Pre-computes parts of the numerator of the max-likelihood estimator of 
        alpha. If spectro or photo is not available, returns np.eye(2) to simplify 
        compute_constant
    """
    if "spectro" in mode and "photo" in mode : 
        inv_cov_spectro = np.linalg.pinv(cov_spectro)
        inv_cov_photo = np.linalg.pinv(cov_photo)
        half_num_constant_spectro = target_spectro @ inv_cov_spectro
        half_num_constant_photo = target_photo @ inv_cov_photo
        
    elif "spectro" in mode and not "photo" in mode:
        inv_cov_spectro = np.linalg.pinv(cov_spectro)
        half_num_constant_spectro = target_spectro @ inv_cov_spectro
        half_num_constant_photo = np.zeros(2)
        inv_cov_photo =  0*np.eye(2)
        
    elif (not "spectro" in mode) and "photo" in mode:
        inv_cov_spectro =  0*np.eye(2)
        inv_cov_photo = np.linalg.pinv(cov_photo)
        half_num_constant_spectro =  np.zeros(2)
        half_num_constant_photo = target_photo @ inv_cov_photo
        
        
    return inv_cov_spectro, inv_cov_photo, half_num_constant_spectro,  half_num_constant_photo


def compute_constant(SED_photo, SED_spectro, pre_computed_factors,weight_spectro = 1):
    """computes the scale constant alpha
    """
    weights_photo = 1
    weights_spectro = weight_spectro 
    inv_cov_spectro, inv_cov_photo, half_num_spectro,  half_num_photo = pre_computed_factors
    num = weights_photo * half_num_photo @ SED_photo  + weights_spectro * half_num_spectro @ SED_spectro
    denom =( weights_photo * SED_photo.T @ inv_cov_photo  @ SED_photo )+( weights_spectro * SED_spectro.T @ inv_cov_spectro @ SED_spectro)
    constant = num/denom
    return constant

def compute_scaled_SED(sample,constants,weight_spectro,CIGALE_parameters,warehouse):
    """Computes the scaled SED, to be fed directly to the likelihood function
    Needs to be parallelized
    """

    n_jobs = CIGALE_parameters["n_jobs"]
    S =sample.copy()
    cigale_input = sample_to_cigale_input(S, CIGALE_parameters)
    global _compute_scaled_SED
    if CIGALE_parameters["deep_modules"] is not None:
        for deep_module in CIGALE_parameters["deep_modules"]:
            importlib.reload(deep_module)
    

    def _compute_scaled_SED(input_cigale):
        SED = cigale(input_cigale,CIGALE_parameters,warehouse)
        SED_photo = SED[0]
        SED_spectro = SED[1]
        SED_lines = SED[2]
        constant = compute_constant(SED_photo, SED_spectro,constants,weight_spectro)
        scaled_photo = constant*SED_photo
        scaled_spectro = constant*SED_spectro
        scaled_lines = constant*SED_lines
        
        return scaled_photo,scaled_spectro, scaled_lines
    with mp.Pool(processes=n_jobs) as pool:
        computed = pool.map(_compute_scaled_SED, cigale_input)
    # MARCHE PAS computed = Parallel(n_jobs = n_jobs)(delayed(_compute_scaled_SED)(input_cigale ) for input_cigale in cigale_input)
                                                                     
    #computed = [_compute_scaled_SED(input_cigale) for input_cigale in cigale_input]
    scaled_photo = [res[0] for res in computed]
    scaled_spectro = [res[1] for res in computed]
    scaled_lines = [res[2] for res in computed]
    return scaled_photo,scaled_spectro, scaled_lines



def lim_target_spectro(observed_galaxy,CIGALE_parameters):
    wave =observed_galaxy["spectroscopy_wavelength"]
    spectrum = observed_galaxy["spectroscopy_fluxes"]
    L_min = CIGALE_parameters['wavelength_limits']["min"]
    L_max = CIGALE_parameters['wavelength_limits']["max"]
    lim_wave, lim_spec = limit_spec(wave,
                                    spectrum,
                                    L_min,
                                    L_max)
    return np.array(lim_wave),np.array(lim_spec)


def extract_target(observed_galaxy,CIGALE_parameters):
    target_photo, covar_photo = None, None
    target_spectro, covar_spectro = None, None
    target_lines, covar_lines = None, None
    
    if  observed_galaxy["photometry_fluxes"] is not None and "photo" in CIGALE_parameters["mode"]:
            target_photo = np.array(observed_galaxy["photometry_fluxes"])
            covar_photo = np.diag(observed_galaxy["photometry_err"]**2)
    
    if observed_galaxy["spectroscopy_fluxes"]  is not None and "spectro" in CIGALE_parameters["mode"]:
        wave_spectro,target_spectro = lim_target_spectro(observed_galaxy,
                                                 CIGALE_parameters)
        target_spectro,wave_spectro,_ = extract_lines(CIGALE_parameters,
                                                    wave_spectro,
                                                    target_spectro)
        wave_spectro,target_spectro = binning_flux(wave_spectro,
                             target_spectro,
                             CIGALE_parameters['n_bins'],
                             CIGALE_parameters['wavelength_limits']["min"],
                             CIGALE_parameters['wavelength_limits']["max"])
        
        covar_spectro = compute_covar_spectro(observed_galaxy, CIGALE_parameters)
    if "lines_fluxes" in observed_galaxy.keys() and "lines" in CIGALE_parameters["mode"]:
        target_lines = observed_galaxy["lines_fluxes"]
        covar_lines =  np.diag(observed_galaxy["lines_err"]**2)
        
        
    return target_photo,target_lines, target_spectro,covar_photo,covar_spectro, covar_lines

class target_SED(object):
    """ Class to be used to call TAMIS, must have a self.dim, self.log_likelihood
    and self.log_prior
    """
    def __init__(self,
                 observed_galaxy,
                 CIGALE_parameters,
                 warehouse,
                 dim_prior,
                 weight_spectro = 1):
        self.dim = dim_prior
        targ_covar= extract_target(observed_galaxy,CIGALE_parameters)
        self.target_photo,self.target_lines, = targ_covar[0:2]
        self.target_spectro,self.covar_photo = targ_covar[2:4]
        self.covar_spectro, self.covar_lines = targ_covar[4:6]
        self.pre_computed_constants =scale_factor_pre_computation(self.target_photo ,
                                                                  self.covar_photo,
                                                                  self.target_spectro,
                                                                  self.covar_spectro,
                                                                  CIGALE_parameters["mode"])
        self.weight_spectro = weight_spectro
        self.warehouse = warehouse
        self.CIGALE_parameters = CIGALE_parameters
        
        
    def log_likelihood(self,sample):
        weight_spectro = self.weight_spectro
        target_photo = self.target_photo
        target_spectro = self.target_spectro
        covar_photo = self.covar_photo
        covar_spectro = self.covar_spectro
        target_lines = self.target_lines
        covar_lines = self.covar_lines
        constants = self.pre_computed_constants
        CIGALE_parameters = self.CIGALE_parameters
        warehouse = self.warehouse
        scaled_SED_photo,scaled_SED_spectro,scaled_SED_lines= compute_scaled_SED(sample,
                                                                 constants,
                                                                 weight_spectro,
                                                                 CIGALE_parameters,
                                                                 warehouse)
        log_likelihood_photo = 0
        log_likelihood_lines = 0
        log_likelihood_spectro = 0
        #We switch mean and x for vectorization
        if "photo" in CIGALE_parameters["mode"]:
            log_likelihood_photo = np.array(stats.multivariate_normal.logpdf(x=scaled_SED_photo,
                                                                             mean = target_photo,
                                                                             cov = covar_photo,
                                                                             allow_singular = True))

        if "spectro" in CIGALE_parameters["mode"]:
            log_likelihood_spectro = np.array(stats.multivariate_normal.logpdf(x=scaled_SED_spectro,
                                                                               mean = target_spectro,
                                                                               cov = covar_spectro,
                                                                               allow_singular = True))

        if "lines" in CIGALE_parameters["mode"]: 
            log_likelihood_lines = np.array(stats.multivariate_normal.logpdf(x=scaled_SED_lines,
                                                                                mean = target_lines,
                                                                                cov = covar_lines,
                                                                                allow_singular = True))
                
        total_loglike = weight_spectro*log_likelihood_spectro + log_likelihood_photo + log_likelihood_lines 
        
        return total_loglike
    
    def log_prior(self,sample):
        """ Computes the unitary gaussian logpdf as the parameter space is 
        projected in R^dim
        """
        dim_cont = len(self.CIGALE_parameters['module_parameters_to_fit'])
        dim_disc = len(self.CIGALE_parameters['module_parameters_discrete'])
        var0 = [1]*dim_cont
        logprior_cont = stats.multivariate_normal.logpdf(sample[:,:dim_cont],
                                                    mean = np.zeros((dim_cont,)),
                                                    cov = np.diag(var0))
        logprior_disc = 0 #To adjust later ? 
        return logprior_cont + logprior_disc
    
def create_output_files(CIGALE_parameters,result):
    parameters = pd.read_csv(CIGALE_parameters['file_store'])
    name_col = parameters.columns[0]
    clean = parameters.drop(parameters[parameters[name_col]==name_col].index)
    clean["weights"] = result.final_weights
    temp = np.zeros(len(result.final_weights))
    temp[result.max_target] = 1
    clean["MAP"] = temp
    clean.columns = split_name_params(clean.columns)
    clean.to_csv(CIGALE_parameters['file_store'],index = False)
    
def check_output_file(CIGALE_parameters):
    while os.path.exists(CIGALE_parameters['file_store']):
        CIGALE_parameters['file_store'] =  CIGALE_parameters['file_store'].replace(".csv","_1.csv")
        
def fit(galaxy_obs , CIGALE_parameters, TAMIS_parameters):
    module_list = CIGALE_parameters['module_list']
    #warehouse = SedWarehouse(nocache ='deep_nebular' )
    warehouse = SedWarehouse(nocache ='deep_bc03_pca_norm' )

    check_output_file(CIGALE_parameters)
    target_distrib = target_SED(galaxy_obs,
                                CIGALE_parameters,
                                warehouse,
                                TAMIS_parameters['dim_prior'],
                                1)
    Sampler = TAMIS(target = target_distrib,
                   n_comp = TAMIS_parameters['n_comp'],
                   init_theta = TAMIS_parameters['init_theta'],
                   ESS_tol = TAMIS_parameters['ESS_tol'],
                   proposal = TAMIS_parameters['proposal'],
                   n_sample = TAMIS_parameters['n_sample'],
                   alpha = TAMIS_parameters['alpha'],
                   verbose = TAMIS_parameters['verbose'])
    result = Sampler.result(T = TAMIS_parameters['T_max'])
    create_output_files(CIGALE_parameters, result)
    estimates = analyse_results(CIGALE_parameters)
    
    return result, estimates


def read_galaxy(file_photo, file_spectro):
    photo = pd.read_csv(file_photo)
    spectro = pd.read_csv(file_spectro)
    observed_galaxy  = {"spectroscopy_wavelength":spectro["wavelength"].to_numpy(),
                        "spectroscopy_fluxes":spectro["flux"].to_numpy(),
                        "spectroscopy_err" : spectro["err"].to_numpy(),
                        "photometry_fluxes" : photo["flux"].to_numpy(),
                        "photometry_err" : photo["err"].to_numpy(),
                        "bands" : photo["band"]
                        }
                        
    return observed_galaxy
def read_spectro_moons(file):
        
    hdul = fits.open(file)
    noisy = hdul[1].data
    noise_array =  hdul[8].data
    wave = np.linspace(hdul[0].header["WMIN"],
                       hdul[0].header["WMAX"],
                       len(noisy))
    redshift = hdul[0].header["Z"]
    return noisy,noise_array,wave, redshift


def read_galaxy_fits(photo_file,spectro_file,ident = None):
    photo_flux = None
    photo_err = None
    spec_flux = None
    spec_err = None
    spec_wavelength = None
    if photo_file is not None:
        #to rewrite with bands in database
        table = Table.read(photo_file)
        table = table.to_pandas()
        if ident :
            photo = table[list(table.columns[4:32])][table['id'] == ident]
            z =table["redshift"][table["id"]==ident].iloc[0]
        else :
            photo = table[list(table.columns[4:32])][0]
            z = table["redshift"][0][0]
        bands = []
        err = []
        for band in photo.columns:
            if band.endswith('_err'):
                err.append(band)
            else:
                bands.append(band)
        photo_flux = np.array(photo[bands]).reshape(len(bands),)
        photo_err = np.array(photo[err]).reshape(len(err),)
    spec = Table.read(spectro_file)
    spec_flux = np.array(spec["Fnu"])
    spec_err = np.array(0.1*spec_flux)
    spec_wavelength = np.array(spec["wavelength"])

    
    observed_galaxy  = {"spectroscopy_wavelength":spec_wavelength,
                        "spectroscopy_fluxes":spec_flux,
                        "spectroscopy_err" : spec_err,
                        "photometry_fluxes" : photo_flux,
                        "photometry_err" :photo_err,
                        "bands" : bands,
                        "redshift" : z
                        }
        # if spectro_file is not None:
    #     spec_flux,spec_err,spec_wavelength,z = read_spectro_moons(spectro_file)
    # observed_galaxy  = {"spectroscopy_wavelength":spec_wavelength,
    #                     "spectroscopy_fluxes":spec_flux,
    #                     "spectroscopy_err" : spec_err,
    #                     "photometry_fluxes" : photo_flux,
    #                     "photometry_err" :photo_err,
    #                     "bands" : bands,
    #                     "redshift" : z
    #                     }
    return observed_galaxy



def read_galaxy_moons(spectro_file, photo_file, ident = None):
    photo_flux = None
    photo_err = None
    spec_flux = None
    spec_err = None
    spec_wavelength = None
    if photo_file is not None:
        #to rewrite with bands in database
        table = Table.read(photo_file)
        table = table.to_pandas()
        if ident :
            photo = table[list(table.columns[4:32])][table['id'] == ident]
            z =table["redshift"][table["id"]==ident].iloc[0]
        else :
            photo = table[list(table.columns[4:32])][0]
            z = table["redshift"][0][0]
        bands = []
        err = []
        for band in photo.columns:
            if band.endswith('_err'):
                err.append(band)
            else:
                bands.append(band)
        photo_flux = np.array(photo[bands]).reshape(len(bands),)
        photo_err = np.array(photo[err]).reshape(len(err),)
    if spectro_file is not None:
        spec_flux,spec_err,spec_wavelength,z = read_spectro_moons(spectro_file)
    observed_galaxy  = {"spectroscopy_wavelength":spec_wavelength,
                        "spectroscopy_fluxes":spec_flux,
                        "spectroscopy_err" : spec_err,
                        "photometry_fluxes" : photo_flux,
                        "photometry_err" :photo_err,
                        "bands" : bands,
                        "redshift" : z
                        }
    return observed_galaxy


def split_name_params(columns):
    new_columns = []
    for col in columns:
        if len(col.split(".")) !=1:
            col = col.split(".")[1]
        new_columns.append(col)
    return new_columns
def line_drawer(x=None,y=None, hue = None, line_dict = None, line_color = 'r',**kwargs):
    ax = plt.gca()
    if line_dict is not None :
        if x.name in line_dict:
            ax.axvline(line_dict[x.name],0,1,color = line_color)
    return ax

def plot_result(CIGALE_parameters, line_dict_fit = None, title = None):
    ### MANQUE line_drawer...
    
    
    results = pd.read_csv(CIGALE_parameters["file_store"])
    to_plot=results[CIGALE_parameters["module_parameters_to_fit"]]
    
    try:
        g = sns.PairGrid(to_plot, corner =True,diag_sharey=False)
        g.map_lower(sns.kdeplot, fill = True,weights = results["weights"], levels = 5)
        g.map_diag(sns.kdeplot,fill = False, weights =results["weights"],levels = 5)
        g.map_diag(line_drawer, line_dict = line_dict_fit, line_color = 'r')
        g.fig.suptitle(title)
    except :
        to_plot2 = to_plot.sample(100)
        g = sns.PairGrid(to_plot2, corner =True,diag_sharey=False)
        g.map_lower(sns.kdeplot, fill = True,weights = results["weights"], levels = 5)
        g.map_diag(sns.kdeplot,fill = False, weights =results["weights"],levels = 5)
        g.map_diag(line_drawer, line_dict = line_dict_fit, line_color = 'r')
        g.fig.suptitle(title)
    to_hist = []
    for param in CIGALE_parameters["module_parameters_discrete"]:
        if len(results[param].unique()) > 1:
            to_hist.append(param)
    to_plot_hist = results[to_hist]
    n_rows = np.int(np.sqrt(len(to_plot_hist.columns))) + 1
    n_cols = np.int(np.sqrt(len(to_plot_hist.columns))) +1
    
    fig,axes = plt.subplots(nrows = n_rows, ncols = n_cols)
    for i, column in enumerate(to_plot_hist.columns):
        #plt.figure()
        sns.histplot(x=to_plot_hist[column].astype(str), 
                     weights =results["weights"], 
                     kde= False,
                     ax = axes[i//n_cols,i%n_cols])
        #plt.show()
    if title :
        plt.suptitle(title)
        
def analyse_results(CIGALE_parameters):
    results = pd.read_csv(CIGALE_parameters["file_store"])
    to_plot=results[CIGALE_parameters["module_parameters_to_fit"]]
    
    res = {}
    for col in to_plot.columns:
        weighted_stats = DescrStatsW(to_plot[col], weights = results["weights"])
        res[col]  = {"mean":weighted_stats.mean,
                     "var":weighted_stats.var,
                     "sd" :weighted_stats.std,
                     #"95% credible interval":np.array(weighted_stats.quantile([0.05,0.95])),
                     "max":to_plot[results["MAP"]==1][col].array[0],
                     }
    return res


def plot_de_secours(CIGALE_parameters, line_dict_fit = None, title = None):

    results = pd.read_csv(CIGALE_parameters["file_store"])
    to_plot=results[CIGALE_parameters["module_parameters_to_fit"]]
    to_hist = []
    for param in CIGALE_parameters["module_parameters_discrete"]:
        if len(results[param].unique()) > 1:
            to_hist.append(param)
    to_plot_hist = results[to_hist]
    n_rows = np.int(np.sqrt(len(to_plot_hist.columns))) + 1
    n_cols = np.int(np.sqrt(len(to_plot_hist.columns))) + 1
    fig,axes = plt.subplots(nrows = n_rows, ncols = n_cols)
    for i, column in enumerate(to_plot_hist.columns):
        #plt.figure()
        sns.histplot(x=to_plot_hist[column].astype(str), 
                     weights =results["weights"], 
                     kde= False,
                     ax = axes[i//n_cols,i%n_cols])
        #plt.show()
    for column in to_plot.columns:
        #plt.figure()
        sns.kdeplot(to_plot[column], 
                     weights =results["weights"])
        plt.axvline(line_dict_fit[column])
        plt.show()
    
    if title :
        plt.suptitle(title)