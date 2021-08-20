#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 12:26:01 2021

@author: aufort
"""

from  TAMIS import TAMIS
import seaborn as sns
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



def extract_lines(gal,wave,spec):
    """
    Integrate emission lines and separate them from the continuum

    Parameters
    ----------
    gal : galaxy
        CIGALE galaxy object
    spectre : array like
        observed spectrum

    Returns
    -------
    tuple
        integrated_lines : emission lines
        continuum : spectrum without the emission lines
        new_wave : wavelength of the spectrum without emission lines

    """
    width = gal.info["nebular.lines_width"]
    limits = [(line[0] - 3. * (line[0] *width * 1e3 / cst.c),line[0] + 3. *  (line[0] *width * 1e3 / cst.c)) for line in gal.lines.values()]
    lines = [limit_spec(wave,spec,limit[0],limit[1]) for limit in limits]
    wave_to_remove = np.array(list(chain(*[line[0] for line in lines])))
    flux_to_remove = gal.fnu[np.in1d(wave, wave_to_remove)]
    integrated_lines = [np.trapz(C[1],C[0]) for C in lines]
    continuum =  np.setdiff1d(spec,flux_to_remove)
    new_wave = np.setdiff1d(wave,wave_to_remove)
    return integrated_lines, continuum, new_wave

def binning_flux(wavelength, spectrum, n_bins):
    """Bins the spectroscopy and the associated uncertainties.
        We assume no correlation and constant band width
    
    Input : 
        wavelength - list, ordered wavelengths
        spectrum - np.array, observed fluxes at each wavelength
        n_bins - integer, number of bins
    
    Output : 
        wave_binned
        spec_binned
    """
    bins = np.logspace(start = np.log10(wavelength[0]),
                       stop = np.log10(wavelength[-1]-1),
                       num = n_bins)
    idx = np.digitize(wavelength, bins)
    spec_binned = [np.mean(spectrum[idx == i]) for i in range(1,n_bins+1)]
    wave_binned = [np.mean(wavelength[idx == i]) for i in range(1,n_bins+1)]
    return wave_binned,spec_binned

def compute_covar_spectro(observed_galaxy, CIGALE_parameters):
    width = CIGALE_parameters["nebular"]["lines_width"]
    wave = observed_galaxy["spectroscopy_wavelength"]
    err = observed_galaxy["spectroscopy_err"]**2
    # limits = [(line_wave - 3. * (line_wave *width * 1e3 / cst.c), line_wave + 3. *  (line_wave *width * 1e3 / cst.c)) for line_wave in CIGALE_parameters["nebular"]["line_waves"]]
    # lines = [limit_spec(wave,err,limit[0],limit[1]) for limit in limits]
    # wave_to_remove = np.array(list(chain(*[line[0] for line in lines])))
    # err_to_remove = err[np.in1d(wave, wave_to_remove)]
    # covar_lines = np.array([var_trapz(C[1],C[0]) for C in lines])
    # continuum =  np.setdiff1d(err,err_to_remove)
    # new_wave = np.setdiff1d(wave,wave_to_remove)
    
    _,covar_continuum = binning_variances(wave,err, CIGALE_parameters["n_bins"])
    # return covar_lines, covar_continuum
    return covar_continuum

def var_trapz(var,wave):
    seq_diff = (wave[1:] -wave[:-1])**2 #consecutive differences for the step
    sum_middle = seq_diff[:-1] + seq_diff[1:] # consecutive sums of step
    res = seq_diff[0]*var[0] + seq_diff[-1]*var[-1] + np.sum(sum_middle*var[1:-1])
    return 0.25*res
def binning_variances(wavelength, variances, n_bins):
    """Bins the spectroscopy and the associated uncertainties.
    We assume no correlation and constant bandwidth
    
    Input : 
    wavelength - list, ordered wavelengths
    spectrum - np.array, observed fluxes at each wavelength
    n_bins - integer, number of bins
    
    Output : 
    wave_binned
    spec_binned
    """
    bins = np.logspace(start = np.log10(wavelength[0]),
                           stop = np.log10(wavelength[-1]-1),
                           num = n_bins)
    idx = np.digitize(wavelength, bins)
    variances_binned = [np.sum(variances[idx == i])/(np.sum(idx==i)**2) for i in range(1,n_bins+1)]
    wave_binned = [np.mean(wavelength[idx == i]) for i in range(1,n_bins+1)]
    return wave_binned,variances_binned

def sample_to_cigale_input(sample,CIGALE_parameters = None):
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
    continuous_parameter_names = continuous_parameters_list.keys()
    logscale_params = [key for key in continuous_parameters_list if continuous_parameters_list[key]["type"] == "log"]
    param_frame = pd.DataFrame(stats.norm.cdf(sample), columns = continuous_parameter_names)
    for name in continuous_parameter_names:
        param_frame[name] = rescale(param_frame[name],
                               continuous_parameters_list[name]["max"],
                               continuous_parameters_list[name]["min"])
        
        if name in logscale_params:
            param_frame[name] = 10**param_frame[name]
    
    discrete_parameters = CIGALE_parameters["module_parameters_discrete"]
    for name in discrete_parameters.keys():
        param_frame[name] = np.random.choice(discrete_parameters[name],
                                             size = param_frame.shape[0])
    modules_params = [list(pcigale.sed_modules.get_module(module,blank = True).parameter_list.keys()) for module in CIGALE_parameters['module_list']]
    parameter_list =[[{param:param_frame[param].iloc[i] for param in module_params} for module_params in modules_params] for i in range(param_frame.shape[0])] 
    
    list_deep_modules = [name for name in CIGALE_parameters['module_list'] if "deep" in name]
    deep_names = [str(module) + "." + i for module in list_deep_modules for i in pcigale.sed_modules.get_module(module,blank = True).parameter_list.keys()]
    old_names = list(chain(*[list(pcigale.sed_modules.get_module(module,blank = True).parameter_list.keys()) for module in list_deep_modules ]))
    
    new_names = dict(zip(old_names,deep_names))
    param_frame.rename(columns = new_names, inplace = True)
    deep_params = param_frame.filter(like = "deep")
    deep_params.to_csv(CIGALE_parameters['path_deep'], index = False, sep = " ")
    param_frame.to_csv(CIGALE_parameters['file_store'], mode ='a',index = False)
    return parameter_list


def cigale(params_input_cigale,CIGALE_parameters,warehouse):
    
    SED = warehouse.get_sed(CIGALE_parameters['module_list'],params_input_cigale)
    photo = np.array([SED.compute_fnu(band) for band in CIGALE_parameters['bands']])
    wavelength,spectrum = SED.wavelength_grid, SED.fnu
    lim_wave, lim_spec = limit_spec(wavelength,spectrum,CIGALE_parameters['wavelength_limits']["min"],CIGALE_parameters['wavelength_limits']["max"])
    #lines, lim_spec, lim_wave  = extract_lines(SED,wave,spec)
    _,spectro = binning_flux(lim_wave,lim_spec,CIGALE_parameters['n_bins'])
    
    return  photo,np.array(spectro) #, lines
    
def scale_factor_pre_computation(target_photo,
                                 cov_photo,
                                 target_spectro,
                                 cov_spectro):
    """Pre-computes parts of the numerator of the max-likelihood estimator of 
        alpha
    """
    inv_cov_spectro = np.linalg.pinv(cov_spectro)
    inv_cov_photo = np.linalg.pinv(cov_photo)
    half_num_constant_spectro = target_spectro @ inv_cov_spectro
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
    cigale_input = sample_to_cigale_input(sample, CIGALE_parameters)
    global _compute_scaled_SED
    if CIGALE_parameters["deep_modules"]:
        for deep_module in CIGALE_parameters["deep_modules"]:
            importlib.reload(deep_module)
    

    def _compute_scaled_SED(input_cigale):
        SED = cigale(input_cigale,CIGALE_parameters,warehouse)
        SED_photo = SED[0]
        SED_spectro = SED[1]
        #SED_lines = SED[2]
        constant = compute_constant(SED_photo, SED_spectro,constants,weight_spectro)
        scaled_photo = constant*SED_photo
        scaled_spectro = constant*SED_spectro
        #scaled_lines.append(constant*SED_lines)
        return scaled_photo,scaled_spectro#, scaled_lines
    with mp.Pool(processes=n_jobs) as pool:
        computed = pool.map(_compute_scaled_SED, cigale_input)
    # computed = Parallel(n_jobs = n_jobs)(delayed(_compute_scaled_SED)(input_cigale
    #                                                                   ) for input_cigale in cigale_input)
    #computed = [_compute_scaled_SED(input_cigale) for input_cigale in cigale_input]
    scaled_photo = [res[0] for res in computed]
    scaled_spectro = [res[1] for res in computed]
    #scaled_lines = [res[2] for res in computed]
    return scaled_photo,scaled_spectro#, scaled_lines

def lim_target_spectro(observed_galaxy,CIGALE_parameters):
    wave =observed_galaxy["spectroscopy_wavelength"]
    spectrum = observed_galaxy["spectroscopy_fluxes"]
    lim_wave, lim_spec = limit_spec(wave,
                                    spectrum,
                                    CIGALE_parameters['wavelength_limits']["min"],
                                    CIGALE_parameters['wavelength_limits']["max"])
    _,binned_spec = binning_flux(lim_wave,lim_spec,CIGALE_parameters['n_bins'])
    return binned_spec


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
        self.target_photo = observed_galaxy["photometry_fluxes"]
        self.target_spectro = lim_target_spectro(observed_galaxy,CIGALE_parameters)
        self.covar_photo = np.diag(observed_galaxy["photometry_err"]**2)
        #self.covar_lines,self.covar_spectro = compute_covar_spectro(observed_galaxy, CIGALE_parameters)
        self.covar_spectro =  np.diag(compute_covar_spectro(observed_galaxy, CIGALE_parameters))
    
        self.pre_computed_constants =scale_factor_pre_computation(self.target_photo ,
                                                                  self.covar_photo,
                                                                  self.target_spectro,
                                                                  self.covar_spectro)
        self.weight_spectro = weight_spectro
        self.warehouse = warehouse
        self.CIGALE_parameters = CIGALE_parameters
        
        
    def log_likelihood(self,sample):
        weight_spectro = self.weight_spectro
        target_photo = self.target_photo
        target_spectro = self.target_spectro
        covar_photo = self.covar_photo
        covar_spectro = self.covar_spectro
        #covar_lines = self.covar_lines
        constants = self.pre_computed_constants
        CIGALE_parameters = self.CIGALE_parameters
        warehouse = self.warehouse
        # scaled_SED_photo,scaled_SED_spectro,scaled_SED_lines = compute_scaled_SED(sample,
        #                                                          constants,
        #                                                          weight_spectro,
        #                                                          CIGALE_parameters,
        #                                                          warehouse)
        scaled_SED_photo,scaled_SED_spectro= compute_scaled_SED(sample,
                                                                 constants,
                                                                 weight_spectro,
                                                                 CIGALE_parameters,
                                                                 warehouse)
        #We switch mean and x for vectorization
        log_likelihood_photo = np.array(stats.multivariate_normal.logpdf(x=scaled_SED_photo,
                                                                         mean = target_photo,
                                                                         cov = covar_photo,
                                                                         allow_singular = True))
        log_likelihood_spectro = np.array(stats.multivariate_normal.logpdf(x=scaled_SED_spectro,
                                                                           mean = target_spectro,
                                                                           cov = covar_spectro,
                                                                           allow_singular = True))
        # log_likelihood_lines = np.array(stats.multivariate_normal.logpdf(x=scaled_SED_lines,
        #                                                                    mean = target_lines,
        #                                                                    cov = covar_lines,
        #                                                                    allow_singular = True))
        total_loglike = weight_spectro*log_likelihood_spectro + log_likelihood_photo #+ log_likelihood_lines 
        
        return total_loglike
    
    def log_prior(self,sample):
        """ Computes the unitary gaussian logpdf as the parameter space is 
        projected in R^dim
        """
        dim = self.dim
        var0 = [1]*dim
        logprior = stats.multivariate_normal.logpdf(sample,
                                                    mean = np.zeros((dim,)),
                                                    cov = np.diag(var0))
        return logprior
    
def create_output_files(CIGALE_parameters,result):
    parameters = pd.read_csv(CIGALE_parameters['file_store'])
    name_col = parameters.columns[0]
    clean = parameters.drop(parameters[parameters[name_col]==name_col].index)
    clean["weights"] = result.final_weights
    clean.columns = split_name_params(clean.columns)
    clean.to_csv(CIGALE_parameters['file_store'],index = False)
    
def check_output_file(CIGALE_parameters):
    while os.path.exists(CIGALE_parameters['file_store']):
        CIGALE_parameters['file_store'] =  CIGALE_parameters['file_store'].replace(".csv","_1.csv")
        
def fit(galaxy_obs , CIGALE_parameters, TAMIS_parameters):
    module_list = CIGALE_parameters['module_list']
    warehouse = SedWarehouse(nocache = module_list)
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
                   alpha = TAMIS_parameters['alpha'])
    result = Sampler.result(T = TAMIS_parameters['T_max'])
    analyse_results(CIGALE_parameters)
    create_output_files(CIGALE_parameters, result)
    
    
    return result


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

def split_name_params(columns):
    new_columns = []
    for col in columns:
        if len(col.split(".")) !=1:
            col = col.split(".")[1]
        new_columns.append(col)
    return new_columns



def plot_result(CIGALE_parameters):
    results = pd.read_csv(CIGALE_parameters["file_store"])
    to_plot=results[CIGALE_parameters["module_parameters_to_fit"]]
    try:
        g = sns.PairGrid(to_plot, corner =True)
        g.map_lower(sns.kdeplot, fill = True,weights = results["weights"], levels = 10)
        g.map_diag(sns.kdeplot,fill = False, levels = 10)
    except :
        to_plot2 = to_plot.sample(1000)
        g = sns.PairGrid(to_plot2, corner =True)
        g.map_lower(sns.kdeplot, fill = True,weights = results["weights"], levels = 10)
        g.map_diag(sns.kdeplot,fill = False, levels = 10)

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
        sns.histplot(x=to_plot_hist[column], 
                     weights =results["weights"], 
                     kde= False,
                     ax = axes[i//n_cols,i%n_cols])
                     #discrete = True)
        #plt.show()
        
        
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
                     "max":"later : to be added to TAMIS",
                     }
    return res