#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 14:32:42 2021

@author: aufort
"""
import importlib
import scipy.stats as stats
import pcigale.sed_modules
from GMM import GMM_fit, Mixture_t, Mixture_gaussian, multivariate_t,Mixture_gaussian_discrete
from TAMIS import TAMIS
import numpy as np
from utils import * 

proposal = Mixture_gaussian_discrete

dim_prior = 2 #Number of continuous parameters to fit
n_comp = 4 #arbitrary
ESS_tol = 300*dim_prior 
T_max = 50
n_sample = [500]*T_max

#NEED TO AUTOMATE THIS PART, USELESS TO SET UP
var0 = [3]*dim_prior
mean0 = 0
init_mean = stats.uniform.rvs(size =(n_comp,dim_prior),loc=-1,scale = 2 )
probs = [[0.5,0.5]]
init = [init_mean,
         np.array([np.diag(var0)]*n_comp),
         np.ones((n_comp,))/n_comp,
         probs]
init_theta= theta_params_discrete(init)


Sample = proposal.rvs(500,init_theta)
lpdf = proposal.logpdf(Sample, init_theta)



obs = Sample[:,2]
p_targ = [0.2,0.8]
lk_obs = np.prod([p_targ[int(obs[i])] for i in range(500)])
class gaussian_discrete(object):
    """
    Simple diagonal multivariate gaussian
    """
    def __init__(self,
                 dim = 2,
                 mean=40,
                 var = 1,
                 probs = [0.5,0.5]):
        self.dim = dim
        self.mean = [mean]*dim
        self.var = [var]*dim
        self.probs = probs
            
    def log_prior(self,sample):
         prior =np.zeros(shape = ((sample.shape[0]),))#improper 1(R)
         return prior
       
    def log_likelihood(self,sample): 
         log_lik = stats.multivariate_normal.logpdf(sample[:,:self.dim], mean = self.mean, cov = np.diag(self.var))
         log_lk_disc = np.log([self.probs[int(i)] for i in sample[:,self.dim]])
         return log_lik + log_lk_disc
     
        
targ_test = gaussian_discrete(dim = dim_prior, mean = 5,var = 1, probs = [0.2,0.8])
Sampler = TAMIS(target = targ_test,
                   n_comp = n_comp ,
                   init_theta = init_theta,
                   ESS_tol = ESS_tol,
                   proposal = proposal,
                   alpha = 50,
                   n_sample = n_sample,recycle = False)
result = Sampler.result(T = T_max)