# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 12:10:25 2020

@author: Greg
"""

import mixmod
from mixmod import gm # gm contains global constants (enum items etc.)
import itertools
from utils import logpdf_student
from matplotlib import pyplot as plt
from collections import Counter
import numpy as np
import scipy.stats as stats
from pomegranate import *
from scipy.special import xlogy


class parameters(object):
        def __init__(self,means,variances, proportions):
            self.mean  = means
            self.variance = variances
            self.proportions = proportions
            

def pome_diagonal_mixture_with_init(sample,weights_int, n_comp, init_parameters):
    Distribs = []
    for i in range(n_comp):
        Distribs.append(IndependentComponentsDistribution([NormalDistribution(init_parameters.mean[i][j],np.diag(init_parameters.variance[i])[j])for j in range(sample.shape[1])]))
    model = GeneralMixtureModel(Distribs,weights = init_parameters.proportions)
    res = model.fit(X=sample, weights  = weights_int, n_jobs = 1)
    means= [0]*n_comp
    variances = [0]*n_comp
    for k in range(n_comp) :
        means[k] = np.array([res.distributions[k].parameters[0][i].parameters[0] for i in range(sample.shape[1])])#means of the 10 1dimensinal distributions of the first component 
        variances[k] = np.diag([res.distributions[k].parameters[0][i].parameters[1] for i in range(sample.shape[1])])
    proportions = np.exp(res.weights)
    params = parameters(means = means, variances = variances, proportions = proportions)
    return params


def pome_diagonal_mixture_wo_init(sample,weights_int, n_comp):
    model = GeneralMixtureModel.from_samples(distributions = NormalDistribution,
                                                    n_components = n_comp,
                                                    X = sample,
                                                    weights =weights_int,
                                                    n_jobs = 1,
                                                    n_init = 3)
    means= [0]*n_comp
    variances = [0]*n_comp
    for k in range(n_comp) :
        means[k] = np.array([model.distributions[k].parameters[0][i].parameters[0] for i in range(sample.shape[1])])#means of the 10 1dimensinal distributions of the first component 
        variances[k] = np.diag([model.distributions[k].parameters[0][i].parameters[1]  for i in range(sample.shape[1])])
    proportions = np.exp(model.weights)
    params = parameters(means = means, variances = variances, proportions = proportions)
    return params

def modify_weights(weights,isint,trunc):
    trunc =  np.quantile(weights,trunc)
    weights2  = weights.copy()
    weights2[weights<trunc] = trunc
    print(trunc)
    if isint == True:
        weights_int = np.int64((1/trunc)*weights2) +1
        return weights_int
    return weights2



def pome_diagonal_mixture(sample,weights, n_comp, init_parameters):
    weights = modify_weights(weights,False,0.6)
    try:
        #assert False
        params = pome_diagonal_mixture_with_init(sample,weights, n_comp, init_parameters)
        if np.isinf(params.proportions[0]):
            error = True
            assert not error
    except :
        print("error with init")
        params = pome_diagonal_mixture_wo_init(sample,weights, n_comp)
        if np.isinf(params.proportions[0]):
            error = True
            assert not error, "EM failed"
    return params


def mixmod_diagonal_mixture_with_init(sample,weights_int, n_comp, init_parameters):
    par = mixmod.GaussianParameter(proportions=init_parameters.proportions,
                                            mean=init_parameters.mean,
                                            variance=init_parameters.variance)
    ini = mixmod.init(name=gm.PARAMETER, parameter=par)
    algo = mixmod.algo(nb_iteration =3)
    st = mixmod.strategy(init=ini, algo = algo, seed = 42)
    cluster = mixmod.cluster(sample,
                          n_comp,
                          gm.QUANTITATIVE,
                          strategy = st,
                          weight = weights_int, 
                          models = mixmod.gaussian_model(family = gm.DIAGONAL))
    params = cluster.best_result.parameters
    return params

def mixmod_diagonal_mixture_wo_init(sample,weights_int, n_comp):
    algo = mixmod.algo(nb_iteration =3)
    st = mixmod.strategy(algo = algo, seed = 42)
    cluster = mixmod.cluster(sample,
                          n_comp,
                          gm.QUANTITATIVE, 
                          strategy = st,
                          weight = weights_int, 
                          models = mixmod.gaussian_model(family = gm.DIAGONAL))
    params = cluster.best_result.parameters
    return params
def mixmod_diagonal_mixture_wo_ncomp(sample,weights_int):
    algo = mixmod.algo(nb_iteration =3)
    st = mixmod.strategy(algo = algo, seed = 42)
    cluster = mixmod.cluster(sample,
                      data_type = gm.QUANTITATIVE, 
                      strategy = st,
                      nb_cluster=1,
                      weight = weights_int, 
                      models = mixmod.gaussian_model(family = gm.DIAGONAL))

    params = cluster.best_result.parameters
    return params
def error_mixmod_params(params):
    return np.isnan(params.mean[0][0]) == False

def ESS(weights):
    return (np.sum(weights)**2)/np.sum(weights**2)


def coef_variation(weights):
    return np.std(weights)/np.mean(weights)

def kullback_temperature(weights):
    norm_weights = weights / np.sum(weights)
    KL= np.sum(xlogy(norm_weights,norm_weights))  
    return KL + np.log(len(norm_weights))

def mixmod_diagonal_mixture(sample,weights, n_comp, init_parameters):
    weights = modify_weights(weights,True,0.6)
    if init_parameters is not None:
        try:
            params = mixmod_diagonal_mixture_with_init(sample,
                                              weights,
                                              n_comp, 
                                              init_parameters)
            assert error_mixmod_params(params)
        except :
            try:
                print("had to change init")
                params = mixmod_diagonal_mixture_wo_init(sample,
                                                         weights,
                                                         n_comp)
                assert error_mixmod_params(params)
            except:
                print("also had to change n_comp")
                params = mixmod_diagonal_mixture_wo_ncomp(sample,weights)
                assert error_mixmod_params(params), "EM failed"
        
    else:    
        
        params =mixmod_diagonal_mixture_wo_ncomp(sample,weights)
    return params
        
def GMM_fit(sample, weights,n_comp = 3, init_parameters = None):
    #params = mixmod_diagonal_mixture(sample, weights, n_comp, init_parameters)
    params = pome_diagonal_mixture(sample, weights, n_comp, init_parameters)

    return params
    


class multivariate_t(object):
        
    def rvs(mean, cov, size=1,df=3):
        '''generate random variables of multivariate t distribution
        Parameters
        ----------
        m : array_like
            mean of random variable, length determines dimension of random variable
        S : array_like
            square array of covariance  matrix
        df : int or float
            degrees of freedom
        n : int
            number of observations, return random array will be (n, len(m))
        Returns
        -------
        rvs : ndarray, (n, len(m))
            each row is an independent draw of a multivariate t distributed
            random variable
        '''
        m = np.asarray(mean)
        d = len(m)
        if df == np.inf:
            x = np.ones((size,))
        else:
            x = np.random.chisquare(df, size)/df
        z = np.random.multivariate_normal(np.zeros(d),cov,(size,))
        return m + z/np.sqrt(x)[:,None]   # same output format as random.multivariate_normal
    
            
    def logpdf(x,mean,cov,df=3):
        '''Computes logpdf of multivariate t distribution
        Parameters
        ----------
        x : array_like
            points where the logpdf must be computed
        m : array_like
            mean of random variable, length determines dimension of random variable
        S : array_like
            square array of covariance  matrix
        df : int or float
            degrees of freedom
        Returns
        -------
        rvs : array, (x.shape[0], )
            each value of the logpdf
        '''
        return logpdf_student(x,mean,cov,df)

class Mixture_t(object):
   
    def rvs(size, means, covs, weights):
        n_comp = len(means)
        S=[]
        index= np.random.choice(range(n_comp), p = weights, size = size)
        n_draw = Counter(index)
        for i in range(n_comp):
            temp = multivariate_t.rvs(mean= means[i],
                                      cov=covs[i],
                                      df=3,
                                      size=n_draw[i])
            S.append(temp.reshape(n_draw[i],means[i].shape[0]))
        return np.array(list(itertools.chain(*S))).reshape((len(index),means[0].shape[0]))
    
    def logpdf(x,means, covs,weights):
         n_comp = len(means)
         lpdf =  np.logaddexp.reduce(
             [np.log(weights[i]) +multivariate_t.logpdf(x,means[i],covs[i]) for i in range(n_comp)],axis = 0)
         return lpdf
        
          

class Mixture_gaussian(object):
   
    def rvs(size, means, covs, weights):
        n_comp = len(means)
        S=[]
        index= np.random.choice(range(n_comp), p = weights, size = size)
        n_draw = Counter(index)
        for i in  n_draw.keys():
            temp = stats.multivariate_normal.rvs(mean= means[i],
                                      cov=covs[i],
                                      size=n_draw[i])
            S.append(temp.reshape(n_draw[i],means[i].shape[0]))
        return np.array(list(itertools.chain(*S))).reshape((len(index),means[0].shape[0]))
    
    def logpdf(x,means, covs,weights):
         n_comp = len(means)
         lpdf =  np.logaddexp.reduce(
             [np.log(weights[i]) +stats.multivariate_normal.logpdf(x,means[i],covs[i]) for i in range(n_comp)],axis = 0)
         return lpdf

        # params = parameters(means = means, variances = variances, proportions = proportions)
        