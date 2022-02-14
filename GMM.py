# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 12:10:25 2020

@author: Greg
"""

import itertools
#from utils import logpdf_student
from collections import Counter
import numpy as np
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
from pomegranate import IndependentComponentsDistribution,NormalDistribution, GeneralMixtureModel
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
def E_step(sample,n_comp, means,variances, weights_comp):
    """
    outputs the logprobas of each obs to be in each cluster
    """
    num =  [np.log(weights_comp[i])+  #### ???????? Wasn't in log but should ? 
             stats.multivariate_normal.logpdf(sample,
                                              means[i],
                                              variances[i],
                                              allow_singular = True) for i in range(n_comp)]
    denom = np.logaddexp.reduce(num,axis = 0)
    
    return num-denom

def M_step(sample, probs,weights, n_comp):
    """
    updates the parameters of each components givent the weighted sample and
    the probs computed in step E
    """
    eps = 1e-6
    means = np.zeros((sample.shape[1],n_comp))
    covar = []
    coeffs = np.zeros((sample.shape[0],n_comp))
    new_weights_comp = np.zeros(n_comp)
    sample = resampling_data(sample, weights, sampling_type="random")
    for i in range(n_comp):
        coeffs[:,i] = np.exp(probs[i])
        means[:,i] = np.average(sample,
                                weights = coeffs[:,i],
                                axis =0)
        variances = [np.cov(sample[:,j],
                             aweights = coeffs[:,i]) + eps for j in range(sample.shape[1])]
        covar.append(np.diag(variances))
        new_weights_comp[i] = np.mean(np.exp(probs[i]))
    return means.T,covar, new_weights_comp

    
def diagonal_EM_homemade(sample,weights, n_comp, init_parameters):
    """
    With initial parameters init_parameters, weighted sample and n_comp, computes
    EM and returns params. Number of steps fixed
    """
    means = init_parameters.mean
    variances = init_parameters.variance
    weights_comp = init_parameters.proportions
    for i in range(3):
        probs = E_step(sample,n_comp, means,variances, weights_comp)
        means,variances,weights_comp = M_step(sample, probs,weights,n_comp) 
    params = parameters(means,variances,weights_comp)
    return params




class parameters(object):
    """
    class to have identical attributes between pomegranate and Mixmod outputs
    """
    def __init__(self,means,variances, proportions):
        self.mean  = means
        self.variance = variances
        self.proportions = proportions
            
def extract_params_pome(model,n_dim,n_comp):
    """
    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    n_dim : TYPE
        DESCRIPTION.
    n_comp : TYPE
        DESCRIPTION.

    Returns
    -------
    params : TYPE
        DESCRIPTION.

    """
    means= [0]*n_comp
    variances = [0]*n_comp
    for k in range(n_comp) :
        comp_params = model.distributions[k].parameters[0]
        means[k] = np.array([comp_params[i].parameters[0] for i in range(n_dim)])
        variances[k] = np.diag([comp_params[i].parameters[1] for i in range(n_dim)])
    proportions = np.exp(model.weights)
    params = parameters(means = means, 
                        variances = variances, 
                        proportions = proportions)
    return params
def resampling_data(data,weights,sampling_type ="random"):
    N = len(data)
    if sampling_type == "random":
        norm_weights =  weights/np.sum(weights)
        sample_index = np.random.choice(range(N), 
                                        size=N,
                                        replace=True,
                                        p = norm_weights)
        sample = data[sample_index]
    else : 
        sample =np.repeat(data,weights, axis = 0)
    return sample

@ignore_warnings(category = ConvergenceWarning)
def sklearn_diagonal_mixture_with_init(sample,
                                        weights_int,
                                        n_comp, 
                                        init_parameters,
                                        sampling_type = "random"):
    data = resampling_data(sample, weights_int, sampling_type=sampling_type)
    init_means =  [init_parameters.mean[i] for i in range(n_comp)]
    # if len(init_parameters.variance[0].shape) == 1:
    #     init_precisions= [1/init_parameters.variance[i] for i in range(n_comp)]
    # else :
    #     init_precisions= [np.linalg.pinv(init_parameters.variance[i]) for i in range(n_comp)]
        
    clf = GaussianMixture(n_components = n_comp, 
                          covariance_type ="diag", 
                          means_init = init_means, 
                          #precisions_init =init_precisions,
                          max_iter = 10)
    clf.fit(data)
    params = parameters(means = clf.means_,
                        variances = clf.covariances_,
                        proportions = clf.weights_)
    
    return params
def pome_diagonal_mixture_with_init(sample,
                                    weights_int,
                                    n_comp, 
                                    init_parameters):
    """
    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    n_dim : TYPE
        DESCRIPTION.
    n_comp : TYPE
        DESCRIPTION.

    Returns
    -------
    params : TYPE
        DESCRIPTION.

    """
    Distribs = []
    M = sample.shape[1]
    for i in range(n_comp):
        mean_comp = init_parameters.mean[i]
        var_comp = np.diag(init_parameters.variance[i])
        dists = [NormalDistribution(mean_comp[j],var_comp[j])for j in range(M)]
        ind_compt_dist = IndependentComponentsDistribution(dists)
        Distribs.append(ind_compt_dist)
    model = GeneralMixtureModel(Distribs,weights = init_parameters.proportions)
    res = model.fit(X=sample, weights  = weights_int, n_jobs = 1)
    params = extract_params_pome(res, M, n_comp)
    return params


def pome_diagonal_mixture_wo_init(sample,weights_int, n_comp):
    """
    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    n_dim : TYPE
        DESCRIPTION.
    n_comp : TYPE
        DESCRIPTION.

    Returns
    -------
    params : TYPE
        DESCRIPTION.

    """
    model = GeneralMixtureModel.from_samples(distributions = NormalDistribution,
                                                    n_components = n_comp,
                                                    X = sample,
                                                    weights =weights_int,
                                                    n_jobs = 1,
                                                    n_init = 3)
    M = sample.shape[1]
    params = extract_params_pome(model, M, n_comp)
    return params

def modify_weights(weights,isint,trunc):
    """
    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    n_dim : TYPE
        DESCRIPTION.
    n_comp : TYPE
        DESCRIPTION.

    Returns
    -------
    params : TYPE
        DESCRIPTION.

    """
    trunc =  np.quantile(weights,trunc)
    weights2  = weights.copy()
    weights2[weights<trunc] = trunc
    if isint == True:
        weights_int = np.int64((1/trunc)*weights2) + 1
        return weights_int
    return weights2



def pome_diagonal_mixture(sample,weights, n_comp, init_parameters):
    """
    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    n_dim : TYPE
        DESCRIPTION.
    n_comp : TYPE
        DESCRIPTION.

    Returns
    -------
    params : TYPE
        DESCRIPTION.

    """
    
    try:
        params = pome_diagonal_mixture_with_init(sample,
                                                 weights,
                                                 n_comp,
                                                 init_parameters)
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


def GMM_fit(sample, 
            weights,
            tau = 0.6,
            n_comp = 3, 
            init_parameters = None,
            EM_solver = "homemade",
            integer_weights = False):
    weights = modify_weights(weights,integer_weights,tau)
    
    if EM_solver =="pomegranate" :
        params = pome_diagonal_mixture(sample, weights, n_comp, init_parameters)
    elif EM_solver == "sklearn":
        if integer_weights == True : 
            sampling_type = "deterministic"
        else : 
            sampling_type = "random"
        params = sklearn_diagonal_mixture_with_init(sample,
                                        weights,
                                        n_comp, 
                                        init_parameters,
                                        sampling_type)
    elif EM_solver =="homemade":
        params = diagonal_EM_homemade(sample,weights,n_comp,init_parameters)
    else :
        assert ValueError("unknown solver")

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
   
    def rvs(size,  theta):
        means = theta.mean
        covs = theta.variance
        weights = theta.proportions
        
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
    
    def logpdf(x, theta):
         means = theta.mean
         covs = theta.variance
         weights = theta.proportions

         n_comp = len(means)
         lpdf =  np.logaddexp.reduce(
             [np.log(weights[i]) +multivariate_t.logpdf(x,means[i],covs[i]) for i in range(n_comp)],axis = 0)
         return lpdf
        
          

class Mixture_gaussian(object):
   
    def rvs(size, theta):
        means = theta.mean
        covs = theta.variance
        weights = theta.proportions
        
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
    
    def logpdf(x, theta):
         means = theta.mean
         covs = theta.variance
         weights = theta.proportions
         
         n_comp = len(means)
         a=[np.log(weights[i]) + stats.multivariate_normal.logpdf(x,means[i],covs[i], allow_singular = True) for i in range(n_comp)]
         lpdf =  np.logaddexp.reduce(a,axis = 0)
         return lpdf


class Mixture_gaussian_discrete(object):
   
    def rvs(size, theta):
        means = theta.mean
        covs = theta.variance
        weights = theta.proportions
        discrete_ws = theta.disc_probs
        n_comp = len(means)
        S=[]
        index= np.random.choice(range(n_comp), p = weights, size = size)
        n_draw = Counter(index)
        for i in  n_draw.keys():
            temp = stats.multivariate_normal.rvs(mean= means[i],
                                      cov=covs[i],
                                      size=n_draw[i])
            S.append(temp.reshape(n_draw[i],means[i].shape[0]))
        cont = np.array(list(itertools.chain(*S))).reshape((len(index),means[0].shape[0]))
        discrete = rvs_discrete(size,discrete_ws) 
        return np.concatenate((cont,discrete),axis = 1)
    
    def logpdf(x, theta):
         means = theta.mean
         covs = theta.variance
         weights = theta.proportions
         discrete_ws = theta.disc_probs
         n_comp = len(means)
         n_params_disc = len(discrete_ws)
         n_params_cont = x.shape[1] - n_params_disc
         a=[np.log(weights[i]) + stats.multivariate_normal.logpdf(x[:,:n_params_cont],means[i],covs[i], allow_singular = True) for i in range(n_comp)]
         cont =  np.logaddexp.reduce(a,axis = 0)
         discrete = lpdf_discrete(x[:,n_params_cont:],discrete_ws)
         lpdf = cont+discrete
         return lpdf


def lpdf_discrete(x,probs):
    n = x.shape[0]
    lpdf = [np.log(np.prod([probs[i][int(x[j,i])] for i in range(len(probs))])) for j in range(n)]
    return np.array(lpdf)

def rvs_discrete(n,probs):
    discrete = [np.random.choice(range(len(probs[i])),
                                     size= n,
                                     replace = True, 
                                     p = probs[i]/np.sum(probs[i]))
        for i in range(len(probs))]
    return np.array(discrete).T


def update_discrete_probs(sample,weights):
    """
    Assumes the weights are in normalized, and returns probs
    """
    probs = []
    for i in range(sample.shape[1]):
        n_values = len(np.unique(sample[:,i]))
        P = []
        for value in range(n_values):
            P.append(np.sum(weights[sample[:,i] == value]) + (1/(10*n_values)))
        probs.append(P)
    return probs
