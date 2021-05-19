#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 13:06:15 2020

@author: aufort
"""


import numpy as np
from scipy.interpolate import griddata
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import bisect, minimize_scalar
from math import gamma
from scipy.special import xlogy

class theta_params(object):
    def __init__(self, params):
        self.mean = params[0]
        self.variance = params[1]
        self.proportions = params[2]
    
def kl_mvn(m0, S0, m1, S1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.

    - accepts stacks of means, but only one S0 and S1

    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N) 

def gauss(x,y,Sigma,mu):
    mu = np.array(mu)
    X=np.vstack((x,y)).T
    #return  stats.multivariate_normal.pdf(X,mean = mu, cov = Sigma)
    mat_multi=np.dot((X-mu[None,...]).dot(np.linalg.inv(Sigma)),(X-mu[None,...]).T)
    return  np.diag(np.exp(-1*(mat_multi)))

        
def mean_res(TATAMIS):
    return np.average(TATAMIS.total_sample, weights=TATAMIS.final_weights,axis = 0)
def cov_res(TATAMIS):
    return np.diag(np.cov(TATAMIS.total_sample.T, aweights=TATAMIS.final_weights[:,]))

def logpdf_student(x,mean,cov,df=3):
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
        [n,d] = x.shape
        x_m = x-mean
        cov = ((df-2)/df)*cov #the Sigma parameter of a student is not the covariance matrix
        inv_cov = np.linalg.pinv(cov)
        det_cov = np.linalg.det(cov)
        log_num = np.log(gamma((df+d)/2))
        L = np.asarray([np.dot(np.dot(x_m[i,:].T,inv_cov),x_m[i,:]) for i in range(n)])
        log_denom = np.log(gamma(df/2.)) + (d/2.)*(np.log(df) +np.log(np.pi)) + .5*np.log(det_cov) +((df+d)/2)*np.log(1+ (1/df)* L)
        #np.diagonal( np.dot( np.dot(x_m, inv_cov), x_m.T)))
        return log_num-log_denom
        
def plot_contour(x,y,z,alpha, color):
    npts = 10000
    # define grid.
    xi = np.linspace(-4.1, 4.1, 1000)
    yi = np.linspace(-4.1, 4.1, 1000)
    ## grid the data.
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    levels = [0.1,0.3,0.5,0.7,0.9]
    # contour the gridded data, plotting dots at the randomly spaced data points.
    CS = plt.contour(xi,
                     yi,
                     zi,
                     linewidths=0.5,
                     colors=color, 
                     levels=levels, 
                     alpha = alpha)
    #CS = plt.contourf(xi,yi,zi,len(levels),cmap=cm.Greys_r, levels=levels)
    return CS
  

def plot_iteration(MAMIS,target,cluster,prev_target = None, title ="", n = 0):
        """
        plot the contour of estimated mixture at iteration max_iter - n 

        Parameters
        ----------
        n : int, optional
            Number of backward steps from last iteration. The default is 0.

        Returns
        -------
        None
        """        
        max_iter = MAMIS.max_iter
        means = MAMIS.theta_total[max_iter - n].mean
        covariances = MAMIS.theta_total[max_iter - n].variance
        weights_prop = MAMIS.theta_total[max_iter - n].proportions
        n_comp = len(means)
        # make up some randomly distributed data
        npts = 5000
        x = np.random.uniform(-4, 4, npts)
        y = np.random.uniform(-4, 4, npts)
        z=[]
        z = [gauss(x, y, Sigma=covariances[i][0:2,0:2],mu=means[i][0:2]) for i in range(n_comp)]
        plt.figure()
        for i in range(n_comp):
            plot_contour(x=x,y=y,z=z[i],alpha =1, color = 'green')
        if cluster is not None:
            plt.scatter([cluster[i][0] for i in range(len(cluster))],[cluster[i][1] for i in range(len(cluster))],s=1)
        est_mean=np.average(MAMIS.total_sample, weights=MAMIS.final_weights,axis = 0)
        est_var=np.cov(MAMIS.total_sample.T, aweights=MAMIS.final_weights[:,])
        z_est = gauss(x, y, Sigma=est_var[0:2,0:2],mu=est_mean[0:2]) 
        if target is not None:
            z_true =  gauss(x, y, Sigma=target[1][0:2,0:2],mu=target[0][0:2])
            CS3 = plot_contour(x,y,z_true,0.6, color = "red")
        CS2 = plot_contour(x,y,z_est,0.6, color = "blue")
        if prev_target is not None:
            z_prev = gauss(x, y, Sigma=prev_target[1][0:2,0:2],mu=prev_target[0][0:2]) 
            CS1 = plot_contour(x,y,z_prev,0.6, color = "black")
            lines = [CS1.collections[0],CS2.collections[0],CS3.collections[0]]
            plt.legend(handles = lines,labels = ("previous","est","true"))
            
        elif target is not None:
            lines = [CS2.collections[0],CS3.collections[0]]
            plt.legend(handles = lines,labels = ("est","true"))
        else : 
            lines = [CS2.collections[0]]
            plt.legend(handles = lines,labels = ("est"))
        plt.title( title +" \n iter = "+str(max_iter-n))
        plt.show()
        
def plot_convergence(MAMIS,target = None,cluster = None,prev_target = None, title =""):
    for i in range(MAMIS.max_iter,-1,-1):
            plot_iteration(MAMIS, 
                             target= target,
                             cluster=cluster,
                             prev_target = prev_target,
                             n=i,
                             title = title) 
            
