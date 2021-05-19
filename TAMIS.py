#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 16:53:30 2020

@author: aufort
"""

import types
import numpy as np
import time

import scipy.stats as stats
from GMM import GMM_fit
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
import pandas as pd
import copy
from scipy.interpolate import griddata
from scipy.special import xlogy
from utils import gauss, plot_contour
from scipy.optimize import bisect, minimize_scalar, minimize,NonlinearConstraint

def adapt_beta(weights, alpha):
        """
        Adapt beta by binary search to get a tempered ESS of alpha
        """
        
        def ESS_beta(beta):
            temp_weights = np.exp(beta * weights)
            num = np.sum(temp_weights)**2
            denom = np.sum(temp_weights**2)
            return (num/denom) 
        def ESS_to_solve(beta):
            return ESS_beta(beta) - alpha
       
      
        res = bisect(ESS_to_solve, 1e-5,1,xtol=1e-4)
        return res
       
class TAMIS(object):
    """
    Modified Adaptive Multiple Importance Sampling
    """
    def __init__(self,
                 target,
                 proposal=None,
                 n_comp = 3,
                 init_theta =[[0,0,0,0],200*np.eye(4)],
                 n_sample=10000,
                 ESS_tol = 1000,
                 alpha = 500,
                 adapt = True,
                 verbose= 0):
        """
        Pass initializing data to importance sampler.
        Arguments::
        - proposal: scipy distribution or None. Used for
        importance sampling.
        -target : logpdf of the target distribution
        -init_theta : list or ndarray.r
        """
        self.iteration = 0
        self.target = target
        self.theta =init_theta
        self.dim = target.dim
        self.total_sample = []
        self.total_target = []
        self.n_sample = n_sample
        self.theta_total = [init_theta]
        self.total_proposal= []
        self.ESS = []
        self.ESS_tol = ESS_tol
        self.tempered_ESS = []
        self.betas = []
        self.previous_temp = 0
        self.max_iter = 0
        self.n_comp = n_comp
        self.alpha = alpha
        self.verbose = verbose
        self.adapt = adapt
        self.total_weight = []
        self.KL = []
        if proposal is None:
            self.proposal = stats.multivariate_normal
            self.p=0
        else:
            self.proposal = proposal
            self.p = 1
        
        
    def proposal_rvs(self):
        """
        Samples from the proposal distribution, a multivariate gaussian if
        None is selected
        """
        n= self.n_sample[self.iteration]
        theta = self.theta
        if self.p == 0:
            self.sample = self.proposal.rvs(size = n, mean =theta.mean, cov = theta.variance)
        else :
            self.sample = self.proposal.rvs(size = n, means =theta.mean, covs = theta.variance,weights =theta.proportions)

    def proposal_logpdf(self, x):
        """
        Computes the logpdf of x from the proposal
        """
        theta = self.theta
        if self.p == 0:
            pdf = self.proposal.logpdf(x, mean = theta.mean, cov = theta.variance)
        else :
            pdf = self.proposal.logpdf(x, means = theta.mean, covs = theta.variance,weights = theta.proportions)
        return pdf 
    
    
    def previous_proposal_logpdf(self,x):
        if self.iteration ==0 :
            theta = self.theta
        else :
            theta = self.theta_total[self.iteration-1]
        pdf = self.proposal.logpdf(x, means = theta.mean, covs = theta.variance,weights = theta.proportions)
        return pdf 
    
    
    
    def weight(self):
        """
        Computes the importance sampling weights
        """
        
        sample = self.sample
        log_likelihood_sample = np.array(self.target.log_likelihood(sample = sample))
        log_prior_sample = np.array(self.target.log_prior(sample))
        proposal_sample = self.proposal_logpdf(sample)
        targ_sample = log_prior_sample + log_likelihood_sample
        log_weight = targ_sample - proposal_sample
        unnorm_weight = np.exp(log_weight -np.max(log_weight))
        self.weights = unnorm_weight/np.sum(unnorm_weight)
        # plt.hist(np.log10(self.weights+1e-50), bins = 50)
        # plt.axvline(np.max(np.log10(self.weights +1e-50)), color = 'r')
        # plt.xlabel("log10(weight)")
        # plt.yscale('log')
        # plt.title("untempered_weights")
        # plt.show()
        self.total_weight.append(self.weights)

        ESS = 1/np.sum(self.weights**2)
        self.ESS.append(ESS)
        if ESS <self.alpha and self.adapt:
            beta = adapt_beta(log_weight,
                                alpha = self.alpha)
        else :
            beta = 1        
            
        
        self.previous_temp = beta
        targ_tempered =  beta *( log_prior_sample + log_likelihood_sample)
        log_unnorm_tempered_weights = targ_tempered - (proposal_sample*beta)
        unnorm_tempered_weights = np.exp(log_unnorm_tempered_weights -np.max(log_unnorm_tempered_weights))
        self.tempered_weights = unnorm_tempered_weights/np.sum(unnorm_tempered_weights)
        # plt.hist(np.log10(self.tempered_weights +1e-50), bins = 50)
        # plt.axvline(np.max(np.log10(self.tempered_weights +1e-50)), color = 'r')
        # plt.axvline(np.quantile(np.log10(self.tempered_weights +1e-50),0.6), color = 'black')
        # plt.xticks(list(plt.xticks()[0]) + [np.quantile(np.log10(self.tempered_weights +1e-50),0.6)])
        # plt.xlabel("log10(weight)")
        # plt.yscale('log')
        # plt.title("tempered_weights")
        # plt.show()
        #print(np.quantile(self.tempered_weights,[0.4,0.5,0.6,0.7,0.95,0.99]))
        self.betas.append(beta)
        self.total_target.append(np.vstack(targ_sample))
        
        tempered_ESS = 1/np.sum(self.tempered_weights**2)
        KL= np.sum(xlogy(self.weights,self.weights)) + np.log(len(self.weights))
        self.tempered_ESS.append(tempered_ESS)
        self.KL.append(KL)
    def test_stop(self):
        """
        test ESS or KL
        """
    
        if self.verbose == 1:
            #print("tempered ESS = ",tempered_ESS)
            print("ESS = ",self.ESS[self.iteration])
            print("Kullback-Leibler divergence = ", self.KL[self.iteration] )
            
        return np.sum(self.ESS)>self.ESS_tol
    
    def store_sample(self):
        """
        stores the current sample with the ones from previous iterations
        """
        self.total_sample.append(self.sample)
        
    
    def update_theta(self):
        """
        Updates proposal parameters using sample and TEMPERED weights
        """
        theta = copy.copy(self.theta)
        weights = self.tempered_weights 
        if self.p == 0:
            theta.mean = np.average(self.sample, weights = weights,axis = 0)
            theta.variance = np.cov(self.sample, aweights = weights,rowvar = False ) 
        else :
            temp = GMM_fit(sample = self.sample,
                           weights= weights,
                           n_comp = self.n_comp,
                           init_parameters = theta)
            #assert np.isnan(temp.mean[0][0]) == False
            theta.mean=temp.mean
            theta.variance=temp.variance
            theta.proportions= temp.proportions
        self.theta = theta
        self.theta_total.append(theta)
        
    def iterate(self):
        """
        Makes one iteration of the AMIS scheme without recycling
        """
        self.proposal_rvs()
        self.weight()
        self.update_theta()
        self.store_sample()


    def final_step(self):
        """
        recycling step
        """
        t1=time.time()
        T =len(self.total_sample)
        self.total_sample=np.row_stack(
                [self.total_sample[i] for i in range(T)])
        self.total_target=np.row_stack(
                [self.total_target[i] for i in range(T)])
        N_total = np.sum(self.n_sample[0:T])
        t2=time.time()
        print(t2-t1, "pour la première partie")
        for i in range(T):
            self.theta = self.theta_total[i]
            self.total_proposal.append(self.proposal_logpdf(self.total_sample))
        log_target = self.total_target.reshape((len(self.total_target),))
        num = np.exp(log_target - np.max(log_target)) #Rescaling target
        denom = (1/N_total)* np.sum( np.array(np.exp(self.total_proposal))*np.array(self.n_sample)[0:T,None], axis =0)
        self.final_weights = num/ denom
        
    def extract_params(self,n):
        """
        Extract parameters at iteration max_iter-n, used for transfer
        """
        n = int(n)
        i_max = self.max_iter
        assert n<=i_max, 'n>i_max'
        to_extract = self.theta_total[(i_max-n)]
        return to_extract
    
    def plot_convergence(self,title = False,save = False):
        KL=[]
        for i in range(self.max_iter + 1):
            temp =  [resample(self.total_weight[i]) for j in range(100)]
            KL.append( [np.sum(xlogy(temp[j],temp[j])) + np.log(len(temp[j])) for j in range(100)])
        arr = np.array(KL).T
        df = pd.DataFrame(data = arr).melt()
        df.columns = ["Iteration","Kullback-Leibler divergence"]
        df2 = pd.DataFrame(data = np.array(self.betas)).melt()
        df2.columns = ["Iteration","Beta"]
        df2["Iteration"] = range(self.max_iter+1)
        fig,ax = plt.subplots()
        pl =sns.lineplot(x="Iteration",y="Kullback-Leibler divergence", data = df,ci="sd")
        #pl.set_xticks(range(self.max_iter + 1))
        ax2 = ax.twinx()
        sns.lineplot(x="Iteration",y="Beta", data = df2,ci="sd", color = 'r')
        if title :
            pl.set_title(title)
        if save :
            fig = pl.get_figure()
            fig.savefig(save)
        
    def result(self, T = 10):
        """
        complete MAMIS scheme
        """
        t1 = time.time()
        for i in range(T):
            self.iteration = i
            self.iterate()
            a= self.test_stop()
            if a:
                break
        self.max_iter = i
        print("Arreté apres étape",i+1)
        print("final_step")
        t2=time.time()
        print(t2-t1, "pour les ", i+1,"iterations")
        self.final_step()
        print(time.time()-t2, "pour la dernière étape")
        norm_weights =self.final_weights / np.sum(self.final_weights)
        self.ESS_final = (np.sum(self.final_weights)**2)/np.sum(self.final_weights**2)
        self.KL_final = np.sum(xlogy(norm_weights,norm_weights))   + np.log(len(self.final_weights))
        print(time.time()-t1, "secondes au total")
        return self
        
    
 
       