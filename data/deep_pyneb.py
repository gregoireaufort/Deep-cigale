# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 16:40:12 2020

@author: Gregoire Aufort
"""



import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import joblib
from copy import copy
import pyneb as pn

class deep_continuum(object):
    #Normalised : outputs the ratios with H_alpha
    def __init__(self):
        self.model = joblib.load("/home/aufort/Desktop/cigale-master/pcigale/data/NN_512_cont.joblib")
        self.scalerx = joblib.load("/home/aufort/Desktop/cigale-master/pcigale/data/scaler_continuum.joblib")
        self.scalery = joblib.load("/home/aufort/Desktop/cigale-master/pcigale/data/scaler_continuum_y.joblib")
        self.pyneb_continuum = pn.Continuum()
        
    def pred_to_params(self,pred_NN):
        inverse_scaling = self.scalery.inverse_transform(pred_NN)
        params = pd.DataFrame({'He1': inverse_scaling[:,0],
                               'He2': np.exp(inverse_scaling[:,1]),
                               'THp':inverse_scaling[:,2],
                               'nH':100})
        return params
    def compute_continuum(self,params_nebular,wl):
        # logU=params_nebular["deep_nebular.logU"]
        # geom_factor=params_nebular["deep_nebular.geometrical_factor"]
        # age=params_nebular["deep_nebular.Age"]
        # Log_O_H=params_nebular["deep_nebular.log(O/H)"]
        # log_N_O = params_nebular["deep_nebular.log(N/O)"]
        # HbFrac =params_nebular["deep_nebular.HbFrac"]
        n = len(params_nebular)
        params_scaled = params_nebular.copy()
        params_scaled["deep_nebular.Age"]/=1e6
        if n ==1:
            model_inputs= self.scalerx.transform(params_scaled).reshape(1, -1)
        else :
            model_inputs= self.scalerx.transform(params_scaled)
        pred_params_pyneb = self.model.predict(model_inputs)
        params_pyneb = self.pred_to_params(pred_params_pyneb)
        continuum = [self.pyneb_continuum.get_continuum(tem = params_pyneb['THp'][i],
                                                        den= params_pyneb['nH'][i],
                                                        He1_H = params_pyneb['He1'][i],
                                                        He2_H = params_pyneb['He2'][i],
                                                        wl =wl) for i in range(n)]
        
        return pd.DataFrame(continuum, columns = wl)
    
#wl=np.array([3500, 3600, 3700, 3800, 3900])