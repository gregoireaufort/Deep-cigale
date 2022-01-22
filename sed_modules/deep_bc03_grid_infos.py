# -*- coding: utf-8 -*-
# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author:G.Aufort & Yannick Roehlly

"""
Bruzual and Charlot (2003) stellar emission module
==================================================

This module implements the Bruzual and Charlot (2003) Single Stellar
Populations.

"""

from collections import OrderedDict

import numpy as np
import os
import tensorflow as tf
from joblib import load
from pandas import read_csv, concat, DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from . import SedModule
from ..data import SimpleDatabase as Database


def deep_approx_BC03():
    try:
        
        path_data = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/data/'
        params_sfh = read_csv(path_data+'deep_sfhdelayed_parameters.csv',sep=" ")
        params_bc03 = read_csv(path_data+'deep_bc03_grid_infos_parameters.csv',sep=" ")
        params = params_sfh.merge(params_bc03, how = 'cross')
        model = tf.keras.models.load_model(path_data+'ANN/NN_pca_norm.h5')
        scaling_params = np.load( path_data+'X_scaling_lumin.npy')
        mean_X, sd_X = scaling_params[:,0], scaling_params[:,1]
        scaling_spec = np.load( path_data+'Y_scaling_lumins.npy')
        mean_Y, sd_Y, mins_Y = scaling_spec[:,0], scaling_spec[:,1], scaling_spec[:,2]
        pca_fit = load( path_data+'pca_fit_norm.joblib') 
    
        test_nn = params[["deep_sfhdelayed.tau_main",
                         "deep_sfhdelayed.age_main",
                         "deep_sfhdelayed.tau_burst",
                         "deep_sfhdelayed.age_burst",
                         "deep_sfhdelayed.f_burst"]]
        
        infos_table = load_infos(params['deep_bc03_grid_infos.metallicity'])
        infos = cut_infos(infos_table,
                          params['deep_bc03_grid_infos.metallicity'],
                          params["deep_sfhdelayed.age_main"])
        labelencoder = LabelEncoder()
        labelencoder.classes_ = np.load(path_data +'classes_metallicity.npy')
        met_enc = labelencoder.transform(params['deep_bc03_grid_infos.metallicity'])
        mat_params = concat([test_nn,params['deep_bc03_grid_infos.metallicity']], axis = 1).values
        
        param_norm = (mat_params-mean_X)/sd_X
        pred_NN = model.predict(param_norm)
        pred_NN_inv = pca_fit.inverse_transform(pred_NN)
        rescaled = np.exp(((pred_NN_inv)*sd_Y) + mean_Y)
        n = mat_params.shape[0]
        mat_params[:,5] = params['deep_bc03_grid_infos.metallicity']
        datadb = dict()
        for i in range(n):
            datadb[tuple(np.around(mat_params[i,:],2))] = {'spec_young' : rescaled[i,0:6941],
        						  'spec_old' : rescaled[i,6941:13882],
        						  'n_ly' : rescaled[i,-1],
                                  'infos' : infos[i]} 
            
        return datadb
    except : 
        pass
    
def cut_infos(infos,metallicities, age):
    info = [infos[metallicities[i]][age[i]] for i in range(metallicities.size)]
    return info


def load_infos(metallicities):
    Z = np.unique(metallicities)
    with Database("bc03") as db:
        infos=[db.get(imf='chab', Z=z).info for z in Z]
    return dict(zip(Z,infos))
class BC03(SedModule):
    """Bruzual and Charlot (2003) stellar emission module

    This SED creation module convolves the SED star formation history with a
    Bruzual and Charlot (2003) single stellar population to add a stellar
    component to the SED.

    """
    parameter_list = OrderedDict([
        ("imf", (
            "cigale_list(dtype=int, options=0. & 1.)",
            "Initial mass function: 0 (Salpeter) or 1 (Chabrier).",
            0
        )),
        ("metallicity", (
            "cigale_list(options=0.0001 & 0.0004 & 0.004 & 0.008 & 0.02 & "
            "0.05)",
            "Metalicity. Possible values are: 0.0001, 0.0004, 0.004, 0.008, "
            "0.02, 0.05.",
            0.02
        )),
        ("separation_age", (
            "cigale_list(dtype=int, minvalue=0)",
            "Age [Myr] of the separation between the young and the old star "
            "populations. The default value in 10^7 years (10 Myr). Set "
            "to 0 not to differentiate ages (only an old population).",
            10
        ))
    ])
    
    datadb = deep_approx_BC03()
    path_data = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/data/'
    
    print("Reloaded")
    def _init_code(self):
        """only reads parameters"""
        self.imf = int(self.parameters["imf"])
        self.metallicity = float(self.parameters["metallicity"])
        self.separation_age = int(self.parameters["separation_age"])
        
        
    def process(self, sed):
        """Add the convolution of a Bruzual and Charlot SSP to the SED

        Parameters
        ----------
        sed: pcigale.sed.SED
            SED object.

        """

        # We compute the Lyman continuum luminosity as it is important to
        # compute the energy absorbed by the dust before ionising gas.
        wave =np.load(self.path_data+'wavelengths.npy')
        
        # We do similarly for the total stellar luminosity
        params_NN = [sed.info["sfh.tau_main"],sed.info["sfh.age_main"],sed.info["sfh.tau_burst"],sed.info["sfh.age_burst"],
                           sed.info["sfh.f_burst"],self.parameters["metallicity"]]
        index = tuple(np.around(params_NN,2))
        imp = self.datadb[index]
        w = np.where(wave <= 91.1)
        sfh = sed.sfh

        spec_young, spec_old = imp['spec_old'],imp['spec_young']
        info = imp.infos
        info_young = 1e6 * np.dot(info[:, :self.separation_age],
                          sfh[-self.separation_age:][::-1])

        info_old = 1e6 * np.dot(info[:, self.separation_age:],
                                sfh[:-self.separation_age][::-1])
        
        info_all = info_young + info_old
        
        info_young = dict(zip(["m_star", "m_gas", "n_ly"], info_young))
        info_old = dict(zip(["m_star", "m_gas", "n_ly"], info_old))
        info_all = dict(zip(["m_star", "m_gas", "n_ly"], info_all))
        T = np.array(range(sfh.size))+1
        info_all['age_mass'] = np.average(T, weights=info[0, :] * sfh[::-1])
        lum_lyc_young, lum_lyc_old = np.trapz([spec_young[w], spec_old[w]],wave[w])
        lum_young, lum_old = np.trapz([spec_young, spec_old], wave)
        
        sed.add_module(self.name, self.parameters)

        sed.add_info("stellar.imf", self.imf)
        sed.add_info("stellar.metallicity", self.metallicity)
        sed.add_info("stellar.old_young_separation_age", self.separation_age)

        sed.add_info("stellar.m_star_young",info_young["m_star"], True)
        sed.add_info("stellar.m_gas_young", info_young["m_gas"], True)
        sed.add_info("stellar.n_ly_young", info_young["n_ly"], True)
        sed.add_info("stellar.lum_ly_young", lum_lyc_young, True)
        sed.add_info("stellar.lum_young", lum_young, True)

        sed.add_info("stellar.m_star_old", info_old["m_star"], True)
        sed.add_info("stellar.m_gas_old", info_old["m_gas"], True)
        sed.add_info("stellar.n_ly_old", info_old["n_ly"], True)
        sed.add_info("stellar.lum_ly_old", lum_lyc_old, True)
        sed.add_info("stellar.lum_old", lum_old, True)

        sed.add_info("stellar.m_star", info_all["m_star"], True)
        sed.add_info("stellar.m_gas", info_all["m_gas"], True)
        sed.add_info("stellar.n_ly", info_all["n_ly"], True)
        sed.add_info("stellar.lum_ly", lum_lyc_young + lum_lyc_old, True)
        sed.add_info("stellar.lum", lum_young + lum_old, True)
        sed.add_info("stellar.age_m_star", info_all["age_mass"], False)

        sed.add_contribution("stellar.old", wave, spec_old)
        sed.add_contribution("stellar.young", wave, spec_young)



# SedModule to be returned by get_module
Module = BC03
