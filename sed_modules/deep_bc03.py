# -*- coding: utf-8 -*-
# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly

"""
Bruzual and Charlot (2003) stellar emission module
==================================================

This module implements the Bruzual and Charlot (2003) Single Stellar
Populations.

"""

from collections import OrderedDict
import os
import numpy as np
import tensorflow as tf
from pandas import read_csv, concat, DataFrame
from sklearn.preprocessing import LabelEncoder
from . import SedModule
from ..data import Database

class BC03(SedModule):
    """Bruzual and Charlot (2003) stellar emission module

    This SED creation module convolves the SED star formation history with a
    Bruzual and Charlot (2003) single stellar population to add a stellar
    component to the SED.

    """
    
    params = read_csv("/home/aufort/Desktop/cigale-master/params_comparison.txt",sep=" ")
    path_data = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/data/'
    model = tf.keras.models.load_model(path_data+'ANN/NN_lumins_bayes_2.h5')
    scaling_params = np.load(path_data +'X_scaling_lumin.npy')
    mean_X, sd_X = scaling_params[:,0], scaling_params[:,1]
    scaling_spec = np.load(path_data+'Y_scaling_lumins.npy')
    mean_Y, sd_Y, mins_Y = scaling_spec[:,0], scaling_spec[:,1], scaling_spec[:,2]
    
    test_nn = params[params.columns[0:5]]
    
    labelencoder = LabelEncoder()
    labelencoder.classes_ = np.load(path_data + 'classes_metallicity.npy')
    met_enc = labelencoder.transform(params['deep_bc03.metallicity'])
    mat_params = concat([test_nn,DataFrame(met_enc)], axis = 1).values
    
    param_norm = (mat_params-mean_X)/sd_X
    pred_NN = model.predict(param_norm)
    rescaled = np.exp(((pred_NN+mins_Y)*sd_Y) + mean_Y)
    n = mat_params.shape[0]
    mat_params[:,5] = params['deep_bc03.metallicity']
    datadb = dict()
    for i in range(n):
        datadb[tuple(np.around(mat_params[i,:],2))] = {'spec_young' : rescaled[i,0:6941],
    						  'spec_old' : rescaled[i,6941:13882],
    						  'n_ly' : rescaled[i,-1] }

    del params, test_nn, scaling_spec, mat_params, pred_NN
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
        wave =np.load(self.path_data +'wavelengths.npy')

        # We do similarly for the total stellar luminosity
        params_NN = [sed.info["sfh.tau_main"],sed.info["sfh.age_main"],sed.info["sfh.tau_burst"],sed.info["sfh.age_burst"],
                           sed.info["sfh.f_burst"],self.parameters["metallicity"]]
        index = tuple(np.around(params_NN,2))
        imp = self.datadb[index]
        w = np.where(wave <= 91.1)
        

        spec_young, spec_old, n_ly = imp['spec_old'],imp['spec_young'], imp['n_ly']
        lum_lyc_young, lum_lyc_old = np.trapz([spec_young[w], spec_old[w]],wave[w])
        lum_young, lum_old = np.trapz([spec_young, spec_old], wave)
        #r1 = 10 *np.random.random()
        #r2 = 10* np.random.random() 
        #spec_young, spec_old, n_ly = r1*np.ones(wave.shape,float),r2*np.ones(wave.shape,float),r2*1e44
        sed.add_module(self.name, self.parameters)

        sed.add_info("stellar.imf", self.imf)
        sed.add_info("stellar.metallicity", self.metallicity)
        sed.add_info("stellar.old_young_separation_age", self.separation_age)

        sed.add_info("stellar.m_star_young",0, True)
        sed.add_info("stellar.m_gas_young", 0, True)
        sed.add_info("stellar.n_ly_young", n_ly/2, True)
        sed.add_info("stellar.lum_ly_young", lum_lyc_young, True)
        sed.add_info("stellar.lum_young", lum_young, True)

        sed.add_info("stellar.m_star_old", 0, True)
        sed.add_info("stellar.m_gas_old", 0, True)
        sed.add_info("stellar.n_ly_old", n_ly/2, True)
        sed.add_info("stellar.lum_ly_old", lum_lyc_old, True)
        sed.add_info("stellar.lum_old", lum_old, True)

        sed.add_info("stellar.m_star", 0, True)
        sed.add_info("stellar.m_gas", 0, True)
        sed.add_info("stellar.n_ly", n_ly, True)
        sed.add_info("stellar.lum_ly", lum_lyc_young + lum_lyc_old, True)
        sed.add_info("stellar.lum", lum_young + lum_old, True)
        sed.add_info("stellar.age_m_star", 0, False)

        sed.add_contribution("stellar.old", wave, spec_old)
        sed.add_contribution("stellar.young", wave, spec_young)


# SedModule to be returned by get_module
Module = BC03
