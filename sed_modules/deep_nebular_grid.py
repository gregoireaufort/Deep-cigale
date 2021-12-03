#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 15:03:05 2021

@author: aufort
"""
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy.constants as cst

from pcigale.data import  deep_cloudy #,deep_pyneb
from . import SedModule

class NebularEmission(SedModule):
    """
    Module computing the nebular emission from the ultraviolet to the
    near-infrared. It includes both the nebular lines and the nubular
    continuum (optional). It takes into account the escape fraction and the
    absorption by dust.
    
    Given the number of Lyman continuum photons, we compute the Hβ line
    luminosity. We then compute the other lines using the
    metallicity-dependent templates that provide the ratio between individual
    lines and Hβ. The nebular continuum is scaled directly from the number of
    ionizing photons.
    
    """
    path_data = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/data/'
    params = pd.read_csv(path_data+'deep_nebular_grid_parameters.csv',sep=" ")
    names_deep_neb = ["deep_nebular_grid.logU",
       		           "deep_nebular_grid.geometrical_factor",
       		           "deep_nebular_grid.Age",
       		           "deep_nebular_grid.log_O_H",
       		           "deep_nebular_grid.log_N_O",
       		           "deep_nebular_grid.HbFrac"]
    params_nebular = params[names_deep_neb]
       
    all_lines = deep_cloudy.Deep_cloudy(params_nebular,path_data )#/(4*np.pi*(3.086*10**21)**2) #F = L/ 4*pi*D^2, D = 3.086*10*21 cm, 
    all_lines /= 1e7 #Erg/S to W
    n = params_nebular.shape[0]
    #-------------------- Continuum part, to be added later -------#
    # pyneb = deep_pyneb.deep_continuum()
    # wavelength_cont = np.array([3500, 3600, 3700, 3800, 3900]) #To choose, not necessarily in a file, in Angstrom, cf PyNeb doc
    # cont_unscaled = pyneb.compute_continuum(params_nebular,wavelength_cont)
    # #Need to scale cont by Hbeta in lines
    # cont = cont_unscaled.multiply(all_lines['H__1_486133A'],axis = 0)
    # del cont_unscaled
    #-------------------------------------------------------------#
    path_wave =path_data + 'dict_wavelength_lines.csv'
    df_wavelength_lines =  pd.read_csv(path_wave) #depends on the Cloudy training set
    lines = all_lines[df_wavelength_lines['name']]
    wavelength_lines = df_wavelength_lines['wavelength']
    datadb_pyneb = dict()
    datadb_cloudy = dict()
    for i in range(n):
        datadb_cloudy[tuple(np.around(params_nebular.iloc[i,:],4))] = {'lumin' : lines.iloc[i,:],
    	                                                     'names' : lines.columns,
    	                                                     'wave' : wavelength_lines}
        # datadb_pyneb[tuple(np.around(params_nebular.iloc[i,:],4))] = {'lumin' : cont.iloc[i,:],
        #                                                          'wave' : wavelength_cont/10}
    parameter_list = OrderedDict([
        ('logU', (
            'cigale_list(minvalue=-4., maxvalue=-1 )',
            "Ionisation parameter",
            -2.
        )),
        ('geometrical_factor',(
            'cigale_list(minvalue = 0.03, maxvalue=3)',
            "geom_factor",
            1.
        )),
        ('Age',(
            'cigale_list(minvalue = 1, maxvalue=6)',
            "Age",
            3.
        )),
        ('log_O_H',(
            'cigale_list(minvalue = -5.4, maxvalue=-2.8)',
            "log(O/H)",
            -3.
        )),
        ('log_N_O',(
            'cigale_list(minvalue = -2., maxvalue=0.)',
            "log(N/O)",
            -1.
        )),
        ('HbFrac',(
            'cigale_list(minvalue = 0., maxvalue=1.)',
            "Nebula cut",
            1.
        )),
        ('f_esc', (
            'cigale_list(minvalue=0., maxvalue=1.)',
            "Fraction of Lyman continuum photons escaping the galaxy",
            0.
        )),
        ('f_dust', (
            'cigale_list(minvalue=0., maxvalue=1.)',
            "Fraction of Lyman continuum photons absorbed by dust",
            0.
        )),
        ('lines_width', (
            'cigale_list(minvalue=0.)',
            "Line width in km/s",
            300.
        )),
        ('emission', (
            'boolean()',
            "Include nebular emission.",
            True
        ))
    ])

    def _init_code(self):
        """Get the nebular emission lines out of the database and resample
           them to see the line profile. Compute scaling coefficients.
        """
        self.logU = float(self.parameters['logU'])
        self.geometrical_factor = float(self.parameters['geometrical_factor'])
        self.Age = float(self.parameters['Age'])
        self.log_O_H= float(self.parameters['log_O_H'])
        self.log_N_O= float(self.parameters['log_N_O'])
        self.HbFrac = float(self.parameters['HbFrac'])
        self.fesc = float(self.parameters['f_esc'])
        self.fdust = float(self.parameters['f_dust'])
        self.lines_width = float(self.parameters['lines_width'])
        self.emission = bool(self.parameters["emission"])

        if self.fesc < 0. or self.fesc > 1:
            raise Exception("Escape fraction must be between 0 and 1")

        if self.fdust < 0 or self.fdust > 1:
            raise Exception("Fraction of lyman photons absorbed by dust must "
                            "be between 0 and 1")

        if self.fesc + self.fdust > 1:
            raise Exception("Escape fraction+f_dust>1")

        if self.emission:
            # To take into acount the escape fraction and the fraction of Lyman
            # continuum photons absorbed by dust we correct by a factor
            # k=(1-fesc-fdust)/(1+(α1/αβ)*(fesc+fdust))
            alpha_B = 2.58e-19  # Ferland 1980, m³ s¯¹
            alpha_1 = 1.54e-19  # αA-αB, Ferland 1980, m³ s¯¹
            k = (1. - self.fesc - self.fdust) / (1. + alpha_1 / alpha_B * (
                self.fesc + self.fdust))

            self.corr = k
        self.idx_Ly_break = None
        self.absorbed_old = None
        self.absorbed_young = None

    def process(self, sed):
        """Add the nebular emission lines

        Parameters
        ----------
        sed: pcigale.sed.SED object
        parameters: dictionary containing the parameters

        """
        if self.idx_Ly_break is None:
            self.idx_Ly_break = np.searchsorted(sed.wavelength_grid, 91.2)
            self.absorbed_old = np.zeros(sed.wavelength_grid.size)
            self.absorbed_young = np.zeros(sed.wavelength_grid.size)

        self.absorbed_old[:self.idx_Ly_break] = -(
            sed.luminosities['stellar.old'][:self.idx_Ly_break] *
            (1. - self.fesc))
        self.absorbed_young[:self.idx_Ly_break] = -(
            sed.luminosities['stellar.young'][:self.idx_Ly_break] *
            (1. - self.fesc))

        sed.add_module(self.name, self.parameters)
        sed.add_info('nebular.f_esc', self.fesc)
        sed.add_info('nebular.f_dust', self.fdust)
        sed.add_info('dust.luminosity', (sed.info['stellar.lum_ly_young'] +
                     sed.info['stellar.lum_ly_old']) * self.fdust, True)

        sed.add_contribution('nebular.absorption_old', sed.wavelength_grid,
                             self.absorbed_old)
        sed.add_contribution('nebular.absorption_young', sed.wavelength_grid,
                             self.absorbed_young)

        if self.emission:
            NLy_old = sed.info['stellar.n_ly_old']
            NLy_young = sed.info['stellar.n_ly_young']
            params_NN = [self.logU,
                        self.geometrical_factor,
                        self.Age,
                        self.log_O_H,
                        self.log_N_O,
                        self.HbFrac]
            index = tuple(np.around(params_NN,4))
            
            self.lines = self.datadb_cloudy[index]
            #cont= self.datadb_pyneb[index]

            linesdict = dict(zip(self.lines["names"],
                                          zip(self.lines["wave"],
                                              self.lines["lumin"])))

            sed.add_info('nebular.lines_width', self.lines_width)
            sed.add_info('nebular.logU', self.logU)
            sed.add_info('nebular.geometrical_factor', self.geometrical_factor)
            sed.add_info('nebular.Age', self.Age)
            sed.add_info('nebular.log_O_H', self.log_O_H)
            sed.add_info('nebular.log_N_O', self.log_N_O)
            sed.add_info('nebular.HbFrac', self.HbFrac)
            
            new_wave = np.array([])
            for line_wave in self.lines["wave"]:
                width = line_wave * self.lines_width * 1e3 / cst.c
                new_wave = np.concatenate((new_wave,
                                        np.linspace(line_wave - 3. * width,
                                                    line_wave + 3. * width,
                                                    9)))
            new_wave.sort()
            new_flux = np.zeros_like(new_wave)
            for line_flux, line_wave in zip(self.lines["lumin"],self.lines["wave"]):
                width = line_wave * self.lines_width * 1e3 / cst.c
                new_flux += (line_flux * np.exp(- 4. * np.log(2.) *
                            (new_wave - line_wave) ** 2. / (width * width)) /
                            (width * np.sqrt(np.pi / np.log(2.)) / 2.))
            self.lines["wave"] = new_wave
            self.lines["lumin"] = new_flux

            for line in self.lines["names"]:
                wave, lumin = linesdict[line]
                sed.lines[line] = (wave,
                                   lumin * NLy_old * self.corr,
                                   lumin * NLy_young * self.corr)

            sed.add_contribution('nebular.lines_old', self.lines["wave"],
                                 self.lines["lumin"] * NLy_old * self.corr)
            sed.add_contribution('nebular.lines_young', self.lines["wave"],
                                 self.lines["lumin"] * NLy_young * self.corr)

            # sed.add_contribution('nebular.continuum_old', cont["wave"],
            #                      cont["lumin"] * NLy_old * self.corr)
            # sed.add_contribution('nebular.continuum_young', cont["wave"],
            #                      cont["lumin"] * NLy_young * self.corr)


# SedModule to be returned by get_module
Module = NebularEmission
