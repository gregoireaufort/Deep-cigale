# -*- coding: utf-8 -*-
# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Copyright (C) 2014, 2016 Laboratoire d'Astrophysique de Marseille
# Copyright (C) 2014 University of Cambridge
# Copyright (C) 2018 Universidad de Antofagasta
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt

"""
Delayed tau model for star formation history with an optional exponential burst
===============================================================================

This module implements a star formation history (SFH) described as a delayed
rise of the SFR up to a maximum, followed by an exponential decrease. Optionally
a decreasing exponential burst can be added to model a recent episode of star
formation.

"""

from collections import OrderedDict

import numpy as np

from . import SedModule


class SFHDelayed(SedModule):
    """Delayed tau model for Star Formation History with an optionally
    exponential burst.

    This module sets the SED star formation history (SFH) proportional to time,
    with a declining exponential parametrised with a time-scale τ. Optionally
    an exp(-t_/τ_burst) component can be added to model the latest episode of
    star formation.

    """

    parameter_list = OrderedDict([
        ("tau_main", (
            "cigale_list()",
            "e-folding time of the main stellar population model in Myr.",
            2000.
        )),
        ("age_main", (
            "cigale_list(dtype=int, minvalue=0.)",
            "Age of the main stellar population in the galaxy in Myr. The "
            "precision is 1 Myr.",
            5000
        )),
        ("tau_burst", (
            "cigale_list()",
            "e-folding time of the late starburst population model in Myr.",
            50.
        )),
        ("age_burst", (
            "cigale_list(dtype=int, minvalue=1.)",
            "Age of the late burst in Myr. The precision is 1 Myr.",
            20
        )),
        ("f_burst", (
            "cigale_list(minvalue=0., maxvalue=0.9999)",
            "Mass fraction of the late burst population.",
            0.
        )),
        ("sfr_A", (
            "cigale_list(minvalue=0.)",
            "Value of SFR at t = 0 in M_sun/yr.",
            1.
        )),
        ("normalise", (
            "boolean()",
            "Normalise the SFH to produce one solar mass.",
            True
        )),
    ])

    def _init_code(self):
        self.tau_main = float(self.parameters["tau_main"])
        self.age_main = float(self.parameters["age_main"])
        self.tau_burst = float(self.parameters["tau_burst"])
        self.age_burst = float(self.parameters["age_burst"])
        self.f_burst = float(self.parameters["f_burst"])
        sfr_A = float(self.parameters["sfr_A"])
        normalise = bool(self.parameters["normalise"])

    def process(self, sed):
        """
        Parameters
        ----------
        sed : pcigale.sed.SED object

        """

        sed.add_module(self.name, self.parameters)

        # Add the sfh and the output parameters to the SED.
        sed.sfh = None
        sed.add_info("sfh.integrated", 0, True)
        sed.add_info("sfh.age_main", self.age_main)
        sed.add_info("sfh.tau_main", self.tau_main)
        sed.add_info("sfh.age_burst", self.age_burst)
        sed.add_info("sfh.tau_burst", self.tau_burst)
        sed.add_info("sfh.f_burst", self.f_burst)

# SedModule to be returned by get_module
Module = SFHDelayed
