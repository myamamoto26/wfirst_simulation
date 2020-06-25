import numpy as np
import sys, os, io
import math
import logging
import time
import yaml
import copy
import galsim as galsim
import galsim.wfirst as wfirst
#wfirst.pixel_scale /= 4
import galsim.config.process as process
import galsim.des as des
import ngmix
import fitsio as fio
from astropy.time import Time
from astropy.table import Table
from mpi4py import MPI
from ngmix.jacobian import Jacobian
from ngmix.observation import Observation, ObsList, MultiBandObsList,make_kobs
from ngmix.galsimfit import GalsimRunner,GalsimSimple,GalsimTemplateFluxFitter
from ngmix.guessers import R50FluxGuesser
from ngmix.bootstrap import PSFRunner
from ngmix import priors, joint_prior
import mof
import meds

import wfirst_simple_sim
from wfirst_simple_sim import get_exp_list

"""
#---------------------------------------------------------------
## Try to build metacalibration from scratch with simpler codes.|
#---------------------------------------------------------------
## The order of operations
1. Deconvolve the original stamp.
	Obs_list: 
2. Shearing of the image. 

3. Reconvolve it by a new PSF. 

"""

def Observation():
	# Obtain observation list, psf list, weight from wfirst_simple_sim.py
	gals=wfirst_simple_sim.main(1, 'Gaussian', "Gaussian", 'metacal')

	gals=[fio.FITS('mcal_gal_0.fits')[-1].read(), fio.FITS('mcal_gal_1.fits')[-1].read()]
	psfs=[fio.FITS('mcal_psf_0.fits')[-1].read(), fio.FITS('mcal_psf_1.fits')[-1].read()]
	offsets=[galsim.PositionD(-0.5,-0.5), galsim.PositionD(0.23248147187155155,0.3059909382209298)]
	sky_stamp=[fio.FITS('mcal_sky_0.fits')[-1].read(), fio.FITS('mcal_sky_1.fits')[-1].read()]
	obs_list, psf_list, w = get_exp_list(gals, psfs, offsets, sky_stamp, psf2=None)
	print(obs_list, psf_list, w)

Observation()