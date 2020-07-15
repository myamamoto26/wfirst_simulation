from __future__ import division
from __future__ import print_function

from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from past.builtins import basestring
from builtins import object
from past.utils import old_div
import numpy as np
#import healpy as hp
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
#import pickle as pickle
#import pickletools
from astropy.time import Time
from astropy.table import Table
from mpi4py import MPI
#from mpi_pool import MPIPool
#import cProfile, pstats
#import glob
#import shutil
from ngmix.jacobian import Jacobian
from ngmix.observation import Observation, ObsList, MultiBandObsList,make_kobs
from ngmix.galsimfit import GalsimRunner,GalsimSimple,GalsimTemplateFluxFitter
from ngmix.guessers import R50FluxGuesser
from ngmix.bootstrap import PSFRunner
from ngmix import priors, joint_prior
import mof
import meds
#import psc

import matplotlib
matplotlib.use ('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pylab
from scipy.interpolate import interp1d

from scipy.optimize import curve_fit

## import functions from other files
from selection_effects_analysis import residual_bias, residual_bias_correction
from roman_imsim import Pointing, Model, Image


def rotate_psf():
	cat = fio.FITS('truth_mag.fits')[-1].read()
	SCA = 1
	filter_ = 'H158'
	bpass = wfirst.getBandpasses(AB_zeropoint=True)[filter_]
	hlr = 1.0
	stamp_size = 32
	selected_dithers = [22535, 22535]
	exposures = [0, 20]
	i_gal = 1
	real_wcs = True
	basis = 'g1'
	rng = galsim.BaseDeviate(314)

	# get star/psf stamp
	gal_model = None
	st_model = None

	profile = Model(cat, 'Gaussian', 'wfirst', SCA, filter_, bpass, hlr, i_gal)
	gal_model, g1, g2 = profile.draw_galaxy(basis)
	st_model = profile.draw_star()
	print('Created galaxy anf psf models.')

	sca_center = Pointing(selected_dithers[0], SCA, filter_, stamp_size, exposures[0], random_angle=False).find_sca_center()
	psfs = []
	for exp in range(2): 
	    gal_stamp=None
	    psf_stamp=None

	    pointing=Pointing(selected_dithers[exp], SCA, filter_, stamp_size, exposures[exp], random_angle=False)
	    image=Image(i_gal, stamp_size, gal_model, st_model, pointing, sca_center, real_wcs)

	    gal_stamp, psf_stamp, offset = image.draw_image(gal_model, st_model)
	    gal_stamp, sky_stamp = image.add_noise(rng, gal_stamp)
	    if real_wcs==True:
	        gal_stamp, psf_stamp = image.wcs_approx(gal_stamp, psf_stamp)
	    print('Drawing images on the stamps.')

	    ## rotate psf to average out some psf shape biases.


	    psfs.append(psf_stamp)

	for j in range(len(psfs)):
		psfs[j].save('rotate_psf_'+str(j)+'.fits')


rotate_psf()













