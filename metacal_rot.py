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

def get_exp_list(gals_array, psfs_array, offsets, skys_array, gal_true, gal_jacob, psf2=None):
    #def get_exp_list(gal, psf, sky_stamp, psf2=None):

    if psf2 is None:
        psf2 = psfs_array

    obs_list=ObsList()
    psf_list=ObsList()

    w = []
    for i in range(len(gal)):
        im = gals_array[i]
        im_psf = psfs_array[i]
        im_psf2 = psf2[i]
        weight = 1/skys_array[i]

        jacob = gal_jacob[i]
        dx = offsets[i].x
        dy = offsets[i].y
        
        gal_jacob = Jacobian(
            row=gal_true[i].y+dy,
            col=gal_true[i].x+dx,
            dvdrow=jacob.dvdy,
            dvdcol=jacob.dvdx,
            dudrow=jacob.dudy,
            dudcol=jacob.dudx)
        psf_jacob2 = gal_jacob

        mask = np.where(weight!=0)
        w.append(np.mean(weight[mask]))
        noise = old_div(np.ones_like(weight),w[-1])

        psf_obs = Observation(im_psf, jacobian=gal_jacob, meta={'offset_pixels':None,'file_id':None})
        psf_obs2 = Observation(im_psf2, jacobian=psf_jacob2, meta={'offset_pixels':None,'file_id':None})
        obs = Observation(im, weight=weight, jacobian=gal_jacob, psf=psf_obs, meta={'offset_pixels':None,'file_id':None})
        obs.set_noise(noise)

        obs_list.append(obs)
        psf_list.append(psf_obs2)

    #print(obs_list)
    return obs_list,psf_list,np.array(w)

def Observation():
	# Obtain observation list, psf list, weight from wfirst_simple_sim.py
	gals_array=[fio.FITS('mcal_gal_0.fits')[-1].read(), fio.FITS('mcal_gal_1.fits')[-1].read()]
	psfs_array=[fio.FITS('mcal_psf_0.fits')[-1].read(), fio.FITS('mcal_psf_1.fits')[-1].read()]
	offsets=[galsim.PositionD(-0.5,-0.5), galsim.PositionD(0.48010974293083564,0.24526554339263384)]
	skys_array=[fio.FITS('mcal_sky_0.fits')[-1].read(), fio.FITS('mcal_sky_1.fits')[-1].read()]
	gal_true=[galsim.PositionD(31.5,31.5), galsim.PositionD(31.5,31.5)]
	gal_jacob=[galsim.JacobianWCS(0.10379201786865774, -0.037313406917181026, 0.03741492083530528, 0.1017841347351108), 
				galsim.JacobianWCS(0.04693577579580337, -0.09835662875798407, 0.09984826919988712, 0.045587756760917426)]

	obs_list, psf_list, w = get_exp_list(gals_array, psfs_array, offsets, skys_array, gal_true, gal_jacob, psf2=None)
	print(obs_list, psf_list, w)

Observation()