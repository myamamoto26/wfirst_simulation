
## a simple galaxy image simulation
# version 1
#   -> just the stamps (don't have to patch on the larger box.)
#   -> gaussian PSF (fwhm=0.178 arcsec/pix)
#   -> gaussian galaxy profile
#   -> pixel scale=0.11 arcsec, which can define Jacobian to calculate WCS. 
#   -> Run metacal and save output. 
# * no detector effects


# Let's start from modifying wfirst module in galsim! 

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

# This is a setting that decides whether or not to output differences images showing what each
# detector effect does.  Since they take up quite a bit of space, we set this to False by default,
# but users who want to see the difference images can change it to True.
diff_mode = False

filter_flux_dict = {
    'J129' : 'j_WFIRST',
    'F184' : 'F184W_WFIRST',
    'Y106' : 'y_WFIRST',
    'H158' : 'h_WFIRST'
}

filter_dither_dict = {
    'J129' : 3,
    'F184' : 1,
    'Y106' : 4,
    'H158' : 2
}

BAD_MEASUREMENT = 1
CENTROID_SHIFT  = 2
MAX_CENTROID_SHIFT = 1.

def dump_truth_gal(filename,store):
    """
    Write galaxy truth catalog to disk.

    Input
    filename    : Fits filename
    store       : Galaxy truth catalog
    """

    fio.write(filename,store,clobber=True)

    return fio.FITS(filename)[-1]

## initiating galaxy catalogs
def init_gal(gal_dist, gal_sample):

    radec_file = fio.FITS(gal_dist)[-1]
    #store = fio.FITS(filename)[-1]
    gal_rng = galsim.UniformDeviate(314)
    shear_list = [[0.05,0.0],[-0.05,0.0],[0.0,0.05],[0.0,-0.05],[0.05,0.05],[0.05,-0.05],[-0.05,0.05],[-0.05,-0.05]]

    # Read in file with photometry/size/redshift distribution similar to WFIRST galaxies
    phot       = fio.FITS(gal_sample)[-1].read(columns=['fwhm','redshift',filter_flux_dict['J129'],filter_flux_dict['F184'],filter_flux_dict['Y106'],filter_flux_dict['H158']])
    pind_list_ = np.ones(len(phot)).astype(bool) # storage list for original index of photometry catalog
    pind_list_ = pind_list_&(phot[filter_flux_dict['J129']]<99)&(phot[filter_flux_dict['J129']]>0) # remove bad mags
    pind_list_ = pind_list_&(phot[filter_flux_dict['F184']]<99)&(phot[filter_flux_dict['F184']]>0) # remove bad mags
    pind_list_ = pind_list_&(phot[filter_flux_dict['Y106']]<99)&(phot[filter_flux_dict['Y106']]>0) # remove bad mags
    pind_list_ = pind_list_&(phot[filter_flux_dict['H158']]<99)&(phot[filter_flux_dict['H158']]>0) # remove bad mags
    pind_list_ = pind_list_&(phot['redshift']>0)&(phot['redshift']<5) # remove bad redshifts
    pind_list_ = np.where(pind_list_)[0]

    n_gal = radec_file.read_header()['NAXIS2']
    #print('Number of galaxies, ', n_gal)

    # Create minimal storage array for galaxy properties
    store = np.ones(n_gal, dtype=[('gind','i4')]
                                +[('ra',float)]
                                +[('dec',float)]
                                +[('g1','f4')]
                                +[('g2','f4')]
                                +[('int_e1','f4')]
                                +[('int_e2','f4')]
                                +[('rot','f4')]
                                +[('size','f4')]
                                +[('z','f4')]
                                +[('J129','f4')]
                                +[('F184','f4')]
                                +[('Y106','f4')]
                                +[('H158','f4')]
                                +[('pind','i4')]
                                +[('bflux','f4')]
                                +[('dflux','f4')])
    store['gind']       = np.arange(n_gal) # Index array into original galaxy position catalog
    store['ra']         = radec_file['ra'][:]*np.pi/180. # Right ascension
    store['dec']        = radec_file['dec'][:]*np.pi/180. # Declination
    r_ = np.zeros(n_gal)
    gal_rng.generate(r_)
    store['pind']       = pind_list_[(r_*len(pind_list_)).astype(int)] # Index array into original galaxy photometry catalog
    r_ = np.zeros(int(old_div(n_gal,2))+n_gal%2)
    gal_rng.generate(r_)
    store['rot'][0::2]  = r_*2.*np.pi # Random rotation (every pair of objects is rotated 90 deg to cancel shape noise)
    store['rot'][1::2]  = store['rot'][0:n_gal-n_gal%2:2]+np.pi
    store['rot'][store['rot']>2.*np.pi]-=2.*np.pi
    r_ = np.zeros(n_gal)
    gal_rng.generate(r_)
    r_ = (r_*len(shear_list)).astype(int)
    np.random.seed(seed=314)

    store['g1']         = np.array(shear_list)[r_,0] # Shears to apply to galaxy
    store['g2']         = np.array(shear_list)[r_,1]
    store['int_e1']     = np.random.normal(scale=0.27,size=n_gal) # Intrinsic shape of galaxy
    store['int_e2']     = np.random.normal(scale=0.27,size=n_gal)
    store['int_e1'][store['int_e1']>0.7] = 0.7
    store['int_e2'][store['int_e2']>0.7] = 0.7
    store['int_e1'][store['int_e1']<-0.7] = -0.7
    store['int_e2'][store['int_e2']<-0.7] = -0.7
    store['size']       = phot['fwhm'][store['pind']] * 0.06 / 2. # half-light radius
    store['z']          = phot['redshift'][store['pind']] # redshift
    for f in list(filter_dither_dict.keys()):
        store[f]        = phot[filter_flux_dict[f]][store['pind']] # magnitude in this filter
    for name in store.dtype.names:
        print(name,np.mean(store[name]),np.min(store[name]),np.max(store[name]))

    # Save truth file with galaxy properties
    return dump_truth_gal('truth.fits',store)

    print('-------truth catalog built-------')


def get_wcs(dither_i, sca, filter_, stamp_size, random_angle):
    dither_i = dither_i
    sca = sca
    filter_ = filter_

    bpass = wfirst.getBandpasses(AB_zeropoint=True)[filter_]

    d = fio.FITS('observing_sequence_hlsonly_5yr.fits')[-1][dither_i]
    ra     = d['ra']  * np.pi / 180. # RA of pointing
    dec    = d['dec'] * np.pi / 180. # Dec of pointing
    #pa     = d['pa']  * np.pi / 180.  # Position angle of pointing
    date   = Time(d['date'],format='mjd').datetime

    #random_dir = galsim.UniformDeviate(314)
    #pa = math.pi * random_dir()
    pa=random_angle * np.pi /180.

    WCS = wfirst.getWCS(world_pos  = galsim.CelestialCoord(ra=ra*galsim.radians, \
                                                           dec=dec*galsim.radians), 
                                PA          = pa*galsim.radians, 
                                date        = date,
                                SCAs        = sca,
                                PA_is_FPA   = True
                                )[sca]

    sky_level = wfirst.getSkyLevel(bpass, 
                                            world_pos=WCS.toWorld(
                                                        galsim.PositionI(old_div(wfirst.n_pix,2),
                                                                        old_div(wfirst.n_pix,2))), 
                                            date=date)
    sky_level *= (1.0 + wfirst.stray_light_fraction)*(wfirst.pixel_scale)**2 # adds stray light and converts to photons/cm^2
    sky_level *= stamp_size*stamp_size

    return WCS, sky_level

def add_background(im,  sky_level, b, thermal_backgrounds=None, filter_='H158', phot=False):
    sky_stamp = galsim.Image(bounds=b, scale=wfirst.pixel_scale)
    #local_wcs.makeSkyImage(sky_stamp, sky_level)

    # This image is in units of e-/pix. Finally we add the expected thermal backgrounds in this
    # band. These are provided in e-/pix/s, so we have to multiply by the exposure time.
    if thermal_backgrounds is None:
        sky_stamp += wfirst.thermal_backgrounds[filter_]*wfirst.exptime
    else:
        sky_stamp += thermal_backgrounds*wfirst.exptime

    # Adding sky level to the image.
    if not phot:
        im += sky_stamp
    
    return im,sky_stamp

def getPSF(PSF_model, sca, filter_, bpass):
    
    if PSF_model == "Gaussian":
        psf = galsim.Gaussian(fwhm=0.178)
    #elif PSF_model == 'exponential':
    elif PSF_model == 'wfirst':
        psf = wfirst.getPSF(sca, filter_, SCA_pos=None, approximate_struts=True, wavelength=bpass.effective_wavelength, high_accuracy=False)

    return psf

def add_poisson_noise(rng, im, sky_image, phot=False):

    noise = galsim.PoissonNoise(rng)
    # Add poisson noise to image
    if phot:
        sky_image_ = sky_image.copy()
        sky_image_.addNoise(noise)
        im += sky_image_
    else:
        im.addNoise(noise)

    return im

def make_sed_model(model, sed, filter_, bpass):
    """
    Modifies input SED to be at appropriate redshift and magnitude, then applies it to the object model.

    Input
    model : Galsim object model
    sed   : Template SED for object
    flux  : flux fraction in this sed
    """

    # Apply correct flux from magnitude for filter bandpass
    sed_ = sed.atRedshift(0) #picking z=0 for now. 
    target_mag = sed_.calculateMagnitude(bpass)
    sed_ = sed_.withMagnitude(target_mag, bpass)

    # Return model with SED applied
    return model * sed_

## metacal shapemeasurement
def get_exp_list(gal, psf, offsets, sky_stamp, psf2=None):
    #def get_exp_list(gal, psf, sky_stamp, psf2=None):

    if psf2 is None:
        psf2 = psf

    obs_list=ObsList()
    psf_list=ObsList()

    w = []
    for i in range(len(gal)):
        im = gal[i].array
        im_psf = psf[i].array
        im_psf2 = psf2[i].array
        weight = 1/sky_stamp[i].array

        jacob = gal[i].wcs.jacobian()
        dx = offsets[i].x
        dy = offsets[i].y
        
        gal_jacob = Jacobian(
            row=gal[i].true_center.y+dy,
            col=gal[i].true_center.x+dx,
            dvdrow=jacob.dvdy,
            dvdcol=jacob.dvdx,
            dudrow=jacob.dudy,
            dudcol=jacob.dudx)
        #gal_jacob = Jacobian(
        #    row=gal[i].true_center.x+dx,
        #    col=gal[i].true_center.y+dy,
        #    dvdrow=jacob.dudx,
        #    dvdcol=jacob.dudy,
        #    dudrow=jacob.dvdx,
        #    dudcol=jacob.dvdy)
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


def shape_measurement_metacal(obs_list, metacal_pars, T, flux=1000.0, fracdev=None, use_e=None):
    pix_range = old_div(galsim.wfirst.pixel_scale,10.)
    e_range = 0.1
    fdev = 1.
    def pixe_guess(n):
        return 2.*n*np.random.random() - n

    cp = ngmix.priors.CenPrior(0.0, 0.0, galsim.wfirst.pixel_scale, galsim.wfirst.pixel_scale)
    gp = ngmix.priors.GPriorBA(0.3)
    hlrp = ngmix.priors.FlatPrior(1.0e-4, 1.0e2)
    fracdevp = ngmix.priors.Normal(0.5, 0.1, bounds=[0., 1.])
    fluxp = ngmix.priors.FlatPrior(0, 1.0e5)

    prior = joint_prior.PriorSimpleSep(cp, gp, hlrp, fluxp)
    guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,500.])

    boot = ngmix.bootstrap.MaxMetacalBootstrapper(obs_list)
    psf_model = "gauss"
    gal_model = "gauss"

    lm_pars={'maxfev':2000, 'xtol':5.0e-5, 'ftol':5.0e-5}
    max_pars={'method': 'lm', 'lm_pars':lm_pars}

    Tguess=T**2/(2*np.log(2))
    ntry=2
    boot.fit_metacal(psf_model, gal_model, max_pars, Tguess, prior=prior, ntry=ntry, metacal_pars=metacal_pars) 
    res_ = boot.get_metacal_result()

    return res_

def measure_shape_ngmix(obs_list,T,flux=1000.0,model='gauss'):
        
    pix_range = old_div(galsim.wfirst.pixel_scale,10.)
    e_range = 0.1
    fdev = 1.
    def pixe_guess(n):
        return 2.*n*np.random.random() - n

    # possible models are 'exp','dev','bdf' galsim.wfirst.pixel_scale
    cp = ngmix.priors.CenPrior(0.0, 0.0, galsim.wfirst.pixel_scale, galsim.wfirst.pixel_scale)
    gp = ngmix.priors.GPriorBA(0.3)
    hlrp = ngmix.priors.FlatPrior(1.0e-4, 1.0e2)
    fracdevp = ngmix.priors.TruncatedGaussian(0.5, 0.5, -0.5, 1.5)
    fluxp = ngmix.priors.FlatPrior(0, 1.0e5) # not sure what lower bound should be in general

    prior = joint_prior.PriorBDFSep(cp, gp, hlrp, fracdevp, fluxp)
    # center1 + center2 + shape + hlr + fracdev + fluxes for each object
    # guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,0.5+pixe_guess(fdev),100.])
    guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,pixe_guess(fdev),300.])

    guesser           = R50FluxGuesser(T,flux)
    ntry              = 5
    runner            = GalsimRunner(obs_list,model,guesser=guesser)
    runner.go(ntry=ntry)
    fitter            = runner.get_fitter()

    res_ = fitter.get_result()
    res_['flux'] = res_['pars'][5]
    return res_

def get_coadd_shape(cat, gals, psfs, offsets, sky_stamp, i, hlr, res_tot, g1, g2, shape):
    #def get_coadd_shape(cat, gals, psfs, sky_stamp, i, hlr, res_tot, g1, g2):

    def get_flux(obj):
        flux=0.
        for obs in obj:
            flux += obs.image.sum()
        flux /= len(obj)
        if flux<0:
            flux = 10.
        return flux

    #truth = cat
    #res = np.zeros(len(gals), dtype=[('ind', int), ('ra', float), ('dec', float), ('int_e1', float), ('int_e2', float), ('g1', float), ('g2', float), ('e1', float), ('e2', float), ('snr', float), ('hlr', float), ('flags', int)])

    metacal_pars={'types': ['noshear', '1p', '1m', '2p', '2m'], 'psf': 'gauss'}
    metacal_keys=['noshear', '1p', '1m', '2p', '2m']

    #for i in range(len(gals)):
    #t = truth[i]
    #obs_list,psf_list,w = get_exp_list(t,gals,psfs,sky_stamp,psf2=None,size=t['size'])
    obs_list,psf_list,w = get_exp_list(gals,psfs,offsets,sky_stamp,psf2=None)
    #obs_list,psf_list,w = get_exp_list(gals,psfs,sky_stamp,psf2=None)
    #res_ = shape_measurement(obs_list,metacal_pars,hlr,flux=get_flux(obs_list),fracdev=t['bflux'],use_e=[t['int_e1'],t['int_e2']])
    if shape=='metacal':
        res_ = shape_measurement_metacal(obs_list,metacal_pars,hlr,flux=get_flux(obs_list),fracdev=None,use_e=None)

        iteration=0
        for key in metacal_keys:
            res_tot[iteration]['ind'][i]                       = i
            #res_tot[iteration]['ra'][i]                        = t['ra']
            #res_tot[iteration]['dec'][i]                       = t['dec']
            res_tot[iteration]['g1'][i]                        = g1
            res_tot[iteration]['g2'][i]                        = g2
            #res_tot[iteration]['int_e1'][i]                    = t['int_e1']
            #res_tot[iteration]['int_e2'][i]                    = t['int_e2']

            res_tot[iteration]['snr'][i]                       = np.copy(res_[key]['s2n_r'])
            res_tot[iteration]['flux'][i]                      = np.copy(res_[key]['flux'])
            res_tot[iteration]['e1'][i]                        = np.copy(res_[key]['pars'][2])
            res_tot[iteration]['e2'][i]                        = np.copy(res_[key]['pars'][3])
            res_tot[iteration]['hlr'][i]                       = np.copy(res_[key]['pars'][4])
            iteration+=1

    elif shape=='ngmix':
        res_ = measure_shape_ngmix(obs_list, hlr, model='gauss')
        res_tot[0]['ind'][i]                       = i
        #res_tot[iteration]['ra'][i]               = t['ra']
        #res_tot[iteration]['dec'][i]              = t['dec']
        res_tot[0]['g1'][i]                        = g1
        res_tot[0]['g2'][i]                        = g2
        #res_tot[iteration]['int_e1'][i]                    = t['int_e1']
        #res_tot[iteration]['int_e2'][i]                    = t['int_e2']
        res_tot[0]['snr'][i]                       = np.copy(res_['s2n_r'])
        res_tot[0]['flux'][i]                      = np.copy(res_['flux'])
        res_tot[0]['e1'][i]                        = np.copy(res_['pars'][2])
        res_tot[0]['e2'][i]                        = np.copy(res_['pars'][3])
        res_tot[0]['hlr'][i]                       = np.copy(res_['pars'][4])

    return res_tot

def main(argv):

    ## fixed parameters
    random_seed = 314
    rng = galsim.BaseDeviate(random_seed)
    random_dir = galsim.UniformDeviate(rng)
    poisson_noise = galsim.PoissonNoise(rng)
    dither_i = 22535
    use_SCA = 1
    filter_ = 'H158'
    stamp_size = 32
    hlr = 1.0
    bpass = wfirst.getBandpasses(AB_zeropoint=True)[filter_]
    galaxy_sed_n = galsim.SED('Mrk_33_spec.dat',  wave_type='Ang', flux_type='flambda')

    ## variable arguments
    gal_num = int(sys.argv[1])
    galaxy_model = sys.argv[2]
    PSF_model = sys.argv[3]
    shape=sys.argv[4]

    # when using more galaxies than the length of truth file. 
    res_noshear = np.zeros(gal_num, dtype=[('ind', int), ('flux', float), ('g1', float), ('g2', float), ('e1', float), ('e2', float), ('snr', float), ('hlr', float), ('flags', int)])
    res_1p = np.zeros(gal_num, dtype=[('ind', int), ('flux', float), ('g1', float), ('g2', float), ('e1', float), ('e2', float), ('snr', float), ('hlr', float), ('flags', int)])
    res_1m = np.zeros(gal_num, dtype=[('ind', int), ('flux', float), ('g1', float), ('g2', float), ('e1', float), ('e2', float), ('snr', float), ('hlr', float), ('flags', int)])
    res_2p = np.zeros(gal_num, dtype=[('ind', int), ('flux', float), ('g1', float), ('g2', float), ('e1', float), ('e2', float), ('snr', float), ('hlr', float), ('flags', int)])
    res_2m = np.zeros(gal_num, dtype=[('ind', int), ('flux', float), ('g1', float), ('g2', float), ('e1', float), ('e2', float), ('snr', float), ('hlr', float), ('flags', int)])
    if shape=='metacal':
        res_tot=[res_noshear, res_1p, res_1m, res_2p, res_2m]
    elif shape=='ngmix':
        res_tot=[res_noshear]

    PSF = getPSF(PSF_model, use_SCA, filter_, bpass)
    position_angle1=20 #degrees
    position_angle2=70 #degrees
    wcs1, sky_level1 = get_wcs(dither_i, use_SCA, filter_, stamp_size, position_angle1)
    wcs2, sky_level2 = get_wcs(dither_i, use_SCA, filter_, stamp_size, position_angle2)
    wcs=[wcs1, wcs2]
    sky_level=[sky_level1, sky_level2]
    

    t0 = time.time()
    for i_gal in range(gal_num):
        if i_gal%size != rank: 
            continue

        if i_gal % 100 == 0:
            print('rank', rank, 'object number, ', i_gal)
        
        gal_model = None
        st_model = None

        if galaxy_model == "Gaussian":
            tot_mag = np.random.choice(cat)
            sed = galsim.SED('CWW_E_ext.sed', 'A', 'flambda')
            sed = sed.withMagnitude(tot_mag, bpass)
            flux = sed.calculateFlux(bpass)
            gal_model = galsim.Gaussian(half_light_radius=hlr, flux=1.) # needs to normalize the flux before multiplying by sed. For bdf, there are bulge, disk, knots fractions to sum to 1. 
            ## making galaxy sed
            #knots = galsim.RandomKnots(10, half_light_radius=1.3, flux=100)
            #knots = make_sed_model(galsim.ChromaticObject(knots), galaxy_sed_n, filter_, bpass)
            #gal_model = galsim.Add([gal_model, knots])
            gal_model = sed * gal_model
            ## shearing
            if i_gal%2 == 0:
                gal_model = gal_model.shear(g1=0.02,g2=0)
                g1=0.02
                g2=0
            else:
                gal_model = gal_model.shear(g1=-0.02,g2=0)
                g1=-0.02
                g2=0
        elif galaxy_model == "exponential":
            tot_mag = np.random.choice(cat)
            sed = galsim.SED('CWW_E_ext.sed', 'A', 'flambda')
            sed = sed.withMagnitude(tot_mag, bpass)
            flux = sed.calculateFlux(bpass)
            gal_model = galsim.Exponential(half_light_radius=hlr, flux=1.)
            ## making galaxy sed ## random knots should be used for bdf model
            #knots = galsim.RandomKnots(10, half_light_radius=1.3, flux=1.)
            #knots = make_sed_model(galsim.ChromaticObject(knots), galaxy_sed_n, filter_, bpass)
            #gal_model = galsim.Add([gal_model, knots])
            gal_model = sed * gal_model
            ## shearing
            if i_gal%2 == 0:
                gal_model = gal_model.shear(g1=0,g2=0.02)
                g1=0
                g2=0.02
            else:
                gal_model = gal_model.shear(g1=0,g2=-0.02)
                g1=0
                g2=-0.02

        gal_model = gal_model * galsim.wfirst.collecting_area * galsim.wfirst.exptime

        flux_ = gal_model.calculateFlux(bpass)
        #mag_ = gal_model.calculateMagnitude(bpass)
        # This makes the object achromatic, which speeds up drawing and convolution
        gal_model  = gal_model.evaluateAtWavelength(bpass.effective_wavelength)
        # Reassign correct flux
        gal_model  = gal_model.withFlux(flux_)
        gal_model = galsim.Convolve(gal_model, PSF)

        st_model = galsim.DeltaFunction(flux=1.)
        st_model = st_model.evaluateAtWavelength(bpass.effective_wavelength)
        # reassign correct flux
        starflux=1.
        st_model = st_model.withFlux(starflux)
        st_model = galsim.Convolve(st_model, PSF)

        stamp_size_factor = old_div(int(gal_model.getGoodImageSize(wfirst.pixel_scale)), stamp_size)
        if stamp_size_factor == 0:
            stamp_size_factor = 1

        # Galsim world coordinate object (ra,dec)
        """
        ra = cat[i_gal]['ra']
        dec = cat[i_gal]['dec']
        int_e1 = cat[i_gal]['int_e1']
        int_e2 = cat[i_gal]['int_e2']
        radec = galsim.CelestialCoord(ra*galsim.radians, dec*galsim.radians)
        # Galsim image coordinate object 
        xy = wcs.toImage(radec)
        # Galsim integer image coordinate object 
        xyI = galsim.PositionI(int(xy.x),int(xy.y))
        """
        #xyI = galsim.PositionI(int(stamp_size_factor*stamp_size), int(stamp_size_factor*stamp_size))
        #b = galsim.BoundsI( xmin=1,
        #                    xmax=xyI.x,
        #                    ymin=1,
        #                    ymax=xyI.y)
        #print(xyI.x, int(stamp_size_factor*stamp_size), xyI.x-old_div(int(stamp_size_factor*stamp_size),2)+1)
        #print(b)
        # Create postage stamp for galaxy
        #print("galaxy ", i_gal, ra, dec, int_e1, int_e2)

        ## translational dither check (multiple exposures)
        """
        position_angle1=20*random_dir() #degrees
        position_angle2=position_angle1 #degrees
        wcs1, sky_level1 = get_wcs(dither_i, use_SCA, filter_, stamp_size, position_angle1)
        wcs2, sky_level2 = get_wcs(dither_i, use_SCA, filter_, stamp_size, position_angle2)
        wcs=[wcs1, wcs2]
        sky_level=[sky_level1, sky_level2]
        """
        
        sca_center=[wcs[0].toWorld(galsim.PositionI(old_div(wfirst.n_pix,2),old_div(wfirst.n_pix,2))), wcs[1].toWorld(galsim.PositionI(old_div(wfirst.n_pix,2),old_div(wfirst.n_pix,2)))]
        gal_radec = sca_center[0]
        thetas = [position_angle1*(np.pi/180)*galsim.radians]
        offsets = []
        gals = []
        psfs = []
        skys = []
        for i in range(2): 
            gal_stamp=None
            psf_stamp=None
            xy = wcs[i].toImage(gal_radec) # galaxy position 
            xyI = galsim.PositionI(int(xy.x), int(xy.y))
            b = galsim.BoundsI( xmin=xyI.x-old_div(int(stamp_size_factor*stamp_size),2)+1,
                                ymin=xyI.y-old_div(int(stamp_size_factor*stamp_size),2)+1,
                                xmax=xyI.x+old_div(int(stamp_size_factor*stamp_size),2),
                                ymax=xyI.y+old_div(int(stamp_size_factor*stamp_size),2))
            #---------------------------------------#
            # if the image does not use a real wcs. #
            #---------------------------------------#
            #b = galsim.BoundsI( xmin=1,
            #                    xmax=int(stamp_size_factor*stamp_size),
            #                    ymin=1,
            #                    ymax=int(stamp_size_factor*stamp_size))
            gal_stamp = galsim.Image(b, wcs=wcs[i]) #scale=wfirst.pixel_scale)
            psf_stamp = galsim.Image(b, wcs=wcs[i]) #scale=wfirst.pixel_scale)

            ## translational dithering test
            #dx = 0 #random_dir() - 0.5
            #dy = 0 #random_dir() - 0.5
            #offset = np.array((dx,dy))

            offset = xy-gal_stamp.true_center # original galaxy position - stamp center
            gal_model.drawImage(image=gal_stamp, offset=offset)
            st_model.drawImage(image=psf_stamp, offset=offset)

            sigma=wfirst.read_noise
            read_noise = galsim.GaussianNoise(rng, sigma=sigma)

            im,sky_image=add_background(gal_stamp, sky_level[i], b, thermal_backgrounds=None, filter_='H158', phot=False)
            #im.addNoise(read_noise)
            gal_stamp = add_poisson_noise(rng, im, sky_image=sky_image, phot=False)
            #sky_image = add_poisson_noise(rng, sky_image, sky_image=sky_image, phot=False)
            gal_stamp -= sky_image

            # set a simple jacobian to the stamps before sending them to ngmix
            # old center of the stamp
            origin_x = gal_stamp.origin.x
            origin_y = gal_stamp.origin.y
            gal_stamp.setOrigin(0,0)
            psf_stamp.setOrigin(0,0)
            new_pos = galsim.PositionD(xy.x-origin_x, xy.y-origin_y)
            wcs_transf = gal_stamp.wcs.affine(image_pos=new_pos)
            new_wcs = galsim.JacobianWCS(wcs_transf.dudx, wcs_transf.dudy, wcs_transf.dvdx, wcs_transf.dvdy)
            gal_stamp.wcs=new_wcs
            psf_stamp.wcs=new_wcs

            if i_gal==0:
                gal_stamp.write('mcal_gal_'+str(i)+'.fits')
                psf_stamp.write('mcal_psf_'+str(i)+'.fits')
                sky_image.write('mcal_sky_'+str(i)+'.fits')
                print(offset)

            offsets.append(offset)
            gals.append(gal_stamp)
            psfs.append(psf_stamp)
            skys.append(sky_image)
        res_tot = get_coadd_shape(cat, gals, psfs, offsets, skys, i_gal, hlr, res_tot, g1, g2, shape)
    exit()
    ## send and receive objects from one processor to others
    if rank!=0:
        # send res_tot to rank 0 processor
        comm.send(res_tot, dest=0)
    else:
        for i in range(comm.size):
            if i == 0:
                continue
            # for other processors, receive res_tot. 
            res_ = comm.recv(source=i)
            for j in range(len(res_tot)):
                for col in res_tot[j].dtype.names:
                    res_tot[j][col]+=res_[j][col]

    if rank==0:
        dirr='v2_3_redo'
        for i in range(len(res_tot)):
            fio.write(dirr+'_sim_'+str(i)+'.fits', res_tot[i])
            
    if rank==0:
        bias = residual_bias(res_tot, shape)
        #final = residual_bias_correction(res_tot,R11,R22,R12,R21)
        print(time.time()-t0)

    
    return None

if __name__ == "__main__":

    t0 = time.time()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #if rank==0:
    #    cat = init_gal('radec_sub.fits', 'Simulated_WFIRST+LSST_photometry_catalog_CANDELSbased.fits')
        ## do not create truth catalog. just draw random magnitudes from the second fits file. -> increase the number of galaxies. 
    #comm.Barrier()    
    cat = fio.FITS('truth_mag.fits')[-1].read()

    main(sys.argv)

