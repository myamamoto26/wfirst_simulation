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

def find_pa(filename):
    dither_number = open(filename).readlines()
    pa_list = []
    dither_list = []
    for i in range(len(dither_number)):
        dither = int(dither_number[i])
        data = fio.FITS('observing_sequence_hlsonly_5yr.fits')[-1][dither]
        pa_list.append(data['pa'] * np.pi / 180.)
        dither_list.append(dither)

    return pa_list, dither_list



class Pointing:

    def __init__(self, dither, sca, filter_, stamp_size, position_angle, random_angle=False):
        self.dither=dither
        self.sca=sca
        self.filter_=filter_
        self.stamp_size=stamp_size
        self.random_angle=random_angle
        self.position_angle=position_angle

        ## create reference ra, dec
        self.bpass=wfirst.getBandpasses(AB_zeropoint=True)[self.filter_]
        d=fio.FITS('observing_sequence_hlsonly_5yr.fits')[-1][self.dither]
        self.ra     = d['ra'] * np.pi / 180.
        self.dec    = d['dec'] * np.pi / 180.
        self.date   = Time(d['date'],format='mjd').datetime
    """
    def find_coordinates(self):
        self.dither_i = find_dither_number(patch=True)

        self.bpass=wfirst.getBandpasses(AB_zeropoint=True)[self.filter_]
        self.d=fio.FITS('observing_sequence_hlsonly_5yr.fits')[-1][self.dither_i]

        self.ra     = self.d['ra']  * np.pi / 180. # RA of pointing
        self.dec    = self.d['dec'] * np.pi / 180. # Dec of pointing
        self.pa     = self.d['pa']  * np.pi / 180.  # Position angle of pointing
        self.date   = Time(self.d['date'],format='mjd').datetime

        if (self.ra >= self.ref_ra-2.5) and (self.ra < self.ref_ra+2.5):
            if (self.dec >= self.ref_dec-2.5) and (self.dec < self.ref_dec+2.5):
                return self.pa
    """

    def find_sca_center(self):
        wcs_ref,sky_ref=self.get_wcs()
        return wcs_ref.toWorld(galsim.PositionI(old_div(wfirst.n_pix,2),old_div(wfirst.n_pix,2)))

    def get_wcs(self):
        #self.find_coordinates()
        WCS = wfirst.getWCS(world_pos  = galsim.CelestialCoord(ra=self.ra*galsim.radians, \
                                                               dec=self.dec*galsim.radians), 
                                    PA          = self.position_angle*galsim.radians, 
                                    date        = self.date,
                                    SCAs        = self.sca,
                                    PA_is_FPA   = True
                                    )[self.sca]

        sky_level = wfirst.getSkyLevel(self.bpass, 
                                                world_pos=WCS.toWorld(
                                                            galsim.PositionI(old_div(wfirst.n_pix,2),
                                                                            old_div(wfirst.n_pix,2))), 
                                                date=self.date)
        sky_level *= (1.0 + wfirst.stray_light_fraction)*(wfirst.pixel_scale)**2 # adds stray light and converts to photons/cm^2
        sky_level *= self.stamp_size*self.stamp_size

        return WCS, sky_level

class Model:
    def __init__(self, cat, gal_prof, psf_prof, sca, filter_, bpass, hlr, i_gal):
        self.cat=cat
        self.gal_prof=gal_prof
        self.psf_prof=psf_prof
        self.sca=sca
        self.filter_=filter_
        self.bpass=bpass
        self.hlr=hlr
        self.i_gal=i_gal


    def getPSF(self):
    
        if self.psf_prof == "Gaussian":
            psf = galsim.Gaussian(fwhm=0.178)
        #elif PSF_model == 'exponential':
        elif self.psf_prof == 'wfirst':
            psf = wfirst.getPSF(self.sca, self.filter_, SCA_pos=None, approximate_struts=True, wavelength=self.bpass.effective_wavelength, high_accuracy=False)

        return psf

    def make_sed_model(self, model, sed):
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

    def flux_model(self):
        tot_mag = np.random.choice(self.cat)
        sed = galsim.SED('CWW_E_ext.sed', 'A', 'flambda')
        sed = sed.withMagnitude(tot_mag, self.bpass)
        #flux = sed.calculateFlux(self.bpass)

        return sed

    def draw_galaxy(self):
        if self.gal_prof=='Gaussian':
            gal_model = galsim.Gaussian(half_light_radius=self.hlr, flux=1.) # needs to normalize the flux before multiplying by sed. For bdf, there are bulge, disk, knots fractions to sum to 1. 
        elif self.gal_prof=='exponential':
            gal_model = galsim.Exponential(half_light_radius=self.hlr, flux=1.)
        ## making galaxy sed
        #knots = galsim.RandomKnots(10, half_light_radius=1.3, flux=100)
        #knots = make_sed_model(galsim.ChromaticObject(knots), galaxy_sed_n, filter_, bpass)
        #gal_model = galsim.Add([gal_model, knots])
        self.sed=self.flux_model()
        gal_model = self.sed * gal_model

        if self.i_gal%2 == 0:
            g1=0.02
            g2=0
            gal_model = gal_model.shear(g1=g1,g2=g2)
        else:
            g1=-0.02
            g2=0
            gal_model = gal_model.shear(g1=g1,g2=g2)

        gal_model = gal_model * galsim.wfirst.collecting_area * galsim.wfirst.exptime

        flux_ = gal_model.calculateFlux(self.bpass)
        #mag_ = gal_model.calculateMagnitude(bpass)
        # This makes the object achromatic, which speeds up drawing and convolution
        gal_model  = gal_model.evaluateAtWavelength(self.bpass.effective_wavelength)
        # Reassign correct flux
        gal_model  = gal_model.withFlux(flux_)
        gal_model = galsim.Convolve(gal_model, self.getPSF())

        return gal_model, g1, g2

    def draw_star(self):
        st_model = galsim.DeltaFunction(flux=1.)
        st_model = st_model.evaluateAtWavelength(self.bpass.effective_wavelength)
        # reassign correct flux
        starflux=1.
        st_model = st_model.withFlux(starflux)
        st_model = galsim.Convolve(st_model, self.getPSF())

        return st_model

class Image:

    def __init__(self, i_gal, stamp_size, gal_model, st_model, pointing, sca_center):
        self.i_gal=i_gal
        self.stamp_size=stamp_size
        self.gal_model=gal_model
        self.st_model=st_model
        self.pointing=pointing
        self.sca_center=sca_center

        self.stamp_size_factor = old_div(int(self.gal_model.getGoodImageSize(wfirst.pixel_scale)), self.stamp_size)
        if self.stamp_size_factor == 0:
            self.stamp_size_factor = 1

        self.wcs, self.sky_level=self.pointing.get_wcs()
        self.xy = self.wcs.toImage(self.sca_center) # galaxy position 
        self.xyI = galsim.PositionI(int(self.xy.x), int(self.xy.y))
        self.b = galsim.BoundsI( xmin=self.xyI.x-old_div(int(self.stamp_size_factor*self.stamp_size),2)+1,
                            ymin=self.xyI.y-old_div(int(self.stamp_size_factor*self.stamp_size),2)+1,
                            xmax=self.xyI.x+old_div(int(self.stamp_size_factor*self.stamp_size),2),
                            ymax=self.xyI.y+old_div(int(self.stamp_size_factor*self.stamp_size),2))

    def make_stamp(self):
        ra=self.pointing.ra
        dec=self.pointing.dec

        # Galsim world coordinate object (ra,dec)
        """
        radec = galsim.CelestialCoord(ra*galsim.radians, dec*galsim.radians)
        # Galsim image coordinate object 
        wcs,sky=self.pointing1.get_wcs()
        xy = wcs.toImage(radec)
        # Galsim integer image coordinate object 
        xyI = galsim.PositionI(int(xy.x),int(xy.y))
        """
        #xyI = galsim.PositionI(int(self.stamp_size_factor*self.stamp_size), int(self.stamp_size_factor*self.stamp_size))
        #b = galsim.BoundsI( xmin=1,
        #                    xmax=xyI.x,
        #                    ymin=1,
        #                    ymax=xyI.y)
        #---------------------------------------#
        # if the image does not use a real wcs. #
        #---------------------------------------#
        #b = galsim.BoundsI( xmin=1,
        #                    xmax=int(stamp_size_factor*stamp_size),
        #                    ymin=1,
        #                    ymax=int(stamp_size_factor*stamp_size))

        self.gal_stamp = galsim.Image(self.b, wcs=self.wcs) #scale=wfirst.pixel_scale)
        self.psf_stamp = galsim.Image(self.b, wcs=self.wcs) #scale=wfirst.pixel_scale)

    def translational_dithering(self):
        ## translational dithering test
        random_dir = galsim.UniformDeviate(314)
        dx = random_dir() - 0.5
        dy = random_dir() - 0.5
        offset = np.array((dx,dy))
        return offset

    def draw_image(self, gal_model, st_model):
        self.make_stamp()

        offset = self.xy-self.gal_stamp.true_center # original galaxy position - stamp center
        gal_model.drawImage(image=self.gal_stamp, offset=offset)
        st_model.drawImage(image=self.psf_stamp, offset=offset)

        return self.gal_stamp, self.psf_stamp, offset

    def add_background(self, im, thermal_backgrounds=None, filter_='H158', phot=False):
        sky_stamp = galsim.Image(bounds=self.b, scale=wfirst.pixel_scale)
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

    def add_poisson_noise(self, rng, im, sky_image, phot=False):

        noise = galsim.PoissonNoise(rng)
        # Add poisson noise to image
        if phot:
            sky_image_ = sky_image.copy()
            sky_image_.addNoise(noise)
            im += sky_image_
        else:
            im.addNoise(noise)

        return im

    def add_noise(self, rng, gal_stamp):
        #self.make_stamp()

        sigma=wfirst.read_noise
        read_noise = galsim.GaussianNoise(rng, sigma=sigma)

        im,sky_stamp = self.add_background(gal_stamp, thermal_backgrounds=None, filter_='H158', phot=False)
        #im.addNoise(read_noise)
        gal_stamp = self.add_poisson_noise(rng, im, sky_image=sky_stamp, phot=False)
        #sky_image = add_poisson_noise(rng, sky_image, sky_image=sky_image, phot=False)
        gal_stamp -= sky_stamp

        return gal_stamp, sky_stamp

    def wcs_approx(self, gal_stamp, psf_stamp):
        #self.make_stamp()
        # set a simple jacobian to the stamps before sending them to ngmix
        # old center of the stamp
        origin_x = gal_stamp.origin.x
        origin_y = gal_stamp.origin.y
        gal_stamp.setOrigin(0,0)
        psf_stamp.setOrigin(0,0)
        new_pos = galsim.PositionD(self.xy.x-origin_x, self.xy.y-origin_y)
        wcs_transf = gal_stamp.wcs.affine(image_pos=new_pos)
        new_wcs = galsim.JacobianWCS(wcs_transf.dudx, wcs_transf.dudy, wcs_transf.dvdx, wcs_transf.dvdy)
        gal_stamp.wcs=new_wcs
        psf_stamp.wcs=new_wcs

        return gal_stamp, psf_stamp

class shape_measurement:

    def __init__(self, cat, gals, psfs, skys, offsets, i_gal, g1, g2, hlr, shape, res_tot):
        self.cat=cat
        self.gals=gals
        self.psfs=psfs
        self.skys=skys
        self.offsets=offsets
        self.i_gal=i_gal
        self.g1=g1
        self.g2=g2
        self.hlr=hlr
        self.shape=shape
        self.res_tot=res_tot

    def get_snr(self,obs_list,res):

        size = res['pars'][4]
        flux = res['flux']
        print(flux, res['pars'][5])

        model_ = galsim.Sersic(1, half_light_radius=1.*size, flux=flux*(1.-res['pars'][5])) + galsim.Sersic(4, half_light_radius=1.*size, flux=flux*res['pars'][5])
        for i in range(len(obs_list)):
            obs = obs_list[i]
            im = obs.psf.image.copy()
            im *= 1.0/im.sum()/len(obs_list)
            psf_gsimage = galsim.Image(im,wcs=obs.psf.jacobian.get_galsim_wcs())
            psf_ii = galsim.InterpolatedImage(psf_gsimage,x_interpolant='lanczos15')

            model = galsim.Convolve(model_,psf_ii)
            gal_stamp = galsim.Image(np.shape(obs.image)[0],np.shape(obs.image)[1], wcs=obs.jacobian.get_galsim_wcs())

            model.drawImage(image=gal_stamp)
            if i==0:
                image = gal_stamp.array*obs.weight
            else:
                image += gal_stamp.array*obs.weight

        return image.sum()

    def get_exp_list(self, psf2=None):

        if psf2 is None:
            psf2 = self.psfs

        obs_list=ObsList()
        psf_list=ObsList()

        w = []
        for i in range(len(self.gals)):
            im = self.gals[i].array
            im_psf = self.psfs[i].array
            im_psf2 = psf2[i].array
            weight = 1/self.skys[i].array

            jacob = self.gals[i].wcs.jacobian()
            dx = self.offsets[i].x
            dy = self.offsets[i].y
            
            gal_jacob = Jacobian(
                row=self.gals[i].true_center.y+dy,
                col=self.gals[i].true_center.x+dx,
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

        return obs_list,psf_list,np.array(w)

    def shape_measurement_metacal(self, obs_list, metacal_pars, flux=1000.0, fracdev=None, use_e=None):
        T = self.hlr
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

    def measure_shape_ngmix(self, obs_list, flux_=1000.0, model='gauss'):
        T = self.hlr
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

        guesser           = R50FluxGuesser(T,flux_)
        ntry              = 5
        runner            = GalsimRunner(obs_list,model,guesser=guesser)
        runner.go(ntry=ntry)
        fitter            = runner.get_fitter()

        res_ = fitter.get_result()
        res_['flux'] = res_['pars'][5]
        print(res_['s2n_r'], self.get_snr(obs_list,res_))
        res_['s2n_r'] = self.get_snr(obs_list,res_)
        return res_

    def ngmix_nobootstrap(self, obs_list, flux_):
        mcal_keys=['noshear', '1p', '1m', '2p', '2m']
        obsdict = ngmix.metacal.get_all_metacal(obs_list, psf='gauss')
        results_metacal = {}
        for key in mcal_keys:
            mobs = obsdict[key]
            res_= self.measure_shape_ngmix(mobs,flux_)
            results_metacal[key] = res_
        return results_metacal

    def get_coadd_shape(self):

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
        obs_list,psf_list,w = self.get_exp_list(psf2=None)
        i = self.i_gal
        #obs_list,psf_list,w = get_exp_list(gals,psfs,sky_stamp,psf2=None)
        #res_ = shape_measurement(obs_list,metacal_pars,hlr,flux=get_flux(obs_list),fracdev=t['bflux'],use_e=[t['int_e1'],t['int_e2']])
        if self.shape=='metacal':
            res_ = self.shape_measurement_metacal(obs_list, metacal_pars, flux=get_flux(obs_list), fracdev=None, use_e=None)

            iteration=0
            for key in metacal_keys:
                self.res_tot[iteration]['ind'][i]                       = i
                #res_tot[iteration]['ra'][i]                        = t['ra']
                #res_tot[iteration]['dec'][i]                       = t['dec']
                self.res_tot[iteration]['g1'][i]                        = self.g1
                self.res_tot[iteration]['g2'][i]                        = self.g2
                #res_tot[iteration]['int_e1'][i]                    = t['int_e1']
                #res_tot[iteration]['int_e2'][i]                    = t['int_e2']

                self.res_tot[iteration]['snr'][i]                       = np.copy(res_[key]['s2n_r'])
                self.res_tot[iteration]['flux'][i]                      = np.copy(res_[key]['flux'])
                self.res_tot[iteration]['e1'][i]                        = np.copy(res_[key]['pars'][2])
                self.res_tot[iteration]['e2'][i]                        = np.copy(res_[key]['pars'][3])
                self.res_tot[iteration]['hlr'][i]                       = np.copy(res_[key]['pars'][4])
                iteration+=1
        elif self.shape=='noboot':
            flux_ = get_flux(obs_list)
            res_ = self.ngmix_nobootstrap(obs_list,flux_)
            iteration=0
            for key in metacal_keys:
                self.res_tot[iteration]['ind'][i]                       = i
                #res_tot[iteration]['ra'][i]                        = t['ra']
                #res_tot[iteration]['dec'][i]                       = t['dec']
                self.res_tot[iteration]['g1'][i]                        = self.g1
                self.res_tot[iteration]['g2'][i]                        = self.g2
                #res_tot[iteration]['int_e1'][i]                    = t['int_e1']
                #res_tot[iteration]['int_e2'][i]                    = t['int_e2']
                self.res_tot[iteration]['snr'][i]                       = np.copy(res_[key]['s2n_r'])
                self.res_tot[iteration]['flux'][i]                      = np.copy(res_[key]['flux'])
                self.res_tot[iteration]['e1'][i]                        = np.copy(res_[key]['pars'][2])
                self.res_tot[iteration]['e2'][i]                        = np.copy(res_[key]['pars'][3])
                self.res_tot[iteration]['hlr'][i]                       = np.copy(res_[key]['pars'][4])
                iteration+=1

        elif self.shape=='ngmix':
            res_ = self.measure_shape_ngmix(obs_list, flux_=get_flux(obs_list), model='gauss')
            self.res_tot[0]['ind'][i]                       = i
            #res_tot[iteration]['ra'][i]               = t['ra']
            #res_tot[iteration]['dec'][i]              = t['dec']
            self.res_tot[0]['g1'][i]                        = self.g1
            self.res_tot[0]['g2'][i]                        = self.g2
            #res_tot[iteration]['int_e1'][i]                    = t['int_e1']
            #res_tot[iteration]['int_e2'][i]                    = t['int_e2']
            self.res_tot[0]['snr'][i]                       = np.copy(res_['s2n_r'])
            self.res_tot[0]['flux'][i]                      = np.copy(res_['flux'])
            self.res_tot[0]['e1'][i]                        = np.copy(res_['pars'][2])
            self.res_tot[0]['e2'][i]                        = np.copy(res_['pars'][3])
            self.res_tot[0]['hlr'][i]                       = np.copy(res_['pars'][4])

        return self.res_tot

def main(argv):

    ## fixed parameters
    random_seed = 314
    rng = galsim.BaseDeviate(random_seed)
    random_dir = galsim.UniformDeviate(rng)
    poisson_noise = galsim.PoissonNoise(rng)
    dither_file = '/hpc/group/cosmology/phy-lsst/dc2_truth/dc2_cen1deg.txt'
    reference_dither = 30592
    SCA = 1
    filter_ = 'H158'
    stamp_size = 32
    hlr = 1.0
    bpass = wfirst.getBandpasses(AB_zeropoint=True)[filter_]
    galaxy_sed_n = galsim.SED('Mrk_33_spec.dat',  wave_type='Ang', flux_type='flambda')

    ## variable arguments
    gal_num = int(sys.argv[1])
    #PA1 = int(sys.argv[2])
    #PA2 = int(sys.argv[3])
    gal_prof = sys.argv[2]
    psf_prof = sys.argv[3]
    shape = sys.argv[4]
    output_name = sys.argv[5]

    #PA_list, D_list = find_pa(dither_file)
    #exposures = np.array(PA_list)[np.random.choice(len(PA_list), 6)]
    #selected_dithers = np.array(D_list)[np.random.choice(len(D_list), 6)]
    exposures = [20, 65]
    selected_dithers = [22535, 22535]

    # when using more galaxies than the length of truth file. 
    res_noshear = np.zeros(gal_num, dtype=[('ind', int), ('flux', float), ('g1', float), ('g2', float), ('e1', float), ('e2', float), ('snr', float), ('hlr', float), ('flags', int)])
    res_1p = np.zeros(gal_num, dtype=[('ind', int), ('flux', float), ('g1', float), ('g2', float), ('e1', float), ('e2', float), ('snr', float), ('hlr', float), ('flags', int)])
    res_1m = np.zeros(gal_num, dtype=[('ind', int), ('flux', float), ('g1', float), ('g2', float), ('e1', float), ('e2', float), ('snr', float), ('hlr', float), ('flags', int)])
    res_2p = np.zeros(gal_num, dtype=[('ind', int), ('flux', float), ('g1', float), ('g2', float), ('e1', float), ('e2', float), ('snr', float), ('hlr', float), ('flags', int)])
    res_2m = np.zeros(gal_num, dtype=[('ind', int), ('flux', float), ('g1', float), ('g2', float), ('e1', float), ('e2', float), ('snr', float), ('hlr', float), ('flags', int)])
    if shape=='metacal':
        res_tot=[res_noshear, res_1p, res_1m, res_2p, res_2m]
    elif shape=='noboot':
        res_tot=[res_noshear, res_1p, res_1m, res_2p, res_2m]
    elif shape=='ngmix':
        res_tot=[res_noshear]

    t0 = time.time()
    for i_gal in range(gal_num):
        if i_gal%size != rank: 
            continue

        if i_gal % 100 == 0:
            print('rank', rank, 'object number, ', i_gal)
        
        gal_model = None
        st_model = None

        profile = Model(cat, gal_prof, psf_prof, SCA, filter_, bpass, hlr, i_gal)

        gal_model, g1, g2 = profile.draw_galaxy()
        st_model = profile.draw_star()

        sca_center = Pointing(selected_dithers[0], SCA, filter_, stamp_size, exposures[0], random_angle=False).find_sca_center()
        offsets = []
        gals = []
        psfs = []
        skys = []
        for exp in range(len(exposures)): 
            gal_stamp=None
            psf_stamp=None

            pointing=Pointing(selected_dithers[exp], SCA, filter_, stamp_size, exposures[exp], random_angle=False)
            image=Image(i_gal, stamp_size, gal_model, st_model, pointing, sca_center)

            translation=False
            if translation==True:
                offset=image.translational_dithering()

            gal_stamp, psf_stamp, offset = image.draw_image(gal_model, st_model)
            gal_stamp, sky_stamp = image.add_noise(rng, gal_stamp)
            gal_stamp, psf_stamp = image.wcs_approx(gal_stamp, psf_stamp)

            offsets.append(offset)
            gals.append(gal_stamp)
            psfs.append(psf_stamp)
            skys.append(sky_stamp)
        res_tot = shape_measurement(cat, gals, psfs, skys, offsets, i_gal, g1, g2, hlr, shape, res_tot).get_coadd_shape()
    print(res_tot)
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
        dirr=output_name
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