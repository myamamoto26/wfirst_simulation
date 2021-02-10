import numpy as np
import sys, os, io
import math
import galsim as galsim
import galsim.roman as wfirst
import fitsio as fio
import ngmix
from ngmix.jacobian import Jacobian
from ngmix.observation import Observation, ObsList, MultiBandObsList,make_kobs
from ngmix.galsimfit import GalsimRunner,GalsimSimple,GalsimTemplateFluxFitter
from ngmix.guessers import R50FluxGuesser
from ngmix.bootstrap import PSFRunner
from ngmix import priors, joint_prior
import meds
import psc
from skimage.measure import block_reduce

def get_flux(obs_list):
    flux = 0.
    for obs in obs_list:
        flux += obs.image.sum()
    flux /= len(obs_list)
    if flux<0:
        flux = 10.
    return flux

def get_psf_SCA(filter_):
    all_scas = np.array([i for i in range(1,19)])
    all_psfs = []
    for sca in all_scas:
        psf_sca = wfirst.getPSF(sca, 
                                filter_, 
                                SCA_pos=None, 
                                pupil_bin=4,
                                wavelength=wfirst.getBandpasses(AB_zeropoint=True)[filter_].effective_wavelength)
        all_psfs.append(psf_sca)
    return all_psfs

def measure_shape_metacal_multiband(obs_list, T, method='bootstrap', fracdev=None, use_e=None):

    metacal_pars = {'types': ['noshear', '1p', '1m', '2p', '2m'], 'psf': 'gauss'}
    #T = self.hlr
    pix_range = old_div(galsim.roman.pixel_scale,10.)
    e_range = 0.1
    fdev = 1.
    def pixe_guess(n):
        return 2.*n*np.random.random() - n

    cp = ngmix.priors.CenPrior(0.0, 0.0, galsim.roman.pixel_scale, galsim.roman.pixel_scale)
    gp = ngmix.priors.GPriorBA(0.3)
    hlrp = ngmix.priors.FlatPrior(1.0e-4, 1.0e2)
    fracdevp = ngmix.priors.Normal(0.5, 0.1, bounds=[0., 1.])
    fluxp = [ngmix.priors.FlatPrior(0, 1.0e5),ngmix.priors.FlatPrior(0, 1.0e5),ngmix.priors.FlatPrior(0, 1.0e5)]

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

def get_exp_list_coadd(m,i,m2=None):

    def make_jacobian(dudx,dudy,dvdx,dvdy,x,y):
        j = galsim.JacobianWCS(dudx, dudy, dvdx, dvdy)
        return j.withOrigin(galsim.PositionD(x,y))

    oversample = 4
    #def psf_offset(i,j,star_):
    m3=[0]
    #relative_offset=[0]
    for jj,psf_ in enumerate(m2): # m2 has psfs for each observation. 
        if jj==0:
            continue
        gal_stamp_center_row=m['orig_start_row'][i][jj] + m['box_size'][i]/2 # m['box_size'] is the galaxy stamp size. 
        gal_stamp_center_col=m['orig_start_col'][i][jj] + m['box_size'][i]/2 # m['orig_start_row/col'] is in SCA coordinates. 
        psf_stamp_size=32*oversample
        
        # Make the bounds for the psf stamp. 
        b = galsim.BoundsI( xmin=(m['orig_start_col'][i][jj]+(m['box_size'][i]-32)/2.)*oversample, 
                            xmax=(m['orig_start_col'][i][jj]+m['box_size'][i]-(m['box_size'][i]-32)/2.)*oversample - 1,
                            ymin=(m['orig_start_row'][i][jj]+(m['box_size'][i]-32)/2.)*oversample,
                            ymax=(m['orig_start_row'][i][jj]+m['box_size'][i]-(m['box_size'][i]-32)/2.)*oversample - 1)
        
        # Make wcs for oversampled psf. 
        wcs_ = make_jacobian(m.get_jacobian(i,jj)['dudcol']/oversample,
                            m.get_jacobian(i,jj)['dudrow']/oversample,
                            m.get_jacobian(i,jj)['dvdcol']/oversample,
                            m.get_jacobian(i,jj)['dvdrow']/oversample,
                            m['orig_col'][i][jj]*oversample,
                            m['orig_row'][i][jj]*oversample) 
        # Taken from galsim/roman_psfs.py line 266. Update each psf to an object-specific psf using the wcs. 
        scale = galsim.PixelScale(wfirst.pixel_scale/oversample)
        psf_ = wcs_.toWorld(scale.toImage(psf_), image_pos=galsim.PositionD(wfirst.n_pix/2, wfirst.n_pix/2))
        
        # Convolve with the star model and get the psf stamp. 
        #st_model = galsim.DeltaFunction(flux=1.)
        #st_model = st_model.evaluateAtWavelength(wfirst.getBandpasses(AB_zeropoint=True)['H158'].effective_wavelength)
        #st_model = st_model.withFlux(1.)
        #st_model = galsim.Convolve(st_model, psf_)
        psf_ = galsim.Convolve(psf_, galsim.Pixel(wfirst.pixel_scale))
        psf_stamp = galsim.Image(b, wcs=wcs_) 

        # Galaxy is being drawn with some subpixel offsets, so we apply the offsets when drawing the psf too. 
        offset_x = m['orig_col'][i][jj] - gal_stamp_center_col 
        offset_y = m['orig_row'][i][jj] - gal_stamp_center_row 
        offset = galsim.PositionD(offset_x, offset_y)
        ## not working -> st_model.drawImage(image=psf_stamp, offset=offset) ## 
        psf_.drawImage(image=psf_stamp, offset=offset, method='no_pixel') # We're not sure if we should use method='no_pixel' here. 
        m3.append(psf_stamp.array)

    obs_list=ObsList()
    psf_list=ObsList()

    included = []
    w        = []
    # For each of these objects create an observation
    for j in range(m['ncutout'][i]):
        if j==0:
            continue
        # if j>1:
        #     continue
        im = m.get_cutout(i, j, type='image')
        weight = m.get_cutout(i, j, type='weight')

        im_psf = m3[j] 
        im_psf2 = im_psf 
        if np.sum(im)==0.:
            print(local_meds, i, j, np.sum(im))
            print('no flux in image ',i,j)
            continue

        jacob = m.get_jacobian(i, j)
        # Get a galaxy jacobian. 
        gal_jacob=Jacobian(
            row=(m['orig_row'][i][j]-m['orig_start_row'][i][j]),
            col=(m['orig_col'][i][j]-m['orig_start_col'][i][j]),
            dvdrow=jacob['dvdrow'],
            dvdcol=jacob['dvdcol'],
            dudrow=jacob['dudrow'],
            dudcol=jacob['dudcol']) 

        psf_center = (32/2.)+0.5 
        # Get a oversampled psf jacobian. 
        psf_jacob2=Jacobian(
            row=(m['orig_row'][i][j]-m['orig_start_row'][i][j]-(m['box_size'][i]-32)/2.)*oversample,
            col=(m['orig_col'][i][j]-m['orig_start_col'][i][j]-(m['box_size'][i]-32)/2.)*oversample, 
            dvdrow=jacob['dvdrow']/oversample,
            dvdcol=jacob['dvdcol']/oversample,
            dudrow=jacob['dudrow']/oversample,
            dudcol=jacob['dudcol']/oversample)

        # Create an obs for each cutout
        mask = np.where(weight!=0)
        if 1.*len(weight[mask])/np.product(np.shape(weight))<0.8:
            continue

        w.append(np.mean(weight[mask]))
        noise = np.ones_like(weight)/w[-1]

        psf_obs = Observation(im_psf, jacobian=gal_jacob, meta={'offset_pixels':None,'file_id':None})
        psf_obs2 = Observation(im_psf2, jacobian=psf_jacob2, meta={'offset_pixels':None,'file_id':None})
        #obs = Observation(im, weight=weight, jacobian=gal_jacob, psf=psf_obs, meta={'offset_pixels':None,'file_id':None})
        # oversampled PSF
        obs = Observation(im, weight=weight, jacobian=gal_jacob, psf=psf_obs2, meta={'offset_pixels':None,'file_id':None})
        obs.set_noise(noise)

        obs_list.append(obs)
        psf_list.append(psf_obs2)
        included.append(j)

    return obs_list,psf_list,np.array(included)-1,np.array(w)


local_Hmeds = './fiducial_H158_2285117.fits'
local_Jmeds = './fiducial_J129_2285117.fits'
local_Fmeds = './fiducial_F184_2285117.fits'
truth = fio.FITS('/hpc/group/cosmology/phy-lsst/my137/roman_H158/g1002/truth/fiducial_lensing_galaxia_g1002_truth_gal.fits')[-1]
m_H158  = meds.MEDS(local_Hmeds)
m_J129  = meds.MEDS(local_Jmeds)
m_F184  = meds.MEDS(local_Fmeds)
indices_H = np.arange(len(m_H158['number'][:]))
indices_J = np.arange(len(m_J129['number'][:]))
indices_F = np.arange(len(m_F184['number'][:]))
roman_H158_psfs = get_psf_SCA('H158')
roman_J129_psfs = get_psf_SCA('J129')
roman_F184_psfs = get_psf_SCA('F184')
oversample = 4
for i,ii in enumerate(indices_H): # looping through all the objects in meds file. 
    if i%100==0:
        print('object number ',i)
    ind = m_H158['number'][ii]
    t   = truth[ind]
    sca_Hlist = m_H158[ii]['sca'] # List of SCAs for the same object in multiple observations. 
    sca_Jlist = m_J129[ii]['sca']
    sca_Flist = m_F184[ii]['sca']
    m2_H158_coadd = [roman_H158_psfs[j-1] for j in sca_Hlist[:m_H158['ncutout'][i]]]
    m2_J129_coadd = [roman_J129_psfs[j-1] for j in sca_Jlist[:m_J129['ncutout'][i]]]
    m2_F184_coadd = [roman_F184_psfs[j-1] for j in sca_Flist[:m_F184['ncutout'][i]]]

    obs_Hlist,psf_Hlist,included_H,w_H = get_exp_list_coadd(m_H158,ii,m2=m2_H158_coadd)
    coadd_H            = psc.Coadder(obs_Hlist,flat_wcs=True).coadd_obs
    coadd_H.psf.image[coadd_H.psf.image<0] = 0 # set negative pixels to zero. 
    coadd_H.set_meta({'offset_pixels':None,'file_id':None})

    obs_Jlist,psf_Jlist,included_J,w_J = get_exp_list_coadd(m_J129,ii,m2=m2_J129_coadd)
    coadd_J            = psc.Coadder(obs_Jlist,flat_wcs=True).coadd_obs
    coadd_J.psf.image[coadd_J.psf.image<0] = 0 # set negative pixels to zero. 
    coadd_J.set_meta({'offset_pixels':None,'file_id':None})

    obs_Flist,psf_Flist,included_F,w_F = get_exp_list_coadd(m_F184,ii,m2=m2_F184_coadd)
    coadd_F            = psc.Coadder(obs_Flist,flat_wcs=True).coadd_obs
    coadd_F.psf.image[coadd_F.psf.image<0] = 0 # set negative pixels to zero. 
    coadd_F.set_meta({'offset_pixels':None,'file_id':None})

    coadd = [coadd_H, coadd_J, coadd_F]
    mb_obs_list = MultiBandObsList()
    for band in range(3):
        obs_list = ObsList()
        new_coadd_psf_block = block_reduce(coadd[band].psf.image, block_size=(4,4), func=np.sum)
        new_coadd_psf_jacob = Jacobian( row=(coadd[band].psf.jacobian.row0/oversample),
                                        col=(coadd[band].psf.jacobian.col0/oversample), 
                                        dvdrow=(coadd[band].psf.jacobian.dvdrow*oversample),
                                        dvdcol=(coadd[band].psf.jacobian.dvdcol*oversample),
                                        dudrow=(coadd[band].psf.jacobian.dudrow*oversample),
                                        dudcol=(coadd[band].psf.jacobian.dudcol*oversample))
        coadd_psf_obs = Observation(new_coadd_psf_block, jacobian=new_coadd_psf_jacob, meta={'offset_pixels':None,'file_id':None})
        coadd[band].psf = coadd_psf_obs
        obs_list.append(coadd[band])
        mb_obs_list.append(obs_list)

    res_ = measure_shape_metacal_multiband(mb_obs_list, t['size'], method='bootstrap', fracdev=t['bflux'],use_e=[t['int_e1'],t['int_e2']])

print(res_['noshear'].dtype.names)
print('done')