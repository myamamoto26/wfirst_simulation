
from warnings import resetwarnings
import fitsio as fio
import numpy as np
import psc 
import sys, os, io
import galsim as galsim
import galsim.roman as wfirst
import galsim.config.process as process
import galsim.des as des
import ngmix
import pickle as pickle
import pickletools
from astropy.time import Time
from mpi4py import MPI
import cProfile, pstats, psutil
import glob
import meds
from ngmix.jacobian import Jacobian
from ngmix.observation import Observation, ObsList, MultiBandObsList,make_kobs

# make PSF
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

# create observation list
def get_exp_list_coadd(m,i,oversample,m2=None):

    def make_jacobian(dudx,dudy,dvdx,dvdy,x,y):
        j = galsim.JacobianWCS(dudx, dudy, dvdx, dvdy)
        return j.withOrigin(galsim.PositionD(x,y))

    m3=[0]
    for jj,psf_model in enumerate(m2): 
        # m2 contains 18 psfs that are centered at each SCA. Created at line 117. 
        # These PSFs are in image coordinates and have not rotated according to the wcs. These are merely templates. 
        # We want to rotate the PSF template according to the wcs, and oversample it.
        if jj==0:
            continue
        gal_stamp_center_row=m['orig_start_row'][i][jj] + m['box_size'][i]/2 - 0.5 # m['box_size'] is the galaxy stamp size. 
        gal_stamp_center_col=m['orig_start_col'][i][jj] + m['box_size'][i]/2 - 0.5 # m['orig_start_row/col'] is in SCA coordinates. 
        psf_stamp_size=32
        
        # Make the bounds for the psf stamp. 
        b = galsim.BoundsI( xmin=(m['orig_start_col'][i][jj]+(m['box_size'][i]-32)/2. - 1)*oversample+1, 
                            xmax=(m['orig_start_col'][i][jj]+(m['box_size'][i]-32)/2.+psf_stamp_size-1)*oversample,
                            ymin=(m['orig_start_row'][i][jj]+(m['box_size'][i]-32)/2. - 1)*oversample+1,
                            ymax=(m['orig_start_row'][i][jj]+(m['box_size'][i]-32)/2.+psf_stamp_size-1)*oversample)

        # Make wcs for oversampled psf. 
        wcs_ = make_jacobian(m.get_jacobian(i,jj)['dudcol']/oversample,
                             m.get_jacobian(i,jj)['dudrow']/oversample,
                             m.get_jacobian(i,jj)['dvdcol']/oversample,
                             m.get_jacobian(i,jj)['dvdrow']/oversample,
                             m['orig_col'][i][jj]*oversample,
                             m['orig_row'][i][jj]*oversample) 
        # Taken from galsim/roman_psfs.py line 266. Update each psf to an object-specific psf using the wcs. 
        # Apply WCS.
        # The PSF is in arcsec units, but oriented parallel to the image coordinates.
        # So to apply the right WCS, project to pixels using the Roman mean pixel_scale, then
        # project back to world coordinates with the provided wcs.
        scale = galsim.PixelScale(wfirst.pixel_scale/oversample)
        # Image coordinates to world coordinates. PSF models were drawn at the center of the SCA. 
        psf_ = wcs_.toWorld(scale.toImage(psf_model), image_pos=galsim.PositionD(wfirst.n_pix/2, wfirst.n_pix/2))
        # Convolve the psf with oversampled pixel scale. Note that we should convolve with galsim.Pixel(self.params['oversample']), not galsim.Pixel(1.0)
        psf_ = wcs_.toWorld(galsim.Convolve(wcs_.toImage(psf_), galsim.Pixel(oversample)))
        psf_stamp = galsim.Image(b, wcs=wcs_) 

        # Galaxy is being drawn with some subpixel offsets, so we apply the offsets when drawing the psf too. 
        offset_x = m['orig_col'][i][jj] - gal_stamp_center_col 
        offset_y = m['orig_row'][i][jj] - gal_stamp_center_row 
        offset = galsim.PositionD(offset_x, offset_y)
        psf_.drawImage(image=psf_stamp, offset=offset, method='no_pixel') 
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
            #print(local_meds, i, j, np.sum(im))
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
        if oversample==1:
            psf_jacob2=Jacobian(
                row=15.5 + (m['orig_row'][i][j]-m['orig_start_row'][i][j]+1-(m['box_size'][i]/2.+0.5))*oversample,
                col=15.5 + (m['orig_col'][i][j]-m['orig_start_col'][i][j]+1-(m['box_size'][i]/2.+0.5))*oversample, 
                dvdrow=jacob['dvdrow']/oversample,
                dvdcol=jacob['dvdcol']/oversample,
                dudrow=jacob['dudrow']/oversample,
                dudcol=jacob['dudcol']/oversample) 
        elif oversample==4:
            psf_jacob2=Jacobian(
                row=63.5 + (m['orig_row'][i][j]-m['orig_start_row'][i][j]+1-(m['box_size'][i]/2.+0.5))*oversample,
                col=63.5 + (m['orig_col'][i][j]-m['orig_start_col'][i][j]+1-(m['box_size'][i]/2.+0.5))*oversample, 
                dvdrow=jacob['dvdrow']/oversample,
                dvdcol=jacob['dvdcol']/oversample,
                dudrow=jacob['dudrow']/oversample,
                dudcol=jacob['dudcol']/oversample) 

        # Create an obs for each cutout
        mask = np.where(weight!=0)
        if 1.*len(weight[mask])/np.product(np.shape(weight))<0.8:
            continue

        # w.append(np.mean(weight[mask]))
        # noise = np.ones_like(weight)/w[-1]
        mask_zero = np.where(weight==0)
        noise = galsim.Image(np.ones_like(weight)/weight, scale=galsim.roman.pixel_scale)
        p_noise = galsim.PoissonNoise(galsim.BaseDeviate(1234), sky_level=0.)
        noise.array[mask_zero] = np.mean(weight[mask])
        noise.addNoise(p_noise)
        noise -= (1/np.mean(weight[mask]))

        psf_obs = Observation(im_psf, jacobian=gal_jacob, meta={'offset_pixels':None,'file_id':None})
        psf_obs2 = Observation(im_psf2, jacobian=psf_jacob2, meta={'offset_pixels':None,'file_id':None})
        #obs = Observation(im, weight=weight, jacobian=gal_jacob, psf=psf_obs, meta={'offset_pixels':None,'file_id':None})
        # oversampled PSF
        obs = Observation(im, weight=weight, jacobian=gal_jacob, psf=psf_obs2, meta={'offset_pixels':None,'file_id':None})
        obs.set_noise(noise.array)

        obs_list.append(obs)
        psf_list.append(psf_obs2)
        included.append(j)

    return obs_list,psf_list,np.array(included)-1,np.array(w)

# import meds file. 
local_Hmeds = './fiducial_H158_2285117.fits'
truth = fio.FITS('/hpc/group/cosmology/phy-lsst/my137/roman_H158/g1002/truth/fiducial_lensing_galaxia_g1002_truth_gal.fits')[-1]
m_H158  = meds.MEDS(local_Hmeds)
indices_H = np.arange(len(m_H158['number'][:]))
roman_H158_psfs = get_psf_SCA('H158')
oversample = 1

# make coadds and save object data. 
# randomly select 50 objects for each meds file. -> This will end up in 24,000 objects in total for 480 meds files. -> a rate of 1 PSF per 1 arcmin x 1 arcmin. 
rand_obj_list = np.random.choice(indices_H, size=1, replace=False)
for i,ii in enumerate(rand_obj_list): 

    ind = m_H158['number'][ii]
    t   = truth[ind]
    sca_Hlist = m_H158[ii]['sca'] # List of SCAs for the same object in multiple observations. 
    m2_H158_coadd = [roman_H158_psfs[j-1] for j in sca_Hlist[:m_H158['ncutout'][ii]]]

    obs_Hlist,psf_Hlist,included_H,w_H = get_exp_list_coadd(m_H158,ii,oversample,m2=m2_H158_coadd)
    res = np.zeros(1, dtype=[('ra', float), ('dec', float), ('mag', float), ('nexp_tot', int)])
    res['ra'][i]                        = t['ra']
    res['dec'][i]                       = t['dec']
    res['nexp_tot'][i]                  = m_H158['ncutout'][ii]-1

    # coadd images
    coadd_H            = psc.Coadder(obs_Hlist,flat_wcs=True).coadd_obs
    coadd_H.psf.image[coadd_H.psf.image<0] = 0 # set negative pixels to zero. 
    coadd_H.set_meta({'offset_pixels':None,'file_id':None})

    fits = fio.FITS('/hpc/group/cosmology/phy-lsst/public/psc_coadd_psf/test_'+str(ii)+'.fits','rw')
    fits.write(coadd_H.psf.image, header=1)
    fits.write(res, header=2)
    fits.close()
