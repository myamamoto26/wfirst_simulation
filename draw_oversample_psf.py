import numpy as np
import sys, os, io
import math
import galsim as galsim
import galsim.roman as wfirst
import ngmix
from ngmix.jacobian import Jacobian
from ngmix.observation import Observation, ObsList, MultiBandObsList,make_kobs
import meds
import psc
from skimage.measure import block_reduce

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

local_meds = './fiducial_H158_2285117.fits'
m  = meds.MEDS(local_meds)
indices = np.arange(len(m['number'][:]))
roman_psfs = get_psf_SCA('H158')
oversample = 4
for i,ii in enumerate(indices): # looping through all the objects in meds file. 
    if i%100==0:
        print('object number ',i)
    ind = m['number'][ii]
    sca_list = m[ii]['sca'] # List of SCAs for the same object in multiple observations. 
    m2_coadd = [roman_psfs[j-1] for j in sca_list[:m['ncutout'][i]]]
    obs_list,psf_list,included,w = get_exp_list_coadd(m,ii,m2=m2_coadd)
    coadd            = psc.Coadder(obs_list,flat_wcs=True).coadd_obs
    coadd.psf.image[coadd.psf.image<0] = 0 # set negative pixels to zero. 
    coadd.set_meta({'offset_pixels':None,'file_id':None})

    if i==50:
        np.savetxt('/hpc/group/cosmology/masaya/wfirst_simulation/psf_oversample_coadd_'+str(i)+'.txt', coadd.psf.image)
    new_coadd_psf_block = block_reduce(coadd.psf.image, block_size=(4,4), func=np.sum)
    new_coadd_psf_jacob = Jacobian( row=(coadd.psf.jacobian.row0/oversample),
                                    col=(coadd.psf.jacobian.col0/oversample), 
                                    dvdrow=(coadd.psf.jacobian.dvdrow*oversample),
                                    dvdcol=(coadd.psf.jacobian.dvdcol*oversample),
                                    dudrow=(coadd.psf.jacobian.dudrow*oversample),
                                    dudcol=(coadd.psf.jacobian.dudcol*oversample))
    coadd_psf_obs = Observation(new_coadd_psf_block, jacobian=new_coadd_psf_jacob, meta={'offset_pixels':None,'file_id':None})
    coadd.psf = coadd_psf_obs
    if i==50:
        np.savetxt('/hpc/group/cosmology/masaya/wfirst_simulation/psf_oversample_ngmix_'+str(i)+'.txt', coadd.psf.image)
print('done')