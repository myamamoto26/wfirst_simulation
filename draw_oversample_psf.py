import numpy as np
import sys, os, io
import math
import galsim as galsim
import galsim.roman as wfirst
import ngmix
from ngmix.jacobian import Jacobian
from ngmix.observation import Observation, ObsList, MultiBandObsList,make_kobs
import meds

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

    oversample = 1
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
        scale = galsim.PixelScale(wfirst.pixel_scale)
        psf_ = wcs_.toWorld(scale.toImage(psf_), image_pos=galsim.PositionD(wfirst.n_pix/2, wfirst.n_pix/2))
        
        # Convolve with the star model and get the psf stamp. 
        st_model = galsim.DeltaFunction(flux=1.)
        st_model = st_model.evaluateAtWavelength(wfirst.getBandpasses(AB_zeropoint=True)['H158'].effective_wavelength)
        st_model = st_model.withFlux(1.)
        st_model = galsim.Convolve(st_model, psf_)
        #psf_ = galsim.Convolve(psf_, galsim.Pixel(wfirst.pixel_scale))
        psf_stamp = galsim.Image(b, wcs=wcs_) 

        # Galaxy is being drawn with some subpixel offsets, so we apply the offsets when drawing the psf too. 
        offset_x = m['orig_col'][i][jj] - gal_stamp_center_col 
        offset_y = m['orig_row'][i][jj] - gal_stamp_center_row 
        offset = galsim.PositionD(offset_x, offset_y)
        ## not working -> st_model.drawImage(image=psf_stamp, offset=offset) ## 
        st_model.drawImage(image=psf_stamp, offset=offset)#, method='no_pixel') # We're not sure if we should use method='no_pixel' here. 
        m3.append(psf_stamp.array)

    return m3

local_meds = './fiducial_H158_2285117.fits'
m  = meds.MEDS(local_meds)
indices = np.arange(len(m['number'][:]))
roman_psfs = get_psf_SCA('H158')
for i,ii in enumerate(indices): # looping through all the objects in meds file. 
    if i%100==0:
        print('object number ',i)
    ind = m['number'][ii]
    sca_list = m[ii]['sca'] # List of SCAs for the same object in multiple observations. 
    m2_coadd = [roman_psfs[j-1] for j in sca_list[:m['ncutout'][i]]]
    m3 = get_exp_list_coadd(m,ii,m2=m2_coadd)
    if i==50:
        np.savetxt('/hpc/group/cosmology/masaya/wfirst_simulation/test_star1'+str(i)+'.txt', m3[1])
    if i>=100:
        break