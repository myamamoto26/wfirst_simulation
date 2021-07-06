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
from past.utils import old_div
import pandas as pd
from des_analysis import analyze_gamma_obs

def get_flux(obs_list):
    flux = 0.
    for obs in obs_list:
        flux += obs.image.sum()
    flux /= len(obs_list)
    if flux<0:
        flux = 10.
    return flux

def get_snr(obs_list):
    
    for i in range(len(obs_list)):
        obs = obs_list[i]
        if i == 0:
            s2n = (obs.image * obs.weight)/np.sqrt((obs.weight**2).sum())
        else:
            s2n += obs.image * obs.weight/np.sqrt((obs.weight**2).sum())
    return s2n.sum()

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

def measure_shape_metacal(obs_list, T, method='bootstrap', fracdev=None, use_e=None):

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
    try:
        boot.fit_metacal(psf_model, gal_model, max_pars, Tguess, prior=prior, ntry=ntry, metacal_pars=metacal_pars) 
        res_ = boot.get_metacal_result()
    except (ngmix.gexceptions.BootGalFailure, ngmix.gexceptions.BootPSFFailure):
        print('it failed')
        res_ = 0

    return res_

def measure_shape_metacal_multiband(obs_list, T, method='bootstrap', fracdev=None, use_e=None):

    metacal_pars = {'types': ['noshear', '1p', '1m', '2p', '2m'], 'psf': 'fitgauss'}
    #T = self.hlr
    pix_range = old_div(galsim.roman.pixel_scale,10.)
    e_range = 0.1
    fdev = 1.
    def pixe_guess(n):
        return 2.*n*np.random.random() - n

    cp = ngmix.priors.CenPrior(0.0, 0.0, galsim.roman.pixel_scale, galsim.roman.pixel_scale)
    gp = ngmix.priors.GPriorBA(0.3)
    hlrp = ngmix.priors.FlatPrior(1.0e-5, 1.0e4)
    fracdevp = ngmix.priors.Normal(0.5, 0.1, bounds=[0., 1.])
    fluxp = [ngmix.priors.FlatPrior(-1.0e3, 1.0e6),ngmix.priors.FlatPrior(-1.0e3, 1.0e6),ngmix.priors.FlatPrior(-1.0e3, 1.0e6)]

    prior = joint_prior.PriorSimpleSep(cp, gp, hlrp, fluxp)
    guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,500.])

    boot = ngmix.bootstrap.MaxMetacalBootstrapper(obs_list)
    psf_model = "gauss"
    gal_model = "gauss"

    lm_pars={'maxfev':2000, 'xtol':5.0e-5, 'ftol':5.0e-5}
    max_pars={'method': 'lm', 'lm_pars':lm_pars}

    Tguess=T**2/(2*np.log(2))
    ntry=2
    try:
        boot.fit_metacal(psf_model, gal_model, max_pars, Tguess, prior=prior, ntry=ntry, metacal_pars=metacal_pars) 
        res_ = boot.get_metacal_result()
    except (ngmix.gexceptions.BootGalFailure, ngmix.gexceptions.BootPSFFailure):
        print('it failed')
        res_ = 0

    return res_

def get_exp_list_coadd(m,i,oversample,m2=None):

    def make_jacobian(dudx,dudy,dvdx,dvdy,x,y):
        j = galsim.JacobianWCS(dudx, dudy, dvdx, dvdy)
        return j.withOrigin(galsim.PositionD(x,y))

    #def psf_offset(i,j,star_):
    m3=[0]
    #relative_offset=[0]
    for jj,psf_ in enumerate(m2): # m2 has psfs for each observation. 
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
        scale = galsim.PixelScale(wfirst.pixel_scale/oversample)
        psf_ = wcs_.toWorld(scale.toImage(psf_), image_pos=galsim.PositionD(wfirst.n_pix/2, wfirst.n_pix/2))
        psf_ = galsim.Convolve(psf_, galsim.Pixel(wfirst.pixel_scale))
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

def search_se_snr():
    local_Hmeds = './fiducial_H158_2285117.fits'
    truth = fio.FITS('/hpc/group/cosmology/phy-lsst/my137/roman_H158_final/g1002/truth/fiducial_lensing_galaxia_g1002_truth_gal.fits')[-1]
    m_H158  = meds.MEDS(local_Hmeds)
    indices_H = np.arange(len(m_H158['number'][:]))
    roman_H158_psfs = get_psf_SCA('H158')
    oversample = 1
    metacal_keys=['noshear', '1p', '1m', '2p', '2m']
    snr=np.zeros(len(m_H158['number'][:]),dtype=[('ind',int), ('snr',float)])
    for i,ii in enumerate(indices_H): # looping through all the objects in meds file. 
        if i%100==0:
            print('object number ',i)
        ind = m_H158['number'][ii]
        t   = truth[ind]
        sca_Hlist = m_H158[ii]['sca'] # List of SCAs for the same object in multiple observations. 
        m2_H158_coadd = [roman_H158_psfs[j-1] for j in sca_Hlist[:m_H158['ncutout'][i]]]

        obs_Hlist,psf_Hlist,included_H,w_H = get_exp_list_coadd(m_H158,ii,oversample,m2=m2_H158_coadd)
        res_ = measure_shape_metacal(obs_Hlist, t['size'], method='bootstrap', fracdev=t['bflux'],use_e=[t['int_e1'],t['int_e2']])
        snr['ind'][i] = i # different from ind defined. 
        snr['snr'][i] = res_['noshear']['s2n']
    np.savetxt('snr_example.txt', snr)
    print(min(snr['snr']), snr['ind'][snr['snr'].index(min(snr['snr']))])
    print(max(snr['snr']), snr['ind'][snr['snr'].index(max(snr['snr']))])

def single_vs_coadd_images():
    local_Hmeds = './fiducial_H158_2285117.fits'
    truth = fio.FITS('/hpc/group/cosmology/phy-lsst/my137/roman_H158/g1002/truth/fiducial_lensing_galaxia_g1002_truth_gal.fits')[-1]
    m_H158  = meds.MEDS(local_Hmeds)
    indices_H = np.arange(len(m_H158['number'][:]))
    roman_H158_psfs = get_psf_SCA('H158')
    oversample = 4
    metacal_keys=['noshear', '1p', '1m', '2p', '2m']
    res_noshear=np.zeros(len(m_H158['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('flags',int),('int_e1',float), ('int_e2',float),('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_hlr',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])
    res_1p=np.zeros(len(m_H158['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('flags',int),('int_e1',float), ('int_e2',float),('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_hlr',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])
    res_1m=np.zeros(len(m_H158['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('flags',int),('int_e1',float), ('int_e2',float),('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_hlr',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])
    res_2p=np.zeros(len(m_H158['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('flags',int),('int_e1',float), ('int_e2',float),('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_hlr',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])
    res_2m=np.zeros(len(m_H158['number'][:]),dtype=[('ind',int), ('ra',float), ('dec',float), ('flags',int),('int_e1',float), ('int_e2',float),('coadd_px',float), ('coadd_py',float), ('coadd_flux',float), ('coadd_snr',float), ('coadd_e1',float), ('coadd_e2',float), ('coadd_hlr',float),('coadd_psf_e1',float), ('coadd_psf_e2',float), ('coadd_psf_T',float)])

    res_tot=[res_noshear, res_1p, res_1m, res_2p, res_2m]
    for i,ii in enumerate(indices_H): # looping through all the objects in meds file. 
        if i%100==0:
            print('object number ',i)
        if i not in [1,600]:
            continue
        ind = m_H158['number'][ii]
        t   = truth[ind]
        sca_Hlist = m_H158[ii]['sca'] # List of SCAs for the same object in multiple observations. 
        m2_H158_coadd = [roman_H158_psfs[j-1] for j in sca_Hlist[:m_H158['ncutout'][i]]]

        obs_Hlist,psf_Hlist,included_H,w_H = get_exp_list_coadd(m_H158,ii,oversample,m2=m2_H158_coadd)
        s2n_test = get_snr(obs_Hlist)
        if i in [1,600]: #in [ 309,  444,  622,  644,  854, 1070, 1282, 1529]:
            for l in range(len(obs_Hlist)):
                #print(i, obs_Hlist[l].jacobian, obs_Hlist[l].psf.jacobian)
                print(i, obs_Hlist[l].weight)
                # np.savetxt('/hpc/group/cosmology/masaya/wfirst_simulation/paper/single_image_oversample4_08scaling_'+str(i)+'_'+str(l)+'.txt', obs_Hlist[l].image)
            # np.savetxt('/hpc/group/cosmology/masaya/wfirst_simulation/paper/single_psf_oversample4_08scaling_'+str(i)+'.txt', obs_Hlist[0].psf.image)
        coadd_H            = psc.Coadder(obs_Hlist,flat_wcs=True).coadd_obs
        coadd_H.psf.image[coadd_H.psf.image<0] = 0 # set negative pixels to zero. 
        coadd_H.set_meta({'offset_pixels':None,'file_id':None})

        obs_list = ObsList()
        if oversample == 4:
            new_coadd_psf_block = block_reduce(coadd_H.psf.image, block_size=(4,4), func=np.sum)
            new_coadd_psf_jacob = Jacobian( row=15.5,
                                            col=15.5, 
                                            dvdrow=(coadd_H.psf.jacobian.dvdrow*oversample),
                                            dvdcol=(coadd_H.psf.jacobian.dvdcol*oversample),
                                            dudrow=(coadd_H.psf.jacobian.dudrow*oversample),
                                            dudcol=(coadd_H.psf.jacobian.dudcol*oversample))
            coadd_psf_obs = Observation(new_coadd_psf_block, jacobian=new_coadd_psf_jacob, meta={'offset_pixels':None,'file_id':None})
            coadd_H.psf = coadd_psf_obs
        obs_list.append(coadd_H)
        s2n_coadd = get_snr(obs_list)
        if i in [1,600]:
            print(i, coadd_H.weight)
            #np.savetxt('/hpc/group/cosmology/masaya/wfirst_simulation/paper/coadd_image_oversample4_08scaling_'+str(i)+'.txt', coadd_H.image)
            # np.savetxt('/hpc/group/cosmology/masaya/wfirst_simulation/paper/coadd_weight_oversample4_08scaling_'+str(i)+'.txt', coadd_H.weight)
            #np.savetxt('/hpc/group/cosmology/masaya/wfirst_simulation/paper/coadd_psf_over-downsample4_08scaling_'+str(i)+'.txt', coadd_H.psf.image)

        iteration=0
        for key in metacal_keys:
            res_tot[iteration]['ind'][i]                       = ind
            res_tot[iteration]['ra'][i]                        = t['ra']
            res_tot[iteration]['dec'][i]                       = t['dec']
            res_tot[iteration]['int_e1'][i]                    = t['int_e1']
            res_tot[iteration]['int_e2'][i]                    = t['int_e2']

            iteration+=1
        
        res_ = measure_shape_metacal(obs_list, t['size'], method='bootstrap', fracdev=t['bflux'],use_e=[t['int_e1'],t['int_e2']])
        print('signal to noise test', i, s2n_test, s2n_coadd, res_['noshear']['s2n_r'])
        # if res_[key]['s2n'] > 1e7:
        #     print('coadd snr', res_[key]['s2n'])
        #     np.savetxt('large_coadd_snr_image.txt', coadd_H.image)
        iteration=0
        for key in metacal_keys:
            if res_==0:
                res_tot[iteration]['ind'][i]                       = 0
            elif res_[key]['flags']==0:
                res_tot[iteration]['coadd_px'][i]                  = res_[key]['pars'][0]
                res_tot[iteration]['coadd_py'][i]                  = res_[key]['pars'][1]
                res_tot[iteration]['coadd_snr'][i]                 = res_[key]['s2n']
                res_tot[iteration]['coadd_e1'][i]                  = res_[key]['pars'][2]
                res_tot[iteration]['coadd_e2'][i]                  = res_[key]['pars'][3]
                res_tot[iteration]['coadd_hlr'][i]                 = res_[key]['pars'][4]
            iteration+=1

    mask=res_tot[0]['ind']!=0
    print(len(res_tot[0]), len(res_tot[0][mask]))
    #print(res_['noshear'].dtype.names)
    print('done')

def roman_psf_rotation():
    b = galsim.BoundsI(1,32,1,32)
    WCS1 = wfirst.getWCS(world_pos  = galsim.CelestialCoord(ra=100*galsim.degrees, dec=-10*galsim.degrees), PA = 0.*galsim.radians)[1]
    WCS2 = wfirst.getWCS(world_pos  = galsim.CelestialCoord(ra=100*galsim.degrees, dec=-10*galsim.degrees), PA = 30.*galsim.radians)[1]
    psf1 = wfirst.getPSF(1, 'H158', wcs=WCS1, pupil_bin=4, wavelength=wfirst.getBandpasses(AB_zeropoint=True)['H158'].effective_wavelength)
    psf2 = wfirst.getPSF(1, 'H158', wcs=WCS2, pupil_bin=4, wavelength=wfirst.getBandpasses(AB_zeropoint=True)['H158'].effective_wavelength)

    #star_model1 = galsim.DeltaFunction(flux=1.)
    #star_model2 = galsim.DeltaFunction(flux=1.)
    star_stamp1 = galsim.Image(b, scale=wfirst.pixel_scale)
    star_stamp2 = galsim.Image(b, scale=wfirst.pixel_scale)
    #star_model1 = galsim.Convolve(star_model1, psf1)
    #star_model2 = galsim.Convolve(star_model2, psf2)
    psf1.drawImage(image=star_stamp1)
    psf2.drawImage(image=star_stamp2)

    np.savetxt("/hpc/group/cosmology/masaya/wfirst_simulation/paper/roman_psf_PA0b.txt", star_stamp1.array)
    np.savetxt("/hpc/group/cosmology/masaya/wfirst_simulation/paper/roman_psf_PA30b.txt", star_stamp2.array)

def mcal_catalog_properties(filter_, coadd_):

    folder = os.path.join("/hpc/group/cosmology/phy-lsst/my137", filter_)

    ## g1 positive sim. (get SNR and magnitude from this. )
    # mcal_noshear = fio.FITS(os.path.join(folder, "g1002/ngmix/coadd_multiband/fiducial_H158_mcal_noshear.fits"))[-1].read()
    # mcal_1p = fio.FITS(os.path.join(folder, "g1002/ngmix/coadd_multiband/fiducial_H158_mcal_1p.fits"))[-1].read()
    # mcal_1m = fio.FITS(os.path.join(folder, "g1002/ngmix/coadd_multiband/fiducial_H158_mcal_1m.fits"))[-1].read()
    # mcal_2p = fio.FITS(os.path.join(folder, "g1002/ngmix/coadd_multiband/fiducial_H158_mcal_2p.fits"))[-1].read()
    # mcal_2m = fio.FITS(os.path.join(folder, "g1002/ngmix/coadd_multiband/fiducial_H158_mcal_2m.fits"))[-1].read()

    # mask = (mcal_noshear['flags']==0)
    # mcal_noshear = mcal_noshear[mask]

    # properties = np.zeros((len(mcal_noshear),5))
    single_filter = False
    if not single_filter:
        columns = ['g1_true', 'g2_true', 'g1_obs', 'g2_obs', 'coadd_psf_e1', 'coadd_psf_e2', 'coadd_psf_T', 'coadd_snr', 'coadd_hlr']
    else:
        columns = ['g1_true', 'g2_true', 'g1_obs', 'g2_obs', 'snr', 'hlr', 'mag'] # 'psf_e1', 'psf_e2', 'psf_T']

    # properties[:,0] = mcal_noshear['coadd_snr']
    # properties[:,1] = mcal_noshear['coadd_hlr']
    # properties[:,2] = mcal_noshear['coadd_psf_e1']
    # properties[:,3] = mcal_noshear['coadd_psf_e2']
    # properties[:,4] = mcal_noshear['coadd_psf_T']

    total_obj = []
    # total_obj = np.append(len(mcal_noshear))
    # properties = np.zeros((np.sum(total_obj), 7))
    start = 0
    sets = ['g1002', 'g1n002', 'g2002', 'g2n002']
    noshear = []
    shear1p = []
    shear1m = []
    shear2p = []
    shear2m = []
    for i in range(4): # four sets of sim. 
        mcal_noshear = fio.FITS(os.path.join(folder, sets[i]+"/ngmix/"+coadd_+"/fiducial_H158_mcal_noshear.fits"))[-1].read()
        mcal_1p = fio.FITS(os.path.join(folder, sets[i]+"/ngmix/"+coadd_+"/fiducial_H158_mcal_1p.fits"))[-1].read()
        mcal_1m = fio.FITS(os.path.join(folder, sets[i]+"/ngmix/"+coadd_+"/fiducial_H158_mcal_1m.fits"))[-1].read()
        mcal_2p = fio.FITS(os.path.join(folder, sets[i]+"/ngmix/"+coadd_+"/fiducial_H158_mcal_2p.fits"))[-1].read()
        mcal_2m = fio.FITS(os.path.join(folder, sets[i]+"/ngmix/"+coadd_+"/fiducial_H158_mcal_2m.fits"))[-1].read()

        mask = (mcal_noshear['flags']==0) & (mcal_noshear['ind']!=0)
        noshear.append(mcal_noshear[mask])
        shear1p.append(mcal_1p[mask])
        shear1m.append(mcal_1m[mask])
        shear2p.append(mcal_2p[mask])
        shear2m.append(mcal_2m[mask])

    a,c00,c1 = np.intersect1d(noshear[0]['ind'], noshear[1]['ind'], return_indices=True)
    b,c01,c2 = np.intersect1d(noshear[0]['ind'][c00], noshear[2]['ind'], return_indices=True)
    c,c02,c3 = np.intersect1d(noshear[0]['ind'][c00][c01], noshear[3]['ind'], return_indices=True)
    tmp_ind = noshear[0]['ind'][c00][c01][c02]
    properties = np.zeros((4*len(noshear[0][np.isin(noshear[0]['ind'],tmp_ind)]), len(columns)))
    for run in range(4):
        new = noshear[run][np.isin(noshear[run]['ind'] ,tmp_ind)]
        new1p = shear1p[run][np.isin(noshear[run]['ind'] ,tmp_ind)]
        new1m = shear1m[run][np.isin(noshear[run]['ind'] ,tmp_ind)]
        new2p = shear2p[run][np.isin(noshear[run]['ind'] ,tmp_ind)]
        new2m = shear2m[run][np.isin(noshear[run]['ind'] ,tmp_ind)]

        total_obj = len(new)
        print('object number', total_obj)

        if single_filter:
            gamma1_t,gamma2_t,gamma1_o,gamma2_o,noshear1,noshear2 = analyze_gamma_obs(new,new1p,new1m,new2p,new2m,coadd_=False)
            properties[start:start+total_obj, 0] = gamma1_t
            properties[start:start+total_obj, 1] = gamma2_t
            properties[start:start+total_obj, 2] = gamma1_o
            properties[start:start+total_obj, 3] = gamma2_o
            #properties[start:start+total_obj, 4] = new['coadd_psf_e1']
            #properties[start:start+total_obj, 5] = new['coadd_psf_e2']
            #properties[start:start+total_obj, 6] = new['coadd_psf_T']
            properties[start:start+total_obj, 4] = new['snr']
            properties[start:start+total_obj, 5] = new['hlr']
            properties[start:start+total_obj, 6] = new['mag_'+filter_[-4:]]
        else:
            gamma1_t,gamma2_t,gamma1_o,gamma2_o,noshear1,noshear2 = analyze_gamma_obs(new,new1p,new1m,new2p,new2m,coadd_=True)
            properties[start:start+total_obj, 0] = gamma1_t
            properties[start:start+total_obj, 1] = gamma2_t
            properties[start:start+total_obj, 2] = gamma1_o
            properties[start:start+total_obj, 3] = gamma2_o
            properties[start:start+total_obj, 4] = new['coadd_psf_e1']
            properties[start:start+total_obj, 5] = new['coadd_psf_e2']
            properties[start:start+total_obj, 6] = new['coadd_psf_T']
            properties[start:start+total_obj, 7] = new['coadd_snr']
            properties[start:start+total_obj, 8] = new['coadd_hlr']

        start += total_obj

    df = pd.DataFrame(data=properties, columns=columns)
    if not single_filter:
        df.to_csv(filter_+'_coadd_properties.csv', columns=columns)
    else:
        df.to_csv(filter_+'_single_properties.csv', columns=columns)

def make_multiband_coadd_stamp():

    oversample = 4
    H = './fiducial_H158_2285117.fits'
    J = './fiducial_J129_2285117.fits'
    F = './fiducial_F184_2285117.fits'
    truth = fio.FITS('/hpc/group/cosmology/phy-lsst/my137/roman_H158/g1002/truth/fiducial_lensing_galaxia_g1002_truth_gal.fits')[-1]
    m_H158  = meds.MEDS(H)
    m_J129  = meds.MEDS(J)
    m_F184  = meds.MEDS(F)
    indices_H = np.arange(len(m_H158['number'][:]))
    indices_J = np.arange(len(m_J129['number'][:]))
    indices_F = np.arange(len(m_F184['number'][:]))
    roman_H158_psfs = get_psf_SCA('H158')
    roman_J129_psfs = get_psf_SCA('J129')
    roman_F184_psfs = get_psf_SCA('F184')

    for i,ii in enumerate(indices_H):

        if i%100==0:
            print('made it to object',i)

        try_save = False

        ind = m_H158['number'][ii]
        t   = truth[ind]

        if (ind not in m_J129['number']) or (ind not in m_F184['number']):
            continue

        print(i)
        if i==13:
            sca_Hlist = m_H158[ii]['sca'] # List of SCAs for the same object in multiple observations. 
            ii_J = m_J129[m_J129['number']==ind]['id'][0]
            sca_Jlist = m_J129[ii_J]['sca']
            m2_H158_coadd = [roman_H158_psfs[j-1] for j in sca_Hlist[:m_H158['ncutout'][i]]]
            m2_J129_coadd = [roman_J129_psfs[j-1] for j in sca_Jlist[:m_J129['ncutout'][ii_J]]]
            ii_F = m_F184[m_F184['number']==ind]['id'][0]
            sca_Flist = m_F184[ii_F]['sca']
            m2_F184_coadd = [roman_F184_psfs[j-1] for j in sca_Flist[:m_F184['ncutout'][ii_F]]]

            obs_Hlist,psf_Hlist,included_H,w_H = get_exp_list_coadd(m_H158,ii,oversample,m2=m2_H158_coadd)
            obs_Jlist,psf_Jlist,included_J,w_J = get_exp_list_coadd(m_J129,ii_J,oversample,m2=m2_J129_coadd)
            obs_Flist,psf_Flist,included_F,w_F = get_exp_list_coadd(m_F184,ii_F,oversample,m2=m2_F184_coadd)
            # check if masking is less than 20%
            if len(obs_Hlist)==0 or len(obs_Jlist)==0 or len(obs_Flist)==0: 
                continue
        
            coadd_H            = psc.Coadder(obs_Hlist,flat_wcs=True).coadd_obs
            coadd_H.psf.image[coadd_H.psf.image<0] = 0 # set negative pixels to zero. 
            coadd_H.psf.set_meta({'offset_pixels':None,'file_id':None})
            coadd_H.set_meta({'offset_pixels':None,'file_id':None})
            
            coadd_J            = psc.Coadder(obs_Jlist,flat_wcs=True).coadd_obs
            coadd_J.psf.image[coadd_J.psf.image<0] = 0 # set negative pixels to zero. 
            coadd_J.psf.set_meta({'offset_pixels':None,'file_id':None})
            coadd_J.set_meta({'offset_pixels':None,'file_id':None})

            coadd_F            = psc.Coadder(obs_Flist,flat_wcs=True).coadd_obs
            coadd_F.psf.image[coadd_F.psf.image<0] = 0 # set negative pixels to zero. 
            coadd_F.psf.set_meta({'offset_pixels':None,'file_id':None})
            coadd_F.set_meta({'offset_pixels':None,'file_id':None})

            obs_list = ObsList()
            multiband = [coadd_H, coadd_J, coadd_F]
            for f in range(3):
                obs_list.append(multiband[f])
            multiband_coadd = psc.Coadder(obs_list,flat_wcs=True).coadd_obs

            np.savetxt('/hpc/group/cosmology/masaya/wfirst_simulation/paper/multiband_H_image.txt', coadd_H.image)
            np.savetxt('/hpc/group/cosmology/masaya/wfirst_simulation/paper/multiband_J_image.txt', coadd_J.image)
            np.savetxt('/hpc/group/cosmology/masaya/wfirst_simulation/paper/multiband_F_image.txt', coadd_F.image)
            np.savetxt('/hpc/group/cosmology/masaya/wfirst_simulation/paper/multiband_coadd_image.txt', multiband_coadd.image)
            np.savetxt('/hpc/group/cosmology/masaya/wfirst_simulation/paper/multiband_coadd_psf_image.txt', multiband_coadd.psf.image)
            exit()

def main(argv):
    single_vs_coadd_images()
    # mcal_catalog_properties(sys.argv[1], sys.argv[2])
    # make_multiband_coadd_stamp()

if __name__ == "__main__":
    main(sys.argv)