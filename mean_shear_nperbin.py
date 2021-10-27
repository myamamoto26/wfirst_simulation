import numpy as np
import sys, os
import fitsio as fio
import pandas as pd
from des_analysis import analyze_gamma_obs
from esutil import stat
from matplotlib import pyplot as plt
import matplotlib

work = '/hpc/group/cosmology/phy-lsst/my137/roman_H158/'
work_out = '/hpc/group/cosmology/masaya/wfirst_simulation/paper/'
sims = ['g1002', 'g1n002', 'g2002', 'g2n002']

def hlr_to_T(d):
    # assuming galaxy profile is gaussian.
    return 2*(2*d/2.3548200450309493)**2

def mean_shear_nperbin(new, new1p, new1m, new2p, new2m, nperbin, par, coadd):
    
    if par=='size':
        x_ = hlr_to_T(new[par])
    else:
        x_ = new[par]
    hist = stat.histogram(x_, nperbin=nperbin, more=True)
    bin_num = len(hist['hist'])
    g_obs = np.zeros((2,bin_num))
    gerr_obs = np.zeros((2,bin_num))
    print(len(hist['low']), hist['mean'])
    for i in range(bin_num):
        bin_mask = (x_ > hist['low'][i]) & (x_ < hist['high'][i])
        gamma1_t,gamma2_t,gamma1_o,gamma2_o,noshear1,noshear2 = analyze_gamma_obs(new[bin_mask], new1p[bin_mask], new1m[bin_mask], new2p[bin_mask], new2m[bin_mask], coadd)
        g_obs[0,i] = np.mean(gamma1_o)
        g_obs[1,i] = np.mean(gamma2_o)
        gerr_obs[0,i] = np.std(gamma1_o)/np.sqrt(len(gamma1_o))
        gerr_obs[1,i] = np.std(gamma2_o)/np.sqrt(len(gamma2_o))

    return hist['mean'], g_obs, gerr_obs

start = 0
sets = ['g1002', 'g1n002', 'g2002', 'g2n002']
fig,axs = plt.subplots(2,4,figsize=(28,12),dpi=100,sharey=True)
matplotlib.rcParams.update({'font.size': 25})
for p in ['coadd', 'single', 'multiband']:
    noshear = []
    shear1p = []
    shear1m = []
    shear2p = []
    shear2m = []
    if p=='coadd':
        xax = ['coadd_snr', 'coadd_hlr', 'size', 'coadd_psf_T']
        coadd=True
        j = 'new_coadd_oversample'
    elif p=='single':
        xax = ['snr', 'hlr', 'size', 'psf_T']
        coadd=False
        j = 'new_single'
    elif p=='multiband':
        xax = ['coadd_snr', 'coadd_hlr', 'size', 'coadd_psf_T']
        coadd=True
        j = 'coadd_multiband_3filter_final'
    
    for i in range(4): # four sets of sim. 
        mcal_noshear = fio.FITS(os.path.join(work, sets[i]+"/ngmix/"+j+"/fiducial_H158_mcal_noshear.fits"))[-1].read()
        mcal_1p = fio.FITS(os.path.join(work, sets[i]+"/ngmix/"+j+"/fiducial_H158_mcal_1p.fits"))[-1].read()
        mcal_1m = fio.FITS(os.path.join(work, sets[i]+"/ngmix/"+j+"/fiducial_H158_mcal_1m.fits"))[-1].read()
        mcal_2p = fio.FITS(os.path.join(work, sets[i]+"/ngmix/"+j+"/fiducial_H158_mcal_2p.fits"))[-1].read()
        mcal_2m = fio.FITS(os.path.join(work, sets[i]+"/ngmix/"+j+"/fiducial_H158_mcal_2m.fits"))[-1].read()

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
    
    # g1=+0.02 run
    run = 0
    new = noshear[run][np.isin(noshear[run]['ind'] ,tmp_ind)]
    new1p = shear1p[run][np.isin(noshear[run]['ind'] ,tmp_ind)]
    new1m = shear1m[run][np.isin(noshear[run]['ind'] ,tmp_ind)]
    new2p = shear2p[run][np.isin(noshear[run]['ind'] ,tmp_ind)]
    new2m = shear2m[run][np.isin(noshear[run]['ind'] ,tmp_ind)]

    bin1_mean_snr, g1_obs_snr, g1err_obs_snr = mean_shear_nperbin(new, new1p, new1m, new2p, new2m, 50000, xax[0], coadd)
    bin1_mean_T, g1_obs_T, g1err_obs_T = mean_shear_nperbin(new, new1p, new1m, new2p, new2m, 50000, xax[1], coadd)
    bin1_mean_size, g1_obs_size, g1err_obs_size = mean_shear_nperbin(new, new1p, new1m, new2p, new2m, 50000, xax[2], coadd)
    bin1_mean_Tpsf, g1_obs_Tpsf, g1err_obs_Tpsf = mean_shear_nperbin(new, new1p, new1m, new2p, new2m, 50000, xax[3], coadd)

    # g2=+0.02 run
    run = 2
    new = noshear[run][np.isin(noshear[run]['ind'] ,tmp_ind)]
    new1p = shear1p[run][np.isin(noshear[run]['ind'] ,tmp_ind)]
    new1m = shear1m[run][np.isin(noshear[run]['ind'] ,tmp_ind)]
    new2p = shear2p[run][np.isin(noshear[run]['ind'] ,tmp_ind)]
    new2m = shear2m[run][np.isin(noshear[run]['ind'] ,tmp_ind)]

    bin2_mean_snr, g2_obs_snr, g2err_obs_snr = mean_shear_nperbin(new, new1p, new1m, new2p, new2m, 50000, xax[0], coadd)
    bin2_mean_T, g2_obs_T, g2err_obs_T = mean_shear_nperbin(new, new1p, new1m, new2p, new2m, 50000, xax[1], coadd)
    bin2_mean_size, g2_obs_size, g2err_obs_size = mean_shear_nperbin(new, new1p, new1m, new2p, new2m, 50000, xax[2], coadd)
    bin2_mean_Tpsf, g2_obs_Tpsf, g2err_obs_Tpsf = mean_shear_nperbin(new, new1p, new1m, new2p, new2m, 50000, xax[3], coadd)

    # gamma1_t,gamma2_t,gamma1_o,gamma2_o,noshear1,noshear2 = analyze_gamma_obs(new,new1p,new1m,new2p,new2m,coadd)

    axs[0].hlines(0.00, 0, bin1_mean_snr[len(bin1_mean_snr)-1],linestyles='dashed')
    axs[0].errorbar(bin1_mean_snr, g1_obs_snr[0,:]-0.02, yerr=g1err_obs_snr[0,:], fmt='o', fillstyle='none', label=p)
    axs[0].set_xlabel('S/N')
    axs[0].set_xscale('log')
    axs[0].set_ylabel(r'$<\Delta e_{1}>$')
    axs[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axs[0].tick_params(labelsize=13)

    axs[1].hlines(0.00, 0, bin1_mean_T[len(bin1_mean_T)-1],linestyles='dashed')
    axs[1].errorbar(bin1_mean_T, g1_obs_T[0,:]-0.02, yerr=g1err_obs_T[0,:], fmt='o', fillstyle='none', label=p)
    axs[1].set_xlabel(r'Measured T $(arcsec^{2})$')
    axs[1].set_xscale('log')
    axs[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axs[1].tick_params(labelsize=13)

    axs[2].hlines(0.00, 0, bin1_mean_size[len(bin1_mean_size)-1],linestyles='dashed')
    axs[2].errorbar(bin1_mean_size, g1_obs_size[0,:]-0.02, yerr=g1err_obs_size[0,:], fmt='o', fillstyle='none', label=p)
    axs[2].set_xlabel(r'Input T $(arcsec^{2})$')
    axs[2].set_xscale('log')
    axs[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axs[2].tick_params(labelsize=13)

    axs[3].hlines(0.00, 0, bin1_mean_Tpsf[len(bin1_mean_Tpsf)-1],linestyles='dashed')
    axs[3].errorbar(bin1_mean_Tpsf, g1_obs_Tpsf[0,:]-0.02, yerr=g1err_obs_Tpsf[0,:], fmt='o', fillstyle='none', label=p)
    axs[3].set_xlabel(r'$T_{psf} (arcsec^{2})$')
    axs[3].set_xscale('log')
    axs[3].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axs[3].tick_params(labelsize=13)
    axs[3].legend(fontsize='large')

    # axs[1,0].hlines(0.00, 0, bin2_mean_snr[len(bin2_mean_snr)-1],linestyles='dashed')
    # axs[1,0].errorbar(bin2_mean_snr, g2_obs_snr[1,:]-0.02, yerr=g2err_obs_snr[1,:], fmt='o', fillstyle='none', label=p)
    # axs[1,0].set_xlabel('S/N')
    # axs[1,0].set_xscale('log')
    # axs[1,0].set_ylabel(r'$<\Delta e_{2}>$')
    # axs[1,0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # axs[1,0].tick_params(labelsize=13)

    # axs[1,1].hlines(0.00, 0, bin2_mean_T[len(bin2_mean_T)-1],linestyles='dashed')
    # axs[1,1].errorbar(bin2_mean_T, g2_obs_T[1,:]-0.02, yerr=g2err_obs_T[1,:], fmt='o', fillstyle='none', label=p)
    # axs[1,1].set_xlabel(r'Measured T $(arcsec^{2})$')
    # axs[1,1].set_xscale('log')
    # axs[1,1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # axs[1,1].tick_params(labelsize=13)

    # axs[1,2].hlines(0.00, 0, bin2_mean_size[len(bin2_mean_size)-1],linestyles='dashed')
    # axs[1,2].errorbar(bin2_mean_size, g2_obs_size[1,:]-0.02, yerr=g2err_obs_size[1,:], fmt='o', fillstyle='none', label=p)
    # axs[1,2].set_xlabel(r'Input T $(arcsec^{2})$')
    # axs[1,2].set_xscale('log')
    # axs[1,2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # axs[1,2].tick_params(labelsize=13)

    # axs[1,3].hlines(0.00, 0, bin2_mean_Tpsf[len(bin2_mean_Tpsf)-1],linestyles='dashed')
    # axs[1,3].errorbar(bin2_mean_Tpsf, g2_obs_Tpsf[1,:]-0.02, yerr=g2err_obs_Tpsf[1,:], fmt='o', fillstyle='none', label=p)
    # axs[1,3].set_xlabel(r'$T_{psf} (arcsec^{2})$')
    # axs[1,3].set_xscale('log')
    # axs[1,3].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # axs[1,3].tick_params(labelsize=13)
    # axs[1,3].legend(loc=4)



plt.subplots_adjust(hspace=0.3,wspace=0.02)
plt.savefig(work_out+'H158_meanshear_measured_properties_perbin_e1.pdf')