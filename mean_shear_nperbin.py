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
    return 2*(2*d/(2.3548200450309493))**2

def mean_shear_nperbin(new, new1p, new1m, new2p, new2m, nperbin, par, coadd):
    
    if 'hlr' in par:
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
matplotlib.rcParams.update({'font.size': 22})
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
    for run in range(1):
        new = noshear[run][np.isin(noshear[run]['ind'] ,tmp_ind)]
        new1p = shear1p[run][np.isin(noshear[run]['ind'] ,tmp_ind)]
        new1m = shear1m[run][np.isin(noshear[run]['ind'] ,tmp_ind)]
        new2p = shear2p[run][np.isin(noshear[run]['ind'] ,tmp_ind)]
        new2m = shear2m[run][np.isin(noshear[run]['ind'] ,tmp_ind)]

        bin_mean_snr, g_obs_snr, gerr_obs_snr = mean_shear_nperbin(new, new1p, new1m, new2p, new2m, 50000, xax[0], coadd)
        bin_mean_T, g_obs_T, gerr_obs_T = mean_shear_nperbin(new, new1p, new1m, new2p, new2m, 50000, xax[1], coadd)
        bin_mean_size, g_obs_size, gerr_obs_size = mean_shear_nperbin(new, new1p, new1m, new2p, new2m, 50000, xax[2], coadd)
        bin_mean_Tpsf, g_obs_Tpsf, gerr_obs_Tpsf = mean_shear_nperbin(new, new1p, new1m, new2p, new2m, 50000, xax[3], coadd)

        gamma1_t,gamma2_t,gamma1_o,gamma2_o,noshear1,noshear2 = analyze_gamma_obs(new,new1p,new1m,new2p,new2m,coadd)

    # d_x = [new['coadd_snr'], new['coadd_hlr'], new['coadd_psf_T']]
    # x_label = ['S/N', 'T', 'T_{psf}']
    # for q,ax in enumerate(axs.ravel()):

    #     x_ = d_x[q]
    #     hist = stat.histogram(x_, nperbin=50000, more=True)
    #     bin_num = len(hist['hist'])
    #     g_obs = np.zeros(bin_num)
    #     gerr_obs = np.zeros(bin_num)
    #     print(len(hist['low']), len(hist['mean']))
    #     for i in range(bin_num):
    #         bin_mask = (x_ > hist['low'][i]) & (x_ < hist['high'][i])
    #         g_obs[i] = np.mean(gamma1_o[bin_mask])
    #         gerr_obs[i] = np.std(gamma1_o[bin_mask])/np.sqrt(len(gamma1_o[bin_mask]))

    #     ax.hlines(0.02, 0, hist['mean'][len(hist['mean'])-1],linestyles='dashed')
    #     ax.errorbar(hist['mean'], g_obs, yerr=gerr_obs, fmt='o', fillstyle='none')
    #     ax.set_xlabel(x_label[q])
    #     ax.set_xscale('log')
    #     ax.set_ylabel('<e1>')
    #     ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


    axs[0,0].hlines(0.00, 0, bin_mean_snr[len(bin_mean_snr)-1],linestyles='dashed')
    axs[0,0].errorbar(bin_mean_snr, g_obs_snr[0,:]-0.02, yerr=gerr_obs_snr[0,:], fmt='o', fillstyle='none', label=p)
    axs[0,0].set_xlabel('S/N')
    axs[0,0].set_xscale('log')
    axs[0,0].set_ylabel(r'$<\Delta e_{1}>$')
    axs[0,0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axs[0,0].tick_params(labelsize=13)

    axs[0,1].hlines(0.00, 0, bin_mean_T[len(bin_mean_T)-1],linestyles='dashed')
    axs[0,1].errorbar(bin_mean_T, g_obs_T[0,:]-0.02, yerr=gerr_obs_T[0,:], fmt='o', fillstyle='none', label=p)
    axs[0,1].set_xlabel(r'Measured T $(arcsec^{2})$')
    axs[0,1].set_xscale('log')
    axs[0,1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axs[0,1].tick_params(labelsize=13)

    axs[0,2].hlines(0.00, 0, bin_mean_size[len(bin_mean_size)-1],linestyles='dashed')
    axs[0,2].errorbar(bin_mean_size, g_obs_size[0,:]-0.02, yerr=gerr_obs_size[0,:], fmt='o', fillstyle='none', label=p)
    axs[0,2].set_xlabel(r'Input T $(arcsec^{2})$')
    axs[0,2].set_xscale('log')
    axs[0,2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axs[0,2].tick_params(labelsize=13)

    axs[0,3].hlines(0.00, 0, bin_mean_Tpsf[len(bin_mean_Tpsf)-1],linestyles='dashed')
    axs[0,3].errorbar(bin_mean_Tpsf, g_obs_Tpsf[0,:]-0.02, yerr=gerr_obs_Tpsf[0,:], fmt='o', fillstyle='none', label=p)
    axs[0,3].set_xlabel(r'$T_{psf} (arcsec^{2})$')
    axs[0,3].set_xscale('log')
    axs[0,3].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axs[0,3].tick_params(labelsize=13)
    axs[0,3].legend(fontsize='large')

    axs[1,0].hlines(0.00, 0, bin_mean_snr[len(bin_mean_snr)-1],linestyles='dashed')
    axs[1,0].errorbar(bin_mean_snr, g_obs_snr[1,:]-0.02, yerr=gerr_obs_snr[1,:], fmt='o', fillstyle='none', label=p)
    axs[1,0].set_xlabel('S/N')
    axs[1,0].set_xscale('log')
    axs[1,0].set_ylabel(r'$<\Delta e_{2}>$')
    axs[1,0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axs[1,0].tick_params(labelsize=13)

    axs[1,1].hlines(0.00, 0, bin_mean_T[len(bin_mean_T)-1],linestyles='dashed')
    axs[1,1].errorbar(bin_mean_T, g_obs_T[1,:]-0.02, yerr=gerr_obs_T[1,:], fmt='o', fillstyle='none', label=p)
    axs[1,1].set_xlabel(r'Measured T $(arcsec^{2})$')
    axs[1,1].set_xscale('log')
    axs[1,1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axs[1,1].tick_params(labelsize=13)

    axs[1,2].hlines(0.00, 0, bin_mean_size[len(bin_mean_size)-1],linestyles='dashed')
    axs[1,2].errorbar(bin_mean_size, g_obs_size[1,:]-0.02, yerr=gerr_obs_size[1,:], fmt='o', fillstyle='none', label=p)
    axs[1,2].set_xlabel(r'Input T $(arcsec^{2})$')
    axs[1,2].set_xscale('log')
    axs[1,2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axs[1,2].tick_params(labelsize=13)

    axs[1,3].hlines(0.00, 0, bin_mean_Tpsf[len(bin_mean_Tpsf)-1],linestyles='dashed')
    axs[1,3].errorbar(bin_mean_Tpsf, g_obs_Tpsf[1,:]-0.02, yerr=gerr_obs_Tpsf[1,:], fmt='o', fillstyle='none', label=p)
    axs[1,3].set_xlabel(r'$T_{psf} (arcsec^{2})$')
    axs[1,3].set_xscale('log')
    axs[1,3].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axs[1,3].tick_params(labelsize=13)
    axs[1,3].legend(fontsize='large')



plt.subplots_adjust(hspace=0.1,wspace=0.02)
plt.savefig(work_out+'H158_meanshear_measured_properties_perbin.pdf')