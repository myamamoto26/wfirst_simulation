import numpy as np
import sys, os
import fitsio as fio
import pandas as pd
from des_analysis import analyze_gamma_obs
from esutil import stat
from matplotlib import pyplot as plt

work = '/hpc/group/cosmology/phy-lsst/my137/roman_H158/'
work_out = '/hpc/group/cosmology/masaya/wfirst_simulation/paper/'
coadd_path = 'new_coadd_oversample_original_coadd_pscfix'
sims = ['g1002', 'g1n002', 'g2002', 'g2n002']

def mean_shear_nperbin(new, new1p, new1m, new2p, new2m, nperbin, par):
        
    x_ = new[par]
    hist = stat.histogram(x_, nperbin=nperbin, more=True)
    bin_num = len(hist['hist'])
    g_obs = np.zeros(bin_num)
    gerr_obs = np.zeros(bin_num)
    print(len(hist['low']), len(hist['mean']))
    for i in range(bin_num):
        bin_mask = (x_ > hist['low'][i]) & (x_ < hist['high'][i])
        gamma1_t,gamma2_t,gamma1_o,gamma2_o,noshear1,noshear2 = analyze_gamma_obs(new[bin_mask], new1p[bin_mask], new1m[bin_mask], new2p[bin_mask], new2m[bin_mask], coadd_=True)
        g_obs[i] = np.mean(gamma1_o)
        gerr_obs[i] = np.std(gamma1_o)/np.sqrt(len(gamma1_o))

    return hist['mean'], g_obs, gerr_obs

start = 0
sets = ['g1002', 'g1n002', 'g2002', 'g2n002']
noshear = []
shear1p = []
shear1m = []
shear2p = []
shear2m = []
for i in range(4): # four sets of sim. 
    mcal_noshear = fio.FITS(os.path.join(work, sets[i]+"/ngmix/"+coadd_path+"/fiducial_H158_mcal_noshear.fits"))[-1].read()
    mcal_1p = fio.FITS(os.path.join(work, sets[i]+"/ngmix/"+coadd_path+"/fiducial_H158_mcal_1p.fits"))[-1].read()
    mcal_1m = fio.FITS(os.path.join(work, sets[i]+"/ngmix/"+coadd_path+"/fiducial_H158_mcal_1m.fits"))[-1].read()
    mcal_2p = fio.FITS(os.path.join(work, sets[i]+"/ngmix/"+coadd_path+"/fiducial_H158_mcal_2p.fits"))[-1].read()
    mcal_2m = fio.FITS(os.path.join(work, sets[i]+"/ngmix/"+coadd_path+"/fiducial_H158_mcal_2m.fits"))[-1].read()

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

    bin_mean_snr, g1_obs_snr, g1err_obs_snr = mean_shear_nperbin(new, new1p, new1m, new2p, new2m, 50000, 'coadd_snr')
    bin_mean_hlr, g1_obs_hlr, g1err_obs_hlr = mean_shear_nperbin(new, new1p, new1m, new2p, new2m, 50000, 'coadd_hlr')

    gamma1_t,gamma2_t,gamma1_o,gamma2_o,noshear1,noshear2 = analyze_gamma_obs(new,new1p,new1m,new2p,new2m,coadd_=True)

fig,axs = plt.subplots(1,4,figsize=(12,4))

d_x = [new['coadd_snr'], new['coadd_hlr']]
x_label = ['SNR', 'hlr']
for q,ax in enumerate(axs.ravel()):
    if q==2 or q==3:
        continue

    x_ = d_x[q]
    hist = stat.histogram(x_, nperbin=50000, more=True)
    bin_num = len(hist['hist'])
    g_obs = np.zeros(bin_num)
    gerr_obs = np.zeros(bin_num)
    print(len(hist['low']), len(hist['mean']))
    for i in range(bin_num):
        bin_mask = (x_ > hist['low'][i]) & (x_ < hist['high'][i])
        g_obs[i] = np.mean(gamma1_o[bin_mask])
        gerr_obs[i] = np.std(gamma1_o[bin_mask])/np.sqrt(len(gamma1_o[bin_mask]))

    ax.hlines(0.02, 0, hist['mean'][len(hist['mean'])-1],linestyles='dashed')
    ax.errorbar(hist['mean'], g_obs, yerr=gerr_obs, fmt='o', fillstyle='none')
    ax.set_xlabel(x_label[q])
    ax.set_xscale('log')
    ax.set_ylabel('<e1>')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


axs[2].hlines(0.02, 0, bin_mean_snr[len(bin_mean_snr)-1],linestyles='dashed')
axs[3].hlines(0.02, 0, bin_mean_hlr[len(bin_mean_hlr)-1],linestyles='dashed')
axs[2].errorbar(bin_mean_snr, g1_obs_snr, yerr=g1err_obs_snr, fmt='o', fillstyle='none')
axs[3].errorbar(bin_mean_hlr, g1_obs_hlr, yerr=g1err_obs_hlr, fmt='o', fillstyle='none')
axs[2].set_xlabel('SNR')
axs[3].set_xlabel('hlr')
axs[2].set_xscale('log')
axs[3].set_xscale('log')
axs[2].set_ylabel('<e1>')
axs[3].set_ylabel('<e1>')
axs[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
axs[3].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.savefig(work_out+'mean_shear_snrhlr_perbin.png')