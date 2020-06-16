import numpy as np
#import healpy as hp
import sys, os, io
import math
import fitsio as fio

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

def main(argv):
    num=3000000

    ##g1=0, g2=0
    dirr=['v2_noshear_offset_0', 'v2_noshear_offset_45']
    g1_0 = []
    for i in range(len(dirr)):
        a=fio.FITS(dirr[i]+'_sim_0.fits')[-1].read() 
        b=fio.FITS(dirr[i]+'_sim_1.fits')[-1].read()
        c=fio.FITS(dirr[i]+'_sim_2.fits')[-1].read()
        d=fio.FITS(dirr[i]+'_sim_3.fits')[-1].read()
        e=fio.FITS(dirr[i]+'_sim_4.fits')[-1].read()

        R11, R22, R12, R21, g1_obs, g2_obs = residual_bias([a,b,c,d,e])
        g1_0.append(g1_obs[0:num])
        g2_0.appned(g2_obs[0:num])

    del_g1_0 = g1_0[1] - g1_0[0]
    del_g2_0 = g2_0[1] - g2_0[0]
    
    ## g1=+-0.02, g2=0
    dirr=['v2_7_offset_0', 'v2_7_offset_45']
    g_pos2 = []
    g_neg2 = []
    g_pos0 = []
    g_neg0 = []
    for i in range(len(dirr)):
        a=fio.FITS(dirr[i]+'_sim_0.fits')[-1].read() 
        b=fio.FITS(dirr[i]+'_sim_1.fits')[-1].read()
        c=fio.FITS(dirr[i]+'_sim_2.fits')[-1].read()
        d=fio.FITS(dirr[i]+'_sim_3.fits')[-1].read()
        e=fio.FITS(dirr[i]+'_sim_4.fits')[-1].read()

        R11, R22, R12, R21, g1_obs, g2_obs = residual_bias([a,b,c,d,e])
        g_pos2.append(g1_obs[0:num:2])
        g_neg2.append(g1_obs[1:num:2])
        g_pos0.append(g2_obs[0:num:2])
        g_neg0.append(g2_obs[1:num:2])

    del_g1_pos2 = g_pos2[1] - g_pos2[0]
    del_g1_neg2 = g_neg2[1] - g_neg2[0]
    del_g2_pos0 = g_pos0[1] - g_pos0[0]
    del_g2_neg0 = g_neg0[1] - g_neg0[0]
    #print('The difference of the measured g1, when sheared in g1 direction, is, \u0394\u03B3='+str("%6.6f"% np.mean(del_gamma1))+"+-"+str("%6.6f"% (np.std(del_gamma1)/np.sqrt(num))))
    #print('The difference of the measured g2, when sheared in g1 direction, is, \u0394\u03B3='+str("%6.6f"% np.mean(del_gamma2))+"+-"+str("%6.6f"% (np.std(del_gamma2)/np.sqrt(num))))

    ## g1=0, g2=+-0.02
    dirr=['v2_8_offset_0', 'v2_8_offset_45']
    g_pos2 = []
    g_neg2 = []
    g_pos0 = []
    g_neg0 = []
    for i in range(len(dirr)):
        a=fio.FITS(dirr[i]+'_sim_0.fits')[-1].read() 
        b=fio.FITS(dirr[i]+'_sim_1.fits')[-1].read()
        c=fio.FITS(dirr[i]+'_sim_2.fits')[-1].read()
        d=fio.FITS(dirr[i]+'_sim_3.fits')[-1].read()
        e=fio.FITS(dirr[i]+'_sim_4.fits')[-1].read()

        R11, R22, R12, R21, g1_obs, g2_obs = residual_bias([a,b,c,d,e])
        g_pos2.append(g2_obs[0:num:2])
        g_neg2.append(g2_obs[1:num:2])
        g_pos0.append(g1_obs[0:num:2])
        g_neg0.append(g1_obs[1:num:2])
        
    del_g2_pos2 = g_pos2[1] - g_pos2[0]
    del_g2_neg2 = g_neg2[1] - g_neg2[0]
    del_g1_pos0 = g_pos0[1] - g_pos0[0]
    del_g1_neg0 = g_neg0[1] - g_neg0[0]
    #print('The difference of the measured g1, when sheared in g2 direction, is, \u0394\u03B3='+str("%6.6f"% np.mean(del_gamma1))+"+-"+str("%6.6f"% (np.std(del_gamma1)/np.sqrt(num))))
    #print('The difference of the measured g2, when sheared in g2 direction, is, \u0394\u03B3='+str("%6.6f"% np.mean(del_gamma2))+"+-"+str("%6.6f"% (np.std(del_gamma2)/np.sqrt(num))))

    fig,ax1=plt.subplots(figsize=(10,8))
    input_shear = [-0.02, 0, 0, 0, 0.02]
    ax1.plot([0.0, 0.0], [np.mean(del_g1_0), np.mean(del_g2_0)], 'o', c='m', label='No shear')
    ax1.errorbar([0.0, 0.0], [np.mean(del_g1_0), np.mean(del_g2_0)], yerr=[np.std(del_g1_0)/np.sqrt(len(del_g1_0)), np.std(del_g2_0)/np.sqrt(len(del_g2_0))], fmt='o', c='m')

    error_g1=[np.std(del_g1_neg2)/np.sqrt(len(del_g1_neg2)), np.std(del_g1_neg0)/np.sqrt(len(del_g1_neg0)), np.std(del_g1_pos0)/np.sqrt(len(del_g1_pos0)), np.std(del_g1_pos2)/np.sqrt(len(del_g1_pos2))]
    mean_difference_g1 = [np.mean(del_g1_neg2), np.mean(del_g1_neg0), np.mean(del_g1_pos0), np.mean(del_g1_pos2)]
    ax1.plot(input_shear, mean_difference_g1, 'o', c='r', label='g1')
    ax1.errorbar(input_shear, mean_difference_g1, yerr=error_g1, c='r', fmt='o')

    error_g2=[np.std(del_g2_neg2)/np.sqrt(len(del_g2_neg2)), np.std(del_g2_neg0)/np.sqrt(len(del_g2_neg0)), np.std(del_g2_pos0)/np.sqrt(len(del_g2_pos0)), np.std(del_g2_pos2)/np.sqrt(len(del_g2_pos2))]
    mean_difference_g2 = [np.mean(del_g2_neg2), np.mean(del_g2_neg0), np.mean(del_g2_pos0), np.mean(del_g2_pos2)]
    ax1.plot(input_shear, mean_difference_g2, 'o', c='b', label='g2')
    ax1.errorbar(input_shear, mean_difference_g2, yerr=error_g2, c='b', fmt='o')
    ax1.set_xlabel('input shear', fontsize=16)
    ax1.set_ylabel("\u0394\u03B3", fontsize=16)
    ax1.set_title('Mean difference in measured shapes (offsets 0 and 45 degrees)', fontsize=13)
    plt.legend(loc=7, fontsize=12)
    ax1.tick_params(labelsize=12)
    ax1.axhline(y=0,ls='--')
    plt.savefig('delta_g_offset45.png')
    plt.show()

    return None

if __name__ == "__main__":
    main(sys.argv)