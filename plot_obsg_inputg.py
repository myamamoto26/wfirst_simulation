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
    #dirr=['v2_7_offset_0', 'v2_8_offset_0', 'v2_7_offset_10', 'v2_8_offset_10', 'v2_7_offset_45', 'v2_8_offset_45']
    #off=['g1_off0', 'g2_off0', 'g1_off10', 'g2_off10', 'g1_off45', 'g2_off45']
    dirr=['v2_7_offset_0', 'v2_7_offset_45']
    g_pos2 = []
    g_neg2 = []
    g_0 = []
    del_gamma1 = np.ones(num)
    del_gamma2 = np.ones(num)
    for i in range(len(dirr)):
        a=fio.FITS(dirr[i]+'_sim_0.fits')[-1].read() 
        b=fio.FITS(dirr[i]+'_sim_1.fits')[-1].read()
        c=fio.FITS(dirr[i]+'_sim_2.fits')[-1].read()
        d=fio.FITS(dirr[i]+'_sim_3.fits')[-1].read()
        e=fio.FITS(dirr[i]+'_sim_4.fits')[-1].read()

        R11, R22, R12, R21, g1_obs, g2_obs = residual_bias([a,b,c,d,e])
        print(np.mean(g1_obs[0:num:2]), np.mean(g2_obs[0:num]))
        g_pos2.append(g1_obs[0:num:2])
        g_neg2.append(g1_obs[1:num:2])
        g_0.append(g2_obs[0:num])
        #g1values,g1errors,g1snr_binslist = residual_bias_correction(a,b,c,d,e)

        #plot_combined(g1values, g1errors, g2values, g2errors, g2snr_binslist)
    del_g_pos2 = g_pos2[0] - g_pos2[1]
    del_g_neg2 = g_neg2[0] - g_neg2[1]
    del_g_0 = g_0[0] - g_0[1]
    #print(del_g_neg2)
    #print('The difference of the measured g1, when sheared in g1 direction, is, \u0394\u03B3='+str("%6.6f"% np.mean(del_gamma1))+"+-"+str("%6.6f"% (np.std(del_gamma1)/np.sqrt(num))))
    #print('The difference of the measured g2, when sheared in g1 direction, is, \u0394\u03B3='+str("%6.6f"% np.mean(del_gamma2))+"+-"+str("%6.6f"% (np.std(del_gamma2)/np.sqrt(num))))

    fig,ax1=plt.subplots(figsize=(10,8))
    input_shear = [-0.02, 0, 0.02]
    #error=[np.sqrt(np.std(g_neg2[0])**2/num + np.std(g_neg2[1])**2/num), np.sqrt(np.std(g_0[0])**2/num + np.std(g_0[1])**2/num), np.sqrt(np.std(g_pos2[0])**2/num + np.std(g_pos2[1])**2/num)]
    error=[np.std(del_g_neg2)/np.sqrt(len(del_g_neg2)), np.std(del_g_0)/np.sqrt(len(del_g_0)), np.std(del_g_pos2)/np.sqrt(len(del_g_pos2))]
    mean_difference = [np.mean(del_g_neg2), np.mean(del_g_0), np.mean(del_g_pos2)]
    ax1.plot(input_shear, mean_difference, 'o', c='r', label='sheared g1')
    ax1.errorbar(input_shear, mean_difference, yerr=error, c='r', fmt='o')

    dirr=['v2_8_offset_0', 'v2_8_offset_45']
    g_pos2 = []
    g_neg2 = []
    g_0 = []
    del_gamma1 = np.ones(num)
    del_gamma2 = np.ones(num)
    for i in range(len(dirr)):
        a=fio.FITS(dirr[i]+'_sim_0.fits')[-1].read() 
        b=fio.FITS(dirr[i]+'_sim_1.fits')[-1].read()
        c=fio.FITS(dirr[i]+'_sim_2.fits')[-1].read()
        d=fio.FITS(dirr[i]+'_sim_3.fits')[-1].read()
        e=fio.FITS(dirr[i]+'_sim_4.fits')[-1].read()

        R11, R22, R12, R21, g1_obs, g2_obs = residual_bias([a,b,c,d,e])
        print(np.mean(g1_obs[0:num]), np.mean(g2_obs[0:num:2]))
        g_pos2.append(g2_obs[0:num:2])
        g_neg2.append(g2_obs[1:num:2])
        g_0.append(g1_obs[0:num])
        #g1values,g1errors,g1snr_binslist = residual_bias_correction(a,b,c,d,e)

        #plot_combined(g1values, g1errors, g2values, g2errors, g2snr_binslist)
    del_g_pos2 = g_pos2[0] - g_pos2[1]
    del_g_neg2 = g_neg2[0] - g_neg2[1]
    del_g_0 = g_0[0] - g_0[1]
    #print(del_g_neg2)
    #print('The difference of the measured g1, when sheared in g2 direction, is, \u0394\u03B3='+str("%6.6f"% np.mean(del_gamma1))+"+-"+str("%6.6f"% (np.std(del_gamma1)/np.sqrt(num))))
    #print('The difference of the measured g2, when sheared in g2 direction, is, \u0394\u03B3='+str("%6.6f"% np.mean(del_gamma2))+"+-"+str("%6.6f"% (np.std(del_gamma2)/np.sqrt(num))))

    error=[np.std(del_g_neg2)/np.sqrt(len(del_g_neg2)), np.std(del_g_0)/np.sqrt(len(del_g_0)), np.std(del_g_pos2)/np.sqrt(len(del_g_pos2))]
    mean_difference = [np.mean(del_g_neg2), np.mean(del_g_0), np.mean(del_g_pos2)]
    ax1.plot(input_shear, mean_difference, 'o', c='b', label='sheared g2')
    ax1.errorbar(input_shear, mean_difference, yerr=error, c='b', fmt='o')
    ax1.set_xlabel('input shear', fontsize=16)
    ax1.set_ylabel("\u0394\u03B3", fontsize=16)
    ax1.set_title('Mean difference in measured shapes (offsets 0 and 45 degrees)', fontsize=13)
    plt.legend(loc=4, fontsize=12)
    ax1.tick_params(labelsize=12)
    plt.savefig('delta_g_offset45.png')
    plt.show()

    return None

if __name__ == "__main__":
    main(sys.argv)