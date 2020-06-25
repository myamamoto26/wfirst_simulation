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

	if sys.argv[1]=='metacal':
		##g1=0, g2=0
		dirr=['v2_noshear_offset_0', 'v2_noshear_offset_45']
		shape=sys.argv[1]
		g1_0 = []
		g2_0 = []
		for i in range(len(dirr)):
			a=fio.FITS(dirr[i]+'_sim_0.fits')[-1].read() 
			b=fio.FITS(dirr[i]+'_sim_1.fits')[-1].read()
			c=fio.FITS(dirr[i]+'_sim_2.fits')[-1].read()
			d=fio.FITS(dirr[i]+'_sim_3.fits')[-1].read()
			e=fio.FITS(dirr[i]+'_sim_4.fits')[-1].read()

			R11, R22, R12, R21, g1_obs, g2_obs = residual_bias([a,b,c,d,e], shape)
			g1_0.append(g1_obs[0:num])
			g2_0.append(g2_obs[0:num])

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

			R11, R22, R12, R21, g1_obs, g2_obs = residual_bias([a,b,c,d,e], shape)
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

			R11, R22, R12, R21, g1_obs, g2_obs = residual_bias([a,b,c,d,e], shape)
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
		
		dirr=['v2_7_offset_0_rand360', 'v2_7_offset_45_rand360']
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

			R11, R22, R12, R21, g1_obs, g2_obs = residual_bias([a,b,c,d,e], shape)
			g_pos2.append(g1_obs[0:num:2])
			g_neg2.append(g1_obs[1:num:2])
			g_pos0.append(g2_obs[0:num:2])
			g_neg0.append(g2_obs[1:num:2])
			
		del_g1_randpos2 = g_pos2[1] - g_pos2[0]
		del_g1_randneg2 = g_neg2[1] - g_neg2[0]
		del_g2_randpos0 = g_pos0[1] - g_pos0[0]
		del_g2_randneg0 = g_neg0[1] - g_neg0[0]

		dirr=['v2_7_offset_0_rand20', 'v2_7_offset_45_rand20']
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

			R11, R22, R12, R21, g1_obs, g2_obs = residual_bias([a,b,c,d,e], shape)
			g_pos2.append(g1_obs[0:num:2])
			g_neg2.append(g1_obs[1:num:2])
			g_pos0.append(g2_obs[0:num:2])
			g_neg0.append(g2_obs[1:num:2])
			
		del_g1_rand2pos2 = g_pos2[1] - g_pos2[0]
		del_g1_rand2neg2 = g_neg2[1] - g_neg2[0]
		del_g2_rand2pos0 = g_pos0[1] - g_pos0[0]
		del_g2_rand2neg0 = g_neg0[1] - g_neg0[0]

		fig,ax1=plt.subplots(figsize=(10,8))
		input_shear = [-0.02, 0, 0, 0.02]
		#ax1.plot([0.0, 0.0], [np.mean(del_g1_0), np.mean(del_g2_0)], 'o', c='m', label='No shear, a fixed angle orientation')
		#ax1.errorbar([0.0, 0.0], [np.mean(del_g1_0), np.mean(del_g2_0)], yerr=[np.std(del_g1_0)/np.sqrt(len(del_g1_0)), np.std(del_g2_0)/np.sqrt(len(del_g2_0))], fmt='o', c='m')
		ax1.plot([0.0], [np.mean(del_g1_0)], 'o', c='m', label='No shear, a fixed angle orientation')
		ax1.errorbar([0.0], [np.mean(del_g1_0)], yerr=[np.std(del_g1_0)/np.sqrt(len(del_g1_0))], fmt='o', c='m')

		error_g1=[np.std(del_g1_neg2)/np.sqrt(len(del_g1_neg2)), np.std(del_g1_neg0)/np.sqrt(len(del_g1_neg0)), np.std(del_g1_pos0)/np.sqrt(len(del_g1_pos0)), np.std(del_g1_pos2)/np.sqrt(len(del_g1_pos2))]
		mean_difference_g1 = [np.mean(del_g1_neg2), np.mean(del_g1_neg0), np.mean(del_g1_pos0), np.mean(del_g1_pos2)]
		ax1.plot(input_shear, mean_difference_g1, 'o', c='r', label='g1, a fixed angle orientation')
		ax1.errorbar(input_shear, mean_difference_g1, yerr=error_g1, c='r', fmt='o')

		#error_g2=[np.std(del_g2_neg2)/np.sqrt(len(del_g2_neg2)), np.std(del_g2_neg0)/np.sqrt(len(del_g2_neg0)), np.std(del_g2_pos0)/np.sqrt(len(del_g2_pos0)), np.std(del_g2_pos2)/np.sqrt(len(del_g2_pos2))]
		#mean_difference_g2 = [np.mean(del_g2_neg2), np.mean(del_g2_neg0), np.mean(del_g2_pos0), np.mean(del_g2_pos2)]
		#ax1.plot(input_shear, mean_difference_g2, 'o', c='b', label='g2')
		#ax1.errorbar(input_shear, mean_difference_g2, yerr=error_g2, c='b', fmt='o')
		
		input2=[-0.02, 0.02]
		error_randg1=[np.std(del_g1_randneg2)/np.sqrt(len(del_g1_randneg2)), np.std(del_g1_randpos2)/np.sqrt(len(del_g1_randpos2))]
		mean_randdiff=[np.mean(del_g1_randneg2), np.mean(del_g1_randpos2)]
		ax1.plot(input2, mean_randdiff, 'o', c='b', label='g1, perfectly randomized orientations')
		ax1.errorbar(input2, mean_randdiff, yerr=error_randg1, c='b', fmt='o')

		error_rand2g1=[np.std(del_g1_rand2neg2)/np.sqrt(len(del_g1_rand2neg2)), np.std(del_g1_rand2pos2)/np.sqrt(len(del_g1_rand2pos2))]
		mean_rand2diff=[np.mean(del_g1_rand2neg2), np.mean(del_g1_rand2pos2)]
		ax1.plot(input2, mean_rand2diff, 'o', c='g', label='g1, slightly randomized orientations')
		ax1.errorbar(input2, mean_rand2diff, yerr=error_rand2g1, c='g', fmt='o')

		ax1.set_xlabel('input shear', fontsize=16)
		ax1.set_ylabel("\u0394\u03B3", fontsize=16)
		ax1.set_title('Mean difference in measured shapes (random orientation angles, offsets=45 degrees)', fontsize=13)
		plt.legend(loc=7, fontsize=10)
		ax1.tick_params(labelsize=10)
		ax1.axhline(y=0,ls='--')
		plt.savefig('delta_g_randoffset45.png')
		plt.show()

		return None

	elif sys.argv[1]=='ngmix':  
		dirr=['v2_11_offset_0', 'v2_11_offset_45']
		shape=sys.argv[1]
		g_pos2 = []
		g_neg2 = []
		g_pos0 = []
		g_neg0 = []
		for i in range(len(dirr)):
			a=fio.FITS(dirr[i]+'_ngmix_0.fits')[-1].read()
			b=None
			c=None
			d=None
			e=None 

			R11, R22, R12, R21, g1_obs, g2_obs = residual_bias([a,b,c,d,e], shape)
			g_pos2.append(g1_obs[0:num:2])
			g_neg2.append(g1_obs[1:num:2])
			g_pos0.append(g2_obs[0:num:2])
			g_neg0.append(g2_obs[1:num:2])

		del_g1_pos2_ng45 = g_pos2[1] - g_pos2[0]
		del_g1_neg2_ng45 = g_neg2[1] - g_neg2[0]
		del_g2_pos0_ng45 = g_pos0[1] - g_pos0[0]
		del_g2_neg0_ng45 = g_neg0[1] - g_neg0[0]

		dirr=['v2_11_offset_0', 'v2_11_offset_35']
		g_pos2 = []
		g_neg2 = []
		g_pos0 = []
		g_neg0 = []
		for i in range(len(dirr)):
			a=fio.FITS(dirr[i]+'_ngmix_0.fits')[-1].read()
			b=None
			c=None
			d=None
			e=None 

			R11, R22, R12, R21, g1_obs, g2_obs = residual_bias([a,b,c,d,e], shape)
			g_pos2.append(g1_obs[0:num:2])
			g_neg2.append(g1_obs[1:num:2])
			g_pos0.append(g2_obs[0:num:2])
			g_neg0.append(g2_obs[1:num:2])

		del_g1_pos2_ng35 = g_pos2[1] - g_pos2[0]
		del_g1_neg2_ng35 = g_neg2[1] - g_neg2[0]
		del_g2_pos0_ng35 = g_pos0[1] - g_pos0[0]
		del_g2_neg0_ng35 = g_neg0[1] - g_neg0[0]

		dirr=['v2_11_offset_0', 'v2_11_offset_20']
		g_pos2 = []
		g_neg2 = []
		g_pos0 = []
		g_neg0 = []
		for i in range(len(dirr)):
			a=fio.FITS(dirr[i]+'_ngmix_0.fits')[-1].read() 
			b=None
			c=None
			d=None
			e=None

			R11, R22, R12, R21, g1_obs, g2_obs = residual_bias([a,b,c,d,e], shape)
			g_pos2.append(g1_obs[0:num:2])
			g_neg2.append(g1_obs[1:num:2])
			g_pos0.append(g2_obs[0:num:2])
			g_neg0.append(g2_obs[1:num:2])
			
		del_g1_pos2_ng20 = g_pos2[1] - g_pos2[0]
		del_g1_neg2_ng20 = g_neg2[1] - g_neg2[0]
		del_g2_pos0_ng20 = g_pos0[1] - g_pos0[0]
		del_g2_neg0_ng20 = g_neg0[1] - g_neg0[0]
		
		dirr=['v2_11_offset_0', 'v2_11_offset_10']
		g_pos2 = []
		g_neg2 = []
		g_pos0 = []
		g_neg0 = []
		for i in range(len(dirr)):
			a=fio.FITS(dirr[i]+'_ngmix_0.fits')[-1].read() 
			b=None
			c=None
			d=None
			e=None

			R11, R22, R12, R21, g1_obs, g2_obs = residual_bias([a,b,c,d,e], shape)
			g_pos2.append(g1_obs[0:num:2])
			g_neg2.append(g1_obs[1:num:2])
			g_pos0.append(g2_obs[0:num:2])
			g_neg0.append(g2_obs[1:num:2])
			
		del_g1_pos2_ng10 = g_pos2[1] - g_pos2[0]
		del_g1_neg2_ng10 = g_neg2[1] - g_neg2[0]
		del_g2_pos0_ng10 = g_pos0[1] - g_pos0[0]
		del_g2_neg0_ng10 = g_neg0[1] - g_neg0[0]

		dirr=['v2_7_offset_0', 'v2_7_offset_60']
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

			R11, R22, R12, R21, g1_obs, g2_obs = residual_bias([a,b,c,d,e], 'metacal')
			g_pos2.append(g1_obs[0:num:2])
			g_neg2.append(g1_obs[1:num:2])
			g_pos0.append(g2_obs[0:num:2])
			g_neg0.append(g2_obs[1:num:2])
		del_g1_pos2_mc60 = g_pos2[1] - g_pos2[0]
		del_g1_neg2_mc60 = g_neg2[1] - g_neg2[0]
		del_g2_pos0_mc60 = g_pos0[1] - g_pos0[0]
		del_g2_neg0_mc60 = g_neg0[1] - g_neg0[0]

		dirr=['v2_7_offset_0', 'v2_7_offset_50']
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

			R11, R22, R12, R21, g1_obs, g2_obs = residual_bias([a,b,c,d,e], 'metacal')
			g_pos2.append(g1_obs[0:num:2])
			g_neg2.append(g1_obs[1:num:2])
			g_pos0.append(g2_obs[0:num:2])
			g_neg0.append(g2_obs[1:num:2])
		del_g1_pos2_mc50 = g_pos2[1] - g_pos2[0]
		del_g1_neg2_mc50 = g_neg2[1] - g_neg2[0]
		del_g2_pos0_mc50 = g_pos0[1] - g_pos0[0]
		del_g2_neg0_mc50 = g_neg0[1] - g_neg0[0]

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

			R11, R22, R12, R21, g1_obs, g2_obs = residual_bias([a,b,c,d,e], 'metacal')
			g_pos2.append(g1_obs[0:num:2])
			g_neg2.append(g1_obs[1:num:2])
			g_pos0.append(g2_obs[0:num:2])
			g_neg0.append(g2_obs[1:num:2])
		del_g1_pos2_mc45 = g_pos2[1] - g_pos2[0]
		del_g1_neg2_mc45 = g_neg2[1] - g_neg2[0]
		del_g2_pos0_mc45 = g_pos0[1] - g_pos0[0]
		del_g2_neg0_mc45 = g_neg0[1] - g_neg0[0]

		dirr=['v2_7_offset_0', 'v2_7_offset_40']
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

			R11, R22, R12, R21, g1_obs, g2_obs = residual_bias([a,b,c,d,e], 'metacal')
			g_pos2.append(g1_obs[0:num:2])
			g_neg2.append(g1_obs[1:num:2])
			g_pos0.append(g2_obs[0:num:2])
			g_neg0.append(g2_obs[1:num:2])
		del_g1_pos2_mc40 = g_pos2[1] - g_pos2[0]
		del_g1_neg2_mc40 = g_neg2[1] - g_neg2[0]
		del_g2_pos0_mc40 = g_pos0[1] - g_pos0[0]
		del_g2_neg0_mc40 = g_neg0[1] - g_neg0[0]

		dirr=['v2_7_offset_0', 'v2_7_offset_35']
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

			R11, R22, R12, R21, g1_obs, g2_obs = residual_bias([a,b,c,d,e], 'metacal')
			g_pos2.append(g1_obs[0:num:2])
			g_neg2.append(g1_obs[1:num:2])
			g_pos0.append(g2_obs[0:num:2])
			g_neg0.append(g2_obs[1:num:2])
		del_g1_pos2_mc35 = g_pos2[1] - g_pos2[0]
		del_g1_neg2_mc35 = g_neg2[1] - g_neg2[0]
		del_g2_pos0_mc35 = g_pos0[1] - g_pos0[0]
		del_g2_neg0_mc35 = g_neg0[1] - g_neg0[0]

		dirr=['v2_7_offset_0', 'v2_7_offset_20']
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

			R11, R22, R12, R21, g1_obs, g2_obs = residual_bias([a,b,c,d,e], 'metacal')
			g_pos2.append(g1_obs[0:num:2])
			g_neg2.append(g1_obs[1:num:2])
			g_pos0.append(g2_obs[0:num:2])
			g_neg0.append(g2_obs[1:num:2])
		del_g1_pos2_mc20 = g_pos2[1] - g_pos2[0]
		del_g1_neg2_mc20 = g_neg2[1] - g_neg2[0]
		del_g2_pos0_mc20 = g_pos0[1] - g_pos0[0]
		del_g2_neg0_mc20 = g_neg0[1] - g_neg0[0]

		dirr=['v2_7_offset_0', 'v2_7_offset_10']
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

			R11, R22, R12, R21, g1_obs, g2_obs = residual_bias([a,b,c,d,e], 'metacal')
			g_pos2.append(g1_obs[0:num:2])
			g_neg2.append(g1_obs[1:num:2])
			g_pos0.append(g2_obs[0:num:2])
			g_neg0.append(g2_obs[1:num:2])
		del_g1_pos2_mc10 = g_pos2[1] - g_pos2[0]
		del_g1_neg2_mc10 = g_neg2[1] - g_neg2[0]
		del_g2_pos0_mc10 = g_pos0[1] - g_pos0[0]
		del_g2_neg0_mc10 = g_neg0[1] - g_neg0[0]

		fig,ax1=plt.subplots(figsize=(10,8))

		ng_offsets=[10,20,35,45]
		error_g1_neg=[np.std(del_g1_neg2_ng10)/np.sqrt(len(del_g1_neg2_ng10)), np.std(del_g1_neg2_ng20)/np.sqrt(len(del_g1_neg2_ng20)), np.std(del_g1_neg2_ng35)/np.sqrt(len(del_g1_neg2_ng35)), np.std(del_g1_neg2_ng45)/np.sqrt(len(del_g1_neg2_ng45))]
		mean_difference_g1_neg = [np.mean(del_g1_neg2_ng10), np.mean(del_g1_neg2_ng20), np.mean(del_g1_neg2_ng35), np.mean(del_g1_neg2_ng20)]
		error_g1_pos=[np.std(del_g1_pos2_ng10)/np.sqrt(len(del_g1_pos2_ng10)), np.std(del_g1_pos2_ng20)/np.sqrt(len(del_g1_pos2_ng20)), np.std(del_g1_pos2_ng35)/np.sqrt(len(del_g1_pos2_ng35)), np.std(del_g1_pos2_ng45)/np.sqrt(len(del_g1_pos2_ng45))]
		mean_difference_g1_pos = [np.mean(del_g1_pos2_ng10), np.mean(del_g1_pos2_ng20), np.mean(del_g1_pos2_ng35), np.mean(del_g1_pos2_ng45)]
		ax1.plot(ng_offsets, mean_difference_g1_neg, 'o', c='r', label='ngmix g=-0.02')
		ax1.errorbar(ng_offsets, mean_difference_g1_neg, yerr=error_g1_neg, c='r', fmt='o')
		ax1.plot(ng_offsets, mean_difference_g1_pos, 'o', c='m', label='ngmix g=+0.02')
		ax1.errorbar(ng_offsets, mean_difference_g1_pos, yerr=error_g1_pos, c='m', fmt='o')
		
		mc_offsets=[10,20,35,40,45,50,60]
		error_g1_neg=[np.std(del_g1_neg2_mc10)/np.sqrt(len(del_g1_neg2_mc10)), np.std(del_g1_neg2_mc20)/np.sqrt(len(del_g1_neg2_mc20)), np.std(del_g1_neg2_mc35)/np.sqrt(len(del_g1_neg2_mc35)), 
						np.std(del_g1_neg2_mc40)/np.sqrt(len(del_g1_neg2_mc40)), np.std(del_g1_neg2_mc45)/np.sqrt(len(del_g1_neg2_mc45)), np.std(del_g1_neg2_mc50)/np.sqrt(len(del_g1_neg2_mc50)),
						np.std(del_g1_neg2_mc60)/np.sqrt(len(del_g1_neg2_mc60))]
		mean_difference_g1_neg = [np.mean(del_g1_neg2_mc10), np.mean(del_g1_neg2_mc20), np.mean(del_g1_neg2_mc35),  np.mean(del_g1_neg2_mc40), 
									np.mean(del_g1_neg2_mc45), np.mean(del_g1_neg2_mc50), np.mean(del_g1_neg2_mc60)]
		error_g1_pos=[np.std(del_g1_pos2_mc10)/np.sqrt(len(del_g1_pos2_mc10)), np.std(del_g1_pos2_mc20)/np.sqrt(len(del_g1_pos2_mc20)), np.std(del_g1_pos2_mc35)/np.sqrt(len(del_g1_pos2_mc35)), 
						np.std(del_g1_pos2_mc40)/np.sqrt(len(del_g1_pos2_mc40)), np.std(del_g1_pos2_mc45)/np.sqrt(len(del_g1_pos2_mc45)), np.std(del_g1_pos2_mc50)/np.sqrt(len(del_g1_pos2_mc50)), 
						np.std(del_g1_pos2_mc60)/np.sqrt(len(del_g1_pos2_mc60))]
		mean_difference_g1_pos = [np.mean(del_g1_pos2_mc10), np.mean(del_g1_pos2_mc20), np.mean(del_g1_pos2_mc35), np.mean(del_g1_pos2_mc40), 
									np.mean(del_g1_pos2_mc45), np.mean(del_g1_pos2_mc50), np.mean(del_g1_pos2_mc60)]
		ax1.plot(mc_offsets, mean_difference_g1_neg, 'o', c='b', label='mcal g=-0.02')
		ax1.errorbar(mc_offsets, mean_difference_g1_neg, yerr=error_g1_neg, c='b', fmt='o')
		ax1.plot(mc_offsets, mean_difference_g1_pos, 'o', c='g', label='mcal g=+0.02')
		ax1.errorbar(mc_offsets, mean_difference_g1_pos, yerr=error_g1_pos, c='g', fmt='o')

		ax1.set_xlabel('Angle offsets', fontsize=16)
		ax1.set_ylabel("\u0394\u03B3", fontsize=16)
		ax1.set_title('Mean difference in measured shapes for different shape measurement techniques', fontsize=13)
		plt.legend(loc=5, fontsize=10)
		ax1.tick_params(labelsize=10)
		ax1.axhline(y=0,ls='--')
		plt.savefig('ngmixmcal_delta_g_offset_add60.png')
		plt.show()

if __name__ == "__main__":
	main(sys.argv)