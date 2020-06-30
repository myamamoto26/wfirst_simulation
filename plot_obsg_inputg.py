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
from astropy.stats import bootstrap

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
		## ngmix plot
		"""
		dirr=[['v2_11_offset_0', 'v2_11_offset_10'], ['v2_11_offset_0', 'v2_11_offset_20'], ['v2_11_offset_0', 'v2_11_offset_35'], 
				['v2_11_offset_0', 'v2_11_offset_45']]
		angles=[10,20,35,45]
		ind=0
		## g1 difference
		fig,ax1=plt.subplots(figsize=(10,8))
		for d in dirr:
			g_pos2 = []
			g_neg2 = []
			g_pos0 = []
			g_neg0 = []
			for name in d:
				a=fio.FITS(name+'_ngmix_0.fits')[-1].read() 
				b=None
				c=None
				d=None
				e=None

				R11, R22, R12, R21, g1_obs, g2_obs = residual_bias([a,b,c,d,e], 'ngmix')
				g_pos2.append(g1_obs[0:num:2])
				g_neg2.append(g1_obs[1:num:2])
				g_pos0.append(g2_obs[0:num:2])
				g_neg0.append(g2_obs[1:num:2])
			del_g1_pos2 = g_pos2[1] - g_pos2[0]
			del_g1_neg2 = g_neg2[1] - g_neg2[0]
			del_g2_pos0 = g_pos0[1] - g_pos0[0]
			del_g2_neg0 = g_neg0[1] - g_neg0[0]

			mean_g1=[np.mean(del_g1_neg2), np.mean(del_g1_pos2)]
			error_g1=[np.std(del_g1_neg2)/np.sqrt(len(del_g1_neg2)), np.std(del_g1_pos2)/np.sqrt(len(del_g1_pos2))]

			l3,=ax1.plot(angles[ind], mean_g1[0], 'o', c='b')
			ax1.errorbar(angles[ind], mean_g1[0], yerr=error_g1[0], c='b', fmt='o')
			l4,=ax1.plot(angles[ind], mean_g1[1], 'o', c='g')
			ax1.errorbar(angles[ind], mean_g1[1], yerr=error_g1[1], c='g', fmt='o') 
			ind+=1
		"""
		## metacal plot
		fig,ax1=plt.subplots(figsize=(10,8))
		dirr=[['v2_7_offset_0', 'v2_7_offset_10'], ['v2_7_offset_0', 'v2_7_offset_20'], ['v2_7_offset_0', 'v2_7_offset_35'],
				['v2_7_offset_0', 'v2_7_offset_40'], ['v2_7_offset_0', 'v2_7_offset_45'], 
				['v2_7_offset_0', 'v2_7_offset_50'], ['v2_7_offset_0', 'v2_7_offset_60']]
		angles=[10,20,35,40,45,50,60]
		ind=0
		## g1 difference
		for d in dirr:
			g_pos2 = []
			g_neg2 = []
			g_pos0 = []
			g_neg0 = []
			for name in d:
				a=fio.FITS(name+'_sim_0.fits')[-1].read() 
				b=fio.FITS(name+'_sim_1.fits')[-1].read()
				c=fio.FITS(name+'_sim_2.fits')[-1].read()
				d=fio.FITS(name+'_sim_3.fits')[-1].read()
				e=fio.FITS(name+'_sim_4.fits')[-1].read()

				R11, R22, R12, R21, g1_obs, g2_obs = residual_bias([a,b,c,d,e], 'metacal')
				g_pos2.append(g1_obs[0:num:2])
				g_neg2.append(g1_obs[1:num:2])
				g_pos0.append(g2_obs[0:num:2])
				g_neg0.append(g2_obs[1:num:2])
			del_g1_pos2 = g_pos2[1] - g_pos2[0]
			del_g1_neg2 = g_neg2[1] - g_neg2[0]
			del_g2_pos0 = g_pos0[1] - g_pos0[0]
			del_g2_neg0 = g_neg0[1] - g_neg0[0]

			mean_g1=[np.mean(del_g1_neg2), np.mean(del_g1_pos2)]
			boot=[bootstrap(del_g1_neg2,100), bootstrap(del_g1_pos2,100)]
			boot_mean=[np.mean([np.mean(sample) for sample in boot[0]]), np.mean([np.mean(sample) for sample in boot[1]])]
			sigma=[(np.sum([(np.mean(sample)-boot_mean[0])**2 for sample in boot[0]])/99)**(1/2), (np.sum([(np.mean(sample)-boot_mean[1])**2 for sample in boot[1]])/99)**(1/2)]
			error_g1=[np.std(del_g1_neg2)/np.sqrt(len(del_g1_neg2)), np.std(del_g1_pos2)/np.sqrt(len(del_g1_pos2))]
			print(sigma, error_g1)
			l1,=ax1.plot(angles[ind], mean_g1[0], 'o', c='r')
			ax1.errorbar(angles[ind], mean_g1[0], yerr=sigma[0], c='r', fmt='o')
			l2,=ax1.plot(angles[ind], mean_g1[1], 'o', c='m')
			ax1.errorbar(angles[ind], mean_g1[1], yerr=sigma[1], c='m', fmt='o') 
			ind+=1
		ax1.set_xlabel('Angle offsets', fontsize=16)
		ax1.set_ylabel("\u0394\u03B3", fontsize=16)
		#ax1.set_title('Mean difference in measured shapes for different shape measurement techniques', fontsize=13)
		l1.set_label('mcal g=-0.02')
		l2.set_label('mcal g=+0.02')
		#l3.set_label('ngmix g=-0.02')
		#l4.set_label('ngmix g=+0.02')
		plt.legend(loc=5, fontsize=10)
		ax1.tick_params(labelsize=10)
		ax1.axhline(y=0,ls='--')
		plt.savefig('ngmixmcal_delta_g_booterr.png')
		plt.show()

if __name__ == "__main__":
	main(sys.argv)