import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import fitsio as fio
from esutil import stat

## bootstrap covariance function. 
def bootstrap_cov_m(N,data1,data2):
	fi = []
	for n in range(N):
		sample1 = np.random.choice(np.arange(len(data1)),len(data1),replace=True)
		sample2 = np.random.choice(np.arange(len(data2)),len(data2),replace=True)
		fi.append((np.mean(data1[sample1]) - np.mean(data2[sample2]))/0.04)
	f_mean = np.sum(fi)/N 
	fi = np.array(fi)
	cov = np.sqrt(np.sum((fi-f_mean)**2)/(N-1))
	return cov

def bootstrap_cov_c(N,m,data1,data2,data3,data4):
	fi = []
	for n in range(N):
		sample1 = np.random.choice(np.arange(len(data1)),len(data1),replace=True)
		sample2 = np.random.choice(np.arange(len(data2)),len(data2),replace=True)
		sample3 = np.random.choice(np.arange(len(data3)),len(data3),replace=True)
		sample4 = np.random.choice(np.arange(len(data4)),len(data4),replace=True)
		#function = np.mean((data1[sample1]-(1+m)*data2[sample2]) + (data3[sample3] - (1+m)*data4[sample4]))/2
		function = (np.mean(data1[sample1])-(1+m)*np.mean(data2[sample2]) + np.mean(data3[sample3]) - (1+m)*np.mean(data4[sample4]))/2
		fi.append(function)
	f_mean = np.sum(fi)/N 
	fi = np.array(fi)
	cov = np.sqrt(np.sum((fi-f_mean)**2)/(N-1))
	return cov

def analyze_gamma_obs(new,new1p,new1m,new2p,new2m,coadd_):

	if coadd_:
		g=0.01
		R11 = (new1p["coadd_e1"] - new1m["coadd_e1"])/(2*g)
		R22 = (new2p["coadd_e2"] - new2m["coadd_e2"])/(2*g)
		R12 = (new2p["coadd_e1"] - new2m["coadd_e1"])/(2*g)
		R21 = (new1p["coadd_e2"] - new1m["coadd_e2"])/(2*g)

		avg_R11 = np.mean(R11)
		avg_R22 = np.mean(R22)
		avg_R12 = np.mean(R12)
		avg_R21 = np.mean(R21)

		gamma1_obs = new['coadd_e1']/avg_R11
		gamma2_obs = new['coadd_e2']/avg_R22
		print('R', avg_R11, avg_R22)

		return new['g1'], new['g2'], gamma1_obs, gamma2_obs, new['coadd_e1'], new['coadd_e2']
	else:
		g=0.01
		R11 = (new1p["e1"] - new1m["e1"])/(2*g)
		R22 = (new2p["e2"] - new2m["e2"])/(2*g)
		R12 = (new2p["e1"] - new2m["e1"])/(2*g)
		R21 = (new1p["e2"] - new1m["e2"])/(2*g)

		avg_R11 = np.mean(R11)
		avg_R22 = np.mean(R22)
		avg_R12 = np.mean(R12)
		avg_R21 = np.mean(R21)

		gamma1_obs = new['e1']/avg_R11
		gamma2_obs = new['e2']/avg_R22
		print('R', avg_R11, avg_R22)

		return new['g1'], new['g2'], gamma1_obs, gamma2_obs, new['e1'], new['e2']


def shear_response(new,new1p,new1m,new2p,new2m,coadd_):
	
	g=0.01
	if coadd_:
		R11 = (new1p["coadd_e1"] - new1m["coadd_e1"])/(2*g)
		R22 = (new2p["coadd_e2"] - new2m["coadd_e2"])/(2*g)
		R12 = (new2p["coadd_e1"] - new2m["coadd_e1"])/(2*g)
		R21 = (new1p["coadd_e2"] - new1m["coadd_e2"])/(2*g)
	else:
		R11 = (new1p["e1"] - new1m["e1"])/(2*g)
		R22 = (new2p["e2"] - new2m["e2"])/(2*g)
	
	return np.mean(R11), np.mean(R22)

def shear_response_selection_correction(new,new1p,new1m,new2p,new2m,par,nperbin,coadd_=False):

	g = 0.01
	x_ = new[par]
	hist_ = stat.histogram(x_, nperbin=nperbin, more=True)
	bin_num = len(hist_['hist'])
	g1_true = []
	g1_obs = []
	g2_true = []
	g2_obs = []
	print('nbin ', bin_num)
	for i in range(bin_num):
		bin_mask = ((x_ > hist_['low'][i]) & (x_ < hist_['high'][i]))
		mask_1p = ((new1p[par] > hist_['low'][i]) & (new1p[par] < hist_['high'][i]))
		mask_1m = ((new1m[par] > hist_['low'][i]) & (new1m[par] < hist_['high'][i]))
		mask_2p = ((new2p[par] > hist_['low'][i]) & (new2p[par] < hist_['high'][i]))
		mask_2m = ((new2m[par] > hist_['low'][i]) & (new2m[par] < hist_['high'][i]))
	
		R_g = shear_response(new[bin_mask], new1p[bin_mask], new1m[bin_mask], new2p[bin_mask], new2m[bin_mask], coadd_)
		if coadd_:
			R11_s = (np.mean(new['coadd_e1'][mask_1p]) - np.mean(new['coadd_e1'][mask_1m]))/(2*g)
			R22_s = (np.mean(new['coadd_e2'][mask_2p]) - np.mean(new['coadd_e2'][mask_2m]))/(2*g)
			# R12_s[i] = (np.mean(new['e1'][mask_2p]) - np.mean(new['e1'][mask_2m]))/(2*g)
			# R21_s[i] = (np.mean(new['e2'][mask_1p]) - np.mean(new['e2'][mask_1m]))/(2*g)
		else:
			R11_s = (np.mean(new['e1'][mask_1p]) - np.mean(new['e1'][mask_1m]))/(2*g)
			R22_s = (np.mean(new['e2'][mask_2p]) - np.mean(new['e2'][mask_2m]))/(2*g)
		R11_tot = R_g[0] + R11_s
		R22_tot = R_g[1] + R22_s

		g1_true.append(new['g1'][bin_mask])
		g2_true.append(new['g2'][bin_mask])
		if coadd_:
			g1_obs.append(new['coadd_e1'][bin_mask]/R11_tot)
			g2_obs.append(new['coadd_e2'][bin_mask]/R22_tot)
		else:
			g1_obs.append(new['e1'][bin_mask]/R11_tot)
			g2_obs.append(new['e2'][bin_mask]/R22_tot)

	return g1_true,g1_obs,g2_true,g2_obs,hist_['mean']


def main(argv):

	g = 0.01
	old = None
	f = sys.argv[2] # example, /hpc/group/cosmology/phy-lsst/my137/roman_H158
	filter_ = sys.argv[3]
	coadd_ = False
	combine_m = False
	additional_mask = False
	v2 = False
	f_coadd = sys.argv[4] # example, coadd_multiband
	if v2:
		f_v2 = sys.argv[2]+'_v2'
		f_coadd_v2 = sys.argv[5]

	if not coadd_:
		folder = [f+'/g1002/ngmix/new_single/', f+'/g1n002/ngmix/new_single/', f+'/g2002/ngmix/new_single/', f+'/g2n002/ngmix/new_single/']
	else:
		folder = [f+'/g1002/ngmix/'+f_coadd+'/', f+'/g1n002/ngmix/'+f_coadd+'/', f+'/g2002/ngmix/'+f_coadd+'/', f+'/g2n002/ngmix/'+f_coadd+'/']
		if v2:
			folder_v2 = [f_v2+'/g1002/ngmix/'+f_coadd_v2+'/', f_v2+'/g1n002/ngmix/'+f_coadd_v2+'/', f_v2+'/g2002/ngmix/'+f_coadd_v2+'/', f_v2+'/g2n002/ngmix/'+f_coadd_v2+'/']
	dirr='fiducial_'+filter_+'_'
	model='mcal'

	start = 0
	noshear = []
	shear1p = []
	shear1m = []
	shear2p = []
	shear2m = []
	g1_true = []
	g2_true = []
	g1_obs = []
	g2_obs = []
	g1_noshear = []
	g2_noshear = []
	bin_x = None

	for j in range(len(folder)):
		new_ = fio.FITS(folder[j]+dirr+model+'_noshear.fits')[-1].read()
		new1p_ = fio.FITS(folder[j]+dirr+model+'_1p.fits')[-1].read()
		new1m_ = fio.FITS(folder[j]+dirr+model+'_1m.fits')[-1].read()
		new2p_ = fio.FITS(folder[j]+dirr+model+'_2p.fits')[-1].read()
		new2m_ = fio.FITS(folder[j]+dirr+model+'_2m.fits')[-1].read()
		print(j,len(new_),len(new1p_),len(new1m_),len(new2p_),len(new2m_),start)
		if additional_mask:
			mask = ((new_['flags']==0) & (new_['ind']!=0) & (new_['coadd_hlr'] > 0.032))
		else:
			mask = (new_['flags']==0) & (new_['ind']!=0) 

		noshear.append(new_[mask])
		shear1p.append(new1p_[mask])
		shear1m.append(new1m_[mask])
		shear2p.append(new2p_[mask])
		shear2m.append(new2m_[mask])

	if v2:
		noshear_v2 = []
		shear1p_v2 = []
		shear1m_v2 = []
		shear2p_v2 = []
		shear2m_v2 = []
		for j in range(len(folder_v2)):
			new_ = fio.FITS(folder_v2[j]+dirr+model+'_noshear.fits')[-1].read()
			new1p_ = fio.FITS(folder_v2[j]+dirr+model+'_1p.fits')[-1].read()
			new1m_ = fio.FITS(folder_v2[j]+dirr+model+'_1m.fits')[-1].read()
			new2p_ = fio.FITS(folder_v2[j]+dirr+model+'_2p.fits')[-1].read()
			new2m_ = fio.FITS(folder_v2[j]+dirr+model+'_2m.fits')[-1].read()
			print(j,len(new_),len(new1p_),len(new1m_),len(new2p_),len(new2m_),start)
			mask_v2 = (new_['flags']==0) & (new_['ind']!=0) # exclude non-zero flags object. 

			noshear_v2.append(new_[mask_v2])
			shear1p_v2.append(new1p_[mask_v2])
			shear1m_v2.append(new1m_[mask_v2])
			shear2p_v2.append(new2p_[mask_v2])
			shear2m_v2.append(new2m_[mask_v2])

	## finding common indices. 
	a,c00,c1 = np.intersect1d(noshear[0]['ind'], noshear[1]['ind'], return_indices=True)
	b,c01,c2 = np.intersect1d(noshear[0]['ind'][c00], noshear[2]['ind'], return_indices=True)
	c,c02,c3 = np.intersect1d(noshear[0]['ind'][c00][c01], noshear[3]['ind'], return_indices=True)
	tmp_ind = noshear[0]['ind'][c00][c01][c02]
	if v2:
		a_v2,c00_v2,c1_v2 = np.intersect1d(noshear_v2[0]['ind'], noshear_v2[1]['ind'], return_indices=True)
		b_v2,c01_v2,c2_v2 = np.intersect1d(noshear_v2[0]['ind'][c00_v2], noshear_v2[2]['ind'], return_indices=True)
		c_v2,c02_v2,c3_v2 = np.intersect1d(noshear_v2[0]['ind'][c00_v2][c01_v2], noshear_v2[3]['ind'], return_indices=True)
		tmp_ind_v2 = noshear_v2[0]['ind'][c00_v2][c01_v2][c02_v2]
	for run in range(4):
		if not v2:
			masking = np.isin(noshear[run]['ind'] ,tmp_ind)
			new = noshear[run][masking]
			new1p = shear1p[run][masking]
			new1m = shear1m[run][masking]
			new2p = shear2p[run][masking]
			new2m = shear2m[run][masking]
			print('the final object number is, ', len(new))
		else:
			masking = np.isin(noshear[run]['ind'] ,tmp_ind)
			masking_v2 = np.isin(noshear_v2[run]['ind'] ,tmp_ind_v2)
			new = np.concatenate((noshear[run][masking], noshear_v2[run][masking_v2]))
			new1p = np.concatenate((shear1p[run][masking], shear1p_v2[run][masking_v2]))
			new1m = np.concatenate((shear1m[run][masking], shear1m_v2[run][masking_v2]))
			new2p = np.concatenate((shear2p[run][masking], shear2p_v2[run][masking_v2]))
			new2m = np.concatenate((shear2m[run][masking], shear2m_v2[run][masking_v2]))
			print('the final object number is, ', len(new))

		if sys.argv[1]=='shear':
			gamma1_t,gamma2_t,gamma1_o,gamma2_o,noshear1,noshear2 = analyze_gamma_obs(new,new1p,new1m,new2p,new2m,coadd_=coadd_)
			g1_true.append(gamma1_t)
			g2_true.append(gamma2_t)
			g1_obs.append(gamma1_o)
			g2_obs.append(gamma2_o)
			g1_noshear.append(noshear1)
			g2_noshear.append(noshear2)

		elif sys.argv[1]=='selection':
			gamma1_t,gamma1_o,gamma2_t,gamma2_o,bin_hist_mean = shear_response_selection_correction(new,new1p,new1m,new2p,new2m,'coadd_hlr',coadd_=coadd_)
			g1_true.append(gamma1_t)
			g1_obs.append(gamma1_o)
			g2_true.append(gamma2_t)
			g2_obs.append(gamma2_o)
			if bin_x is None:
				bin_x = bin_hist_mean
	
	if combine_m:
		g_true_all = [np.concatenate([g1_true[0], g2_true[2]], axis=0), np.concatenate([g1_true[1], g2_true[3]], axis=0)]
		g_obs_all = [np.concatenate([g1_obs[0], g2_obs[2]], axis=0), np.concatenate([g1_obs[1], g2_obs[3]], axis=0)]
		g_noshear_all = [np.concatenate([g1_noshear[0], g2_noshear[2]], axis=0), np.concatenate([g1_noshear[1], g2_noshear[3]], axis=0)]

		m = ((np.mean(g_obs_all[0])-np.mean(g_obs_all[1]))/0.04) - 1
		m_err = bootstrap_cov_m(200,g_obs_all[0],g_obs_all[1])
		c = (np.mean(g_obs_all[0] - (1+m)*g_true_all[0]) + np.mean(g_obs_all[1] - (1+m)*g_true_all[1]))/2
		c_err = bootstrap_cov_c(200,m,g_obs_all[0],g_true_all[0],g_obs_all[1],g_true_all[1])

		print("metacalibration correction: ")
		print("m="+str("%6.4f"% m)+"+-"+str("%6.4f"% m_err), "c1="+str("%6.6f"% c)+"+-"+str("%6.6f"% c_err))
		return None

	if sys.argv[1]=='shear':
		## m1,c1 calculation before metacalibration correction. 
		m1 = ((np.mean(g1_noshear[0])-np.mean(g1_noshear[1]))/0.04) - 1
		m1_err = bootstrap_cov_m(200,g1_noshear[0],g1_noshear[1])
		c1 = (np.mean(g1_noshear[0] - (1+m1)*g1_true[0]) + np.mean(g1_noshear[1] - (1+m1)*g1_true[1]))/2
		c1_err = bootstrap_cov_c(200,m1,g1_noshear[0],g1_true[0],g1_noshear[1],g1_true[1])

		## m2,c2 calculation
		m2 = ((np.mean(g2_noshear[2])-np.mean(g2_noshear[3]))/0.04) - 1
		m2_err = bootstrap_cov_m(200,g2_noshear[2],g2_noshear[3])
		c2 = (np.mean(g2_noshear[2] - (1+m2)*g2_true[2]) + np.mean(g2_noshear[3] - (1+m2)*g2_true[3]))/2
		c2_err = bootstrap_cov_c(200,m2,g2_noshear[2],g2_true[2],g2_noshear[3],g2_true[3])

		print("before metacalibration correction: ")
		print("m1="+str("%6.4f"% m1)+"+-"+str("%6.4f"% m1_err), "c1="+str("%6.6f"% c1)+"+-"+str("%6.6f"% c1_err))
		print("m2="+str("%6.4f"% m2)+"+-"+str("%6.4f"% m2_err), "c2="+str("%6.6f"% c2)+"+-"+str("%6.6f"% c2_err))

		## m1,c1 calculation
		m11 = ((np.mean(g1_obs[0])-np.mean(g1_obs[1]))/0.04) - 1
		m11_err = bootstrap_cov_m(200,g1_obs[0],g1_obs[1])
		c11 = (np.mean(g1_obs[0] - (1+m11)*g1_true[0]) + np.mean(g1_obs[1] - (1+m11)*g1_true[1]))/2
		c11_err = bootstrap_cov_c(200,m11,g1_obs[0],g1_true[0],g1_obs[1],g1_true[1])

		## m2,c2 calculation
		m22 = ((np.mean(g2_obs[2])-np.mean(g2_obs[3]))/0.04) - 1
		m22_err = bootstrap_cov_m(200,g2_obs[2],g2_obs[3])
		c22 = (np.mean(g2_obs[2] - (1+m22)*g2_true[2]) + np.mean(g2_obs[3] - (1+m22)*g2_true[3]))/2
		c22_err = bootstrap_cov_c(200,m22,g2_obs[2],g2_true[2],g2_obs[3],g2_true[3])

		## off-diagonal components
		m12 = ((np.mean(g1_obs[2])-np.mean(g1_obs[3]))/0.04) 
		m12_err = bootstrap_cov_m(200,g1_obs[2],g1_obs[3])
		c12 = (np.mean(g1_obs[2] - (1+m12)*g1_true[2]) + np.mean(g1_obs[3] - (1+m12)*g1_true[3]))/2
		c12_err = bootstrap_cov_c(200,m12,g1_obs[2],g1_true[2],g1_obs[3],g1_true[3])

		m21 = ((np.mean(g2_obs[0])-np.mean(g2_obs[1]))/0.04) 
		m21_err = bootstrap_cov_m(200,g2_obs[0],g2_obs[1])
		c21 = (np.mean(g2_obs[0] - (1+m21)*g2_true[0]) + np.mean(g2_obs[1] - (1+m21)*g2_true[1]))/2
		c21_err = bootstrap_cov_c(200,m21,g2_obs[0],g2_true[0],g2_obs[1],g2_true[1])

		print('off-diagonal components: ')
		print("m12="+str("%6.4f"% m12)+"+-"+str("%6.4f"% m12_err), "c12="+str("%6.6f"% c12)+"+-"+str("%6.6f"% c12_err))
		print("m21="+str("%6.4f"% m21)+"+-"+str("%6.4f"% m21_err), "c21="+str("%6.6f"% c21)+"+-"+str("%6.6f"% c21_err))

		print("metacalibration correction: ")
		print("m1="+str("%6.4f"% m11)+"+-"+str("%6.4f"% m11_err), "c1="+str("%6.6f"% c11)+"+-"+str("%6.6f"% c11_err))
		print("m2="+str("%6.4f"% m22)+"+-"+str("%6.4f"% m22_err), "c2="+str("%6.6f"% c22)+"+-"+str("%6.6f"% c22_err))
	

	elif sys.argv[1]=='selection':
		m11=np.zeros(len(bin_x))
		m11_err=np.zeros(len(bin_x))
		m22=np.zeros(len(bin_x))
		m22_err=np.zeros(len(bin_x))
		c11=np.zeros(len(bin_x))
		c11_err=np.zeros(len(bin_x))
		c22=np.zeros(len(bin_x))
		c22_err=np.zeros(len(bin_x))
		
		for p in range(len(bin_x)):
			m11[p] = ((np.mean(g1_obs[0][p])-np.mean(g1_obs[1][p]))/0.04) - 1
			m11_err[p] = bootstrap_cov_m(200,g1_obs[0][p],g1_obs[1][p])
			c11[p] = (np.mean(g1_obs[0][p] - (1+m11[p])*g1_true[0][p]) + np.mean(g1_obs[1][p] - (1+m11[p])*g1_true[1][p]))/2
			c11_err[p] = bootstrap_cov_c(200,m11[p],g1_obs[0][p],g1_true[0][p],g1_obs[1][p],g1_true[1][p])

			## m2,c2 calculation
			m22[p] = ((np.mean(g2_obs[2][p])-np.mean(g2_obs[3][p]))/0.04) - 1
			m22_err[p] = bootstrap_cov_m(200,g2_obs[2][p],g2_obs[3][p])
			c22[p] = (np.mean(g2_obs[2][p] - (1+m22[p])*g2_true[2][p]) + np.mean(g2_obs[3][p] - (1+m22[p])*g2_true[3][p]))/2
			c22_err[p] = bootstrap_cov_c(200,m22[p],g2_obs[2][p],g2_true[2][p],g2_obs[3][p],g2_true[3][p])
		## shear response correction. 
		print(m11, m11_err)
		print(m22, m22_err)
		print('selection corrected m, b: ')
		print("m1="+str("%6.4f"% np.mean(m11))+"+-"+str("%6.4f"% np.mean(m11_err)), "b1="+str("%6.6f"% np.mean(c11))+"+-"+str("%6.6f"% np.mean(c11_err)))
		print("m2="+str("%6.4f"% np.mean(m22))+"+-"+str("%6.4f"% np.mean(m22_err)), "b2="+str("%6.6f"% np.mean(c22))+"+-"+str("%6.6f"% np.mean(c22_err)))

		fig,ax1=plt.subplots(figsize=(8,6))
		ax1.errorbar(bin_x, m11, yerr=m11_err,markerfacecolor='None', fmt='o', label='m1')
		ax1.errorbar(bin_x, m22, yerr=m22_err,markerfacecolor='None', fmt='o', label='m2')
		ax1.set_xscale('log')
		ax1.set_xlabel('log(T)', fontsize=15)
		ax1.set_ylabel('Multiplicative Bias, m', fontsize=15)
		ax1.set_title('No T cuts')
		plt.legend()
		plt.savefig('Hcoadd_ogpixel_nopixel_noTcuts_correction.png')


	return None

if __name__ == "__main__":
    main(sys.argv)