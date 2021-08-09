import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import fitsio as fio

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

def analyze_gamma_obs(new,new1p,new1m,new2p,new2m,coadd_=False):
	if not coadd_:
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

		return new['g1'], new['g2'], gamma1_obs, gamma2_obs, new['e1'], new['e2']

	else:
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

def shear_response(new,new1p,new1m,new2p,new2m):
	g=0.01
	R11 = (new1p["e1"] - new1m["e1"])/(2*g)
	R22 = (new2p["e2"] - new2m["e2"])/(2*g)
	R12 = (new2p["e1"] - new2m["e1"])/(2*g)
	R21 = (new1p["e2"] - new1m["e2"])/(2*g)
	
	return R11, R22, R12, R21

def shear_response_correction(new,new1p,new1m,new2p,new2m):

	g = 0.01
	R11, R22, R12, R21 = shear_response(new, new1p, new1m, new2p, new2m)

	avg_R11 = np.mean(R11)
	avg_R22 = np.mean(R22)

	#snr_binslist = np.linspace(np.log10(15),np.log10(500),11)
	#print(snr_min, snr_max, snr_binslist)

	R11_g = np.zeros(10)
	R22_g = np.zeros(10)
	R12_g = np.zeros(10)
	R21_g = np.zeros(10)
	"""
	for a in range(10):
		mask = (np.log10(new['snr']) >= snr_binslist[a]) & (np.log10(new['snr']) < snr_binslist[a+1])

		R11_g[a] = np.mean(R11[mask])
		R22_g[a] = np.mean(R22[mask])
		R12_g[a] = np.mean(R12[mask])
		R21_g[a] = np.mean(R21[mask])
	"""

	## getting cuts on the snr from the sheared catalogs and calculating selection response <R>selection
	R11_s = np.zeros(10)
	R22_s = np.zeros(10)
	R12_s = np.zeros(10)
	R21_s = np.zeros(10)
	"""
	for i in range(10):
		mask_1p = (np.log10(new1p['snr']) >= snr_binslist[i]) & (np.log10(new1p['snr']) < snr_binslist[i+1])
		mask_1m = (np.log10(new1m['snr']) >= snr_binslist[i]) & (np.log10(new1m['snr']) < snr_binslist[i+1])
		mask_2p = (np.log10(new2p['snr']) >= snr_binslist[i]) & (np.log10(new2p['snr']) < snr_binslist[i+1])
		mask_2m = (np.log10(new2m['snr']) >= snr_binslist[i]) & (np.log10(new2m['snr']) < snr_binslist[i+1])

		R11_s[i] = (np.mean(new['e1'][mask_1p]) - np.mean(new['e1'][mask_1m]))/(2*g)
		R22_s[i] = (np.mean(new['e2'][mask_2p]) - np.mean(new['e2'][mask_2m]))/(2*g)
		R12_s[i] = (np.mean(new['e1'][mask_2p]) - np.mean(new['e1'][mask_2m]))/(2*g)
		R21_s[i] = (np.mean(new['e2'][mask_1p]) - np.mean(new['e2'][mask_1m]))/(2*g)

	## total response
	tot_R11 = R11_g + R11_s
	tot_R22 = R22_g + R22_s
	#tot_R12 = R12_g + R12_s
	#tot_R21 = R21_g + R21_s
	"""

	## equal nubmer of objects in each bin.
	equal_bin = np.linspace(0,len(new['hlr']),11)#np.linspace(0,len(new['snr']),11)
	idx = np.argsort(new['hlr'])#np.argsort(new['snr'])
	start=0
	for a in range(10):
		end = int(equal_bin[a+1])
		R11_g[a] = np.mean(np.array(R11)[idx][start:end+1])
		R22_g[a] = np.mean(np.array(R22)[idx][start:end+1])
		R12_g[a] = np.mean(np.array(R12)[idx][start:end+1])
		R21_g[a] = np.mean(np.array(R21)[idx][start:end+1])
		start = end+1

	mask_1p = np.argsort(new1p['hlr'])#np.argsort(new1p['snr'])
	mask_1m = np.argsort(new1m['hlr'])#np.argsort(new1m['snr'])
	mask_2p = np.argsort(new2p['hlr'])#np.argsort(new2p['snr'])
	mask_2m = np.argsort(new2m['hlr'])#np.argsort(new2m['snr'])
	start=0
	for i in range(10):
		end = int(equal_bin[i+1])
		R11_s[i] = (np.mean(np.array(new['e1'])[mask_1p][start:end+1]) - np.mean(np.array(new['e1'])[mask_1m][start:end+1]))/(2*g)
		R22_s[i] = (np.mean(np.array(new['e2'])[mask_2p][start:end+1]) - np.mean(np.array(new['e2'])[mask_2m][start:end+1]))/(2*g)
		R12_s[i] = (np.mean(np.array(new['e1'])[mask_2p][start:end+1]) - np.mean(np.array(new['e1'])[mask_2m][start:end+1]))/(2*g)
		R21_s[i] = (np.mean(np.array(new['e2'])[mask_1p][start:end+1]) - np.mean(np.array(new['e2'])[mask_1m][start:end+1]))/(2*g)
		start = end+1
	## total response
	tot_R11 = R11_g + R11_s
	tot_R22 = R22_g + R22_s
	#tot_R12 = R12_g + R12_s
	#tot_R21 = R21_g + R21_s

	return tot_R11,tot_R22


def residual_bias_correction(new, new1p, new1m, new2p, new2m, R11, R22):

	#snr_binslist = np.linspace(np.log10(15),np.log10(500),11)

	g1_true_snr=[]
	g1_obs_snr=[]
	g2_true_snr=[]
	g2_obs_snr=[]
	snr_bin=[]
	"""
	for p in range(10):
		mask = (np.log10(new['snr']) >= snr_binslist[p]) & (np.log10(new['snr']) < snr_binslist[p+1])
		#mask = (new['hlr'] >= snr_binslist[p]) & (new['hlr'] < snr_binslist[p+1])
		g1_true_snr.append(new['g1'][mask])
		g1_obs_snr.append(new['e1'][mask]/R11[p])

		g2_true_snr.append(new['g2'][mask])
		g2_obs_snr.append(new['e2'][mask]/R22[p])
	"""

	## equal nubmer of objects in each bin.
	equal_bin = np.linspace(0,len(new['hlr']),11)#np.linspace(0,len(new['snr']),11)
	idx = np.argsort(new['hlr'])#np.argsort(new['snr'])
	sorted_g1 = np.array(new['g1'])[idx]
	sorted_e1 = np.array(new['e1'])[idx]
	sorted_g2 = np.array(new['g2'])[idx]
	sorted_e2 = np.array(new['e2'])[idx]
	start=0
	for p in range(10):
		end = int(equal_bin[p+1])
		g1_true_snr.append(sorted_g1[start:end+1])
		g1_obs_snr.append(sorted_e1[start:end+1]/R11[p])
		g2_true_snr.append(sorted_g2[start:end+1])
		g2_obs_snr.append(sorted_e2[start:end+1]/R22[p])
		snr_bin.append(np.mean(np.array(new['hlr'])[idx][start:end+1]))#snr_bin.append(np.mean(np.array(new['snr'])[idx][start:end+1]))
		start = end+1


	return g1_true_snr,g1_obs_snr,g2_true_snr,g2_obs_snr,snr_bin


def main(argv):

	g = 0.01
	old = None
	f = sys.argv[2] # example, /hpc/group/cosmology/phy-lsst/my137/roman_H158
	filter_ = sys.argv[3]
	coadd_ = False
	combine_m = True
	v2 = False
	f_coadd = sys.argv[4] # example, coadd_multiband
	if v2:
		f_v2 = sys.argv[2]+'_v2'
		f_coadd_v2 = sys.argv[5]

	if not coadd_:
		folder = [f+'/g1002/ngmix/single/', f+'/g1n002/ngmix/single/', f+'/g2002/ngmix/single/', f+'/g2n002/ngmix/single/']
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
	g1snr_true = []
	g2snr_true = []
	g1snr_obs = []
	g2snr_obs = []
	snr_x = []

	for j in range(len(folder)):
		new_ = fio.FITS(folder[j]+dirr+model+'_noshear.fits')[-1].read()
		new1p_ = fio.FITS(folder[j]+dirr+model+'_1p.fits')[-1].read()
		new1m_ = fio.FITS(folder[j]+dirr+model+'_1m.fits')[-1].read()
		new2p_ = fio.FITS(folder[j]+dirr+model+'_2p.fits')[-1].read()
		new2m_ = fio.FITS(folder[j]+dirr+model+'_2m.fits')[-1].read()
		print(j,len(new_),len(new1p_),len(new1m_),len(new2p_),len(new2m_),start)
		mask = (new_['flags']==0) & (new_['ind']!=0) # exclude non-zero flags object. 

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
			R11_correction, R22_correction = shear_response_correction(new,new1p,new1m,new2p,new2m)
			g1_true_snr,g1_obs_snr,g2_true_snr,g2_obs_snr,snr_bin = residual_bias_correction(new,new1p,new1m,new2p,new2m,R11_correction,R22_correction)
			g1snr_true.append(g1_true_snr)
			g1snr_obs.append(g1_obs_snr)
			g2snr_true.append(g2_true_snr)
			g2snr_obs.append(g2_obs_snr)
			snr_x.append(snr_bin)
	
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

		#print(m11,c11,m22,c22,m12,c12,m21,c21)

		print('off-diagonal components: ')
		print("m12="+str("%6.4f"% m12)+"+-"+str("%6.4f"% m12_err), "c12="+str("%6.6f"% c12)+"+-"+str("%6.6f"% c12_err))
		print("m21="+str("%6.4f"% m21)+"+-"+str("%6.4f"% m21_err), "c21="+str("%6.6f"% c21)+"+-"+str("%6.6f"% c21_err))

		print("metacalibration correction: ")
		print("m1="+str("%6.4f"% m11)+"+-"+str("%6.4f"% m11_err), "c1="+str("%6.6f"% c11)+"+-"+str("%6.6f"% c11_err))
		print("m2="+str("%6.4f"% m22)+"+-"+str("%6.4f"% m22_err), "c2="+str("%6.6f"% c22)+"+-"+str("%6.6f"% c22_err))
	

	elif sys.argv[1]=='selection':
		m11_snr=np.zeros(10)
		m11_snr_err=np.zeros(10)
		m22_snr=np.zeros(10)
		m22_snr_err=np.zeros(10)
		c11_snr=np.zeros(10)
		c11_snr_err=np.zeros(10)
		c22_snr=np.zeros(10)
		c22_snr_err=np.zeros(10)
		for p in range(10):
			print(len(g1snr_obs[0][p]),len(g1snr_true[0][p]))
			m11_snr[p] = ((np.mean(g1snr_obs[0][p])-np.mean(g1snr_obs[1][p]))/0.04) - 1
			m11_snr_err[p] = bootstrap_cov_m(200,g1snr_obs[0][p],g1snr_obs[1][p])
			c11_snr[p] = (np.mean(g1snr_obs[0][p] - (1+m11_snr[p])*g1snr_true[0][p]) + np.mean(g1snr_obs[1][p] - (1+m11_snr[p])*g1snr_true[1][p]))/2
			c11_snr_err[p] = bootstrap_cov_c(200,m11_snr[p],g1snr_obs[0][p],g1snr_true[0][p],g1snr_obs[1][p],g1snr_true[1][p])

			## m2,c2 calculation
			m22_snr[p] = ((np.mean(g2snr_obs[2][p])-np.mean(g2snr_obs[3][p]))/0.04) - 1
			m22_snr_err[p] = bootstrap_cov_m(200,g2snr_obs[2][p],g2snr_obs[3][p])
			c22_snr[p] = (np.mean(g2snr_obs[2][p] - (1+m22_snr[p])*g2snr_true[2][p]) + np.mean(g2snr_obs[3][p] - (1+m22_snr[p])*g2snr_true[3][p]))/2
			c22_snr_err[p] = bootstrap_cov_c(200,m22_snr[p],g2snr_obs[2][p],g2snr_true[2][p],g2snr_obs[3][p],g2snr_true[3][p])
		## shear response correction. 
		print(m11_snr, m11_snr_err)
		print(m22_snr, m22_snr_err)
		print('corrected m, b: ')
		print("m1="+str("%6.4f"% np.mean(m11_snr))+"+-"+str("%6.4f"% np.mean(m11_snr_err)), "b1="+str("%6.6f"% np.mean(c11_snr))+"+-"+str("%6.6f"% np.mean(c11_snr_err)))
		print("m2="+str("%6.4f"% np.mean(m22_snr))+"+-"+str("%6.4f"% np.mean(m22_snr_err)), "b2="+str("%6.6f"% np.mean(c22_snr))+"+-"+str("%6.6f"% np.mean(c22_snr_err)))

		fig,ax1=plt.subplots(figsize=(8,6))
		#snr = np.linspace(np.log10(10),np.log10(500),11)
		#x_ = [(snr[i]+snr[i+1])/2 for i in range(len(snr)-1)]
		x_ = [np.log10(np.mean([snr_x[0][i],snr_x[1][i],snr_x[2][i],snr_x[3][i]])) for i in range(10)]
		ax1.plot(x_, m11_snr, 'o', markeredgecolor='b',markerfacecolor='None', label='m1')
		ax1.errorbar(x_, m11_snr, yerr=m11_snr_err, markeredgecolor='b',markerfacecolor='None', fmt='o')
		ax1.plot(x_, m22_snr, 'o', markeredgecolor='r',markerfacecolor='None', label='m2')
		ax1.errorbar(x_, m22_snr, yerr=m22_snr_err, markeredgecolor='r',markerfacecolor='None', fmt='o')
		ax1.set_xlabel('log(hlr)', fontsize=15)
		ax1.set_ylabel('Multiplicative Bias, m', fontsize=15)
		plt.legend()
		plt.savefig('roman_g002_m_equalhlr.png')


	return None

if __name__ == "__main__":
    main(sys.argv)