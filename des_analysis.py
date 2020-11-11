import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import fitsio as fio


def analyze_gamma_obs(new,new1p,new1m,new2p,new2m):
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

	return new['g1'], new['g2'], gamma1_obs, gamma2_obs

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

	snr_binn = 10
	snr_min = np.log(15) #np.min(new['hlr']) #np.log(15) #np.log(min(new['snr']))
	snr_max = np.log(500) #np.max(new['hlr']) #np.log(max(new['snr']))
	snr_binslist = [snr_min+(x*((snr_max-snr_min)/10)) for x in range(11)]
	#print(snr_min, snr_max, snr_binslist)
	if snr_binslist[10] != snr_max:
		print("raise an error.")

	R11_g = np.zeros(10)
	R22_g = np.zeros(10)
	R12_g = np.zeros(10)
	R21_g = np.zeros(10)
	for a in range(10):
		mask = (np.log(new['snr']) >= snr_binslist[a]) & (np.log(new['snr']) < snr_binslist[a+1])

		R11_g[a] = np.mean(R11[mask])
		R22_g[a] = np.mean(R22[mask])
		R12_g[a] = np.mean(R12[mask])
		R21_g[a] = np.mean(R21[mask])

	## getting cuts on the snr from the sheared catalogs and calculating selection response <R>selection
	R11_s = np.zeros(10)
	R22_s = np.zeros(10)
	R12_s = np.zeros(10)
	R21_s = np.zeros(10)
	for i in range(10):
		mask_1p = (np.log(new1p['snr']) >= snr_binslist[i]) & (np.log(new1p['snr']) < snr_binslist[i+1])
		mask_1m = (np.log(new1m['snr']) >= snr_binslist[i]) & (np.log(new1m['snr']) < snr_binslist[i+1])
		mask_2p = (np.log(new2p['snr']) >= snr_binslist[i]) & (np.log(new2p['snr']) < snr_binslist[i+1])
		mask_2m = (np.log(new2m['snr']) >= snr_binslist[i]) & (np.log(new2m['snr']) < snr_binslist[i+1])

		R11_s[i] = (np.mean(new['e1'][mask_1p]) - np.mean(new['e1'][mask_1m]))/(2*g)
		R22_s[i] = (np.mean(new['e2'][mask_2p]) - np.mean(new['e2'][mask_2m]))/(2*g)
		R12_s[i] = (np.mean(new['e1'][mask_2p]) - np.mean(new['e1'][mask_2m]))/(2*g)
		R21_s[i] = (np.mean(new['e2'][mask_1p]) - np.mean(new['e2'][mask_1m]))/(2*g)

	## total response
	tot_R11 = R11_g + R11_s
	tot_R22 = R22_g + R22_s
	#tot_R12 = R12_g + R12_s
	#tot_R21 = R21_g + R21_s
	return tot_R11,tot_R22


def residual_bias_correction(new, new1p, new1m, new2p, new2m, R11, R22):

	snr_binn = 10
	snr_min = np.log(15) #np.min(new['hlr']) #np.log(15) #np.log(min(new['snr']))
	snr_max = np.log(500) #np.max(new['hlr']) #np.log(max(new['snr']))
	snr_binslist = [snr_min+(x*((snr_max-snr_min)/10)) for x in range(11)]

	g1_true_snr=[]
	g1_obs_snr=[]
	g2_true_snr=[]
	g2_obs_snr=[]
	for p in range(10):
		mask = (np.log(new['snr']) >= snr_binslist[p]) & (np.log(new['snr']) < snr_binslist[p+1])
		#mask = (new['hlr'] >= snr_binslist[p]) & (new['hlr'] < snr_binslist[p+1])
		g1_true_snr.append(new['g1'][mask])
		g1_obs_snr.append(new['e1'][mask]/R11[p])

		g2_true_snr.append(new['g2'][mask])
		g2_obs_snr.append(new['e1'][mask]/R22[p])

	return g1_true_snr,g1_obs_snr,g2_true_snr,g2_obs_snr


def main(argv):

	g = 0.01
	old = None

	folder=['/hpc/group/cosmology/phy-lsst/my137/g1002/ngmix/','/hpc/group/cosmology/phy-lsst/my137/g1n002/ngmix/',
	'/hpc/group/cosmology/phy-lsst/my137/g2002/ngmix/','/hpc/group/cosmology/phy-lsst/my137/g2n002/ngmix/']
	dirr='fiducial_H158_'
	model='mcal'

	start = 0
	g1_true = []
	g2_true = []
	g1_obs = []
	g2_obs = []
	g1snr_true = []
	g2snr_true = []
	g1snr_obs = []
	g2snr_obs = []
    #object_number = 863305+863306+863306+863306
	for j in range(len(folder)):
		new_ = fio.FITS(folder[j]+dirr+model+'_noshear.fits')[-1].read()
		new1p_ = fio.FITS(folder[j]+dirr+model+'_1p.fits')[-1].read()
		new1m_ = fio.FITS(folder[j]+dirr+model+'_1m.fits')[-1].read()
		new2p_ = fio.FITS(folder[j]+dirr+model+'_2p.fits')[-1].read()
		new2m_ = fio.FITS(folder[j]+dirr+model+'_2m.fits')[-1].read()
		print(j,len(new_),len(new1p_),len(new1m_),len(new2p_),len(new2m_),start)
		object_number = len(new_['ind'])
		new   = np.zeros(object_number,dtype=new_.dtype)
		new1p = np.zeros(object_number,dtype=new_.dtype)
		new1m = np.zeros(object_number,dtype=new_.dtype)
		new2p = np.zeros(object_number,dtype=new_.dtype)
		new2m = np.zeros(object_number,dtype=new_.dtype)
		for col in new.dtype.names:
			new[col][start:start+len(new_)] += new_[col]
			new1p[col][start:start+len(new_)] += new1p_[col]
			new1m[col][start:start+len(new_)] += new1m_[col]
			new2p[col][start:start+len(new_)] += new2p_[col]
			new2m[col][start:start+len(new_)] += new2m_[col]
		start = 0
		#start+=len(new_)

        ## remove the extra object (object number = 40118)
		if (j==1 or j==2 or j==3):
			new = new[np.where(new['ind']!=40118)]
			new1p = new1p[np.where(new1p['ind']!=40118)]
			new1m = new1m[np.where(new1m['ind']!=40118)]
			new2p = new2p[np.where(new2p['ind']!=40118)]
			new2m = new2m[np.where(new2m['ind']!=40118)]

		gamma1_t,gamma2_t,gamma1_o,gamma2_o = analyze_gamma_obs(new,new1p,new1m,new2p,new2m)
		g1_true.append(gamma1_t)
		g2_true.append(gamma2_t)
		g1_obs.append(gamma1_o)
		g2_obs.append(gamma2_o)

		R11_correction, R22_correction = shear_response_correction(new,new1p,new1m,new2p,new2m)
		g1_true_snr,g1_obs_snr,g2_true_snr,g2_obs_snr = residual_bias_correction(new,new1p,new1m,new2p,new2m,R11_correction,R22_correction)
		g1snr_true.append(g1_true_snr)
		g1snr_obs.append(g1_obs_snr)
		g2snr_true.append(g2_true_snr)
		g2snr_obs.append(g2_obs_snr)

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
	"""
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

	print('off-diagonal cpomponents: ')
	print("m12="+str("%6.4f"% m12)+"+-"+str("%6.4f"% m12_err), "b12="+str("%6.6f"% c12)+"+-"+str("%6.6f"% c12_err))
	print("m21="+str("%6.4f"% m21)+"+-"+str("%6.4f"% m21_err), "b21="+str("%6.6f"% c21)+"+-"+str("%6.6f"% c21_err))

	print("before correction: ")
	print("m1="+str("%6.4f"% m11)+"+-"+str("%6.4f"% m11_err), "b1="+str("%6.6f"% c11)+"+-"+str("%6.6f"% c11_err))
	print("m2="+str("%6.4f"% m22)+"+-"+str("%6.4f"% m22_err), "b2="+str("%6.6f"% c22)+"+-"+str("%6.6f"% c22_err))
	"""

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
	print('corrected m, b: ')
	print("m1="+str("%6.4f"% np.mean(m11_snr))+"+-"+str("%6.4f"% np.mean(m11_snr_err)), "b1="+str("%6.6f"% np.mean(c11_snr))+"+-"+str("%6.6f"% np.mean(c11_snr_err)))
	print("m2="+str("%6.4f"% np.mean(m22_snr))+"+-"+str("%6.4f"% np.mean(m22_snr_err)), "b2="+str("%6.6f"% np.mean(c22_snr))+"+-"+str("%6.6f"% np.mean(c22_snr_err)))


	return None

if __name__ == "__main__":
    main(sys.argv)