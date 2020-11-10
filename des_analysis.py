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

def analyze_g12(new,new1p,new1m,new2p,new2m):
	g=0.01
	R11 = (new1p["e1"] - new1m["e1"])/(2*g)
	R22 = (new2p["e2"] - new2m["e2"])/(2*g)
	R12 = (new2p["e1"] - new2m["e1"])/(2*g)
	R21 = (new1p["e2"] - new1m["e2"])/(2*g)

	avg_R11 = np.mean(R11)
	avg_R22 = np.mean(R22)
	avg_R12 = np.mean(R12)
	avg_R21 = np.mean(R21)

	print("Mean shear response: ")
	N=len(new1p['e1'])
	print(N)
	print("<R11> = "+str("%6.4f"% avg_R11)+"+-"+str("%6.4f"% (np.std(R11)/np.sqrt(N))))
	print("<R22> = "+str("%6.4f"% avg_R22)+"+-"+str("%6.4f"% (np.std(R22)/np.sqrt(N))))
	print("<R12> = "+str("%6.4f"% avg_R12)+"+-"+str("%6.4f"% (np.std(R12)/np.sqrt(N))))
	print("<R21> = "+str("%6.4f"% avg_R21)+"+-"+str("%6.4f"% (np.std(R21)/np.sqrt(N))))

	def func(x,m,b):
		return (1+m)*x+b
	def func_off(x,m,b):
		return m*x+b

	gamma1_obs = new['e1']/avg_R11
	params2 = curve_fit(func,new['g1'],gamma1_obs,p0=(0.,0.))
	m5,b5=params2[0]
	m5err,b5err=np.sqrt(np.diagonal(params2[1]))

	gamma2_obs = new['e2']/avg_R22
	params2 = curve_fit(func,new['g2'],gamma2_obs,p0=(0.,0.))
	m6,b6=params2[0]
	m6err,b6err=np.sqrt(np.diagonal(params2[1]))

	## off-diagonal bias check
	params_off1 = curve_fit(func_off,new['g2'],gamma1_obs,p0=(0.,0.))
	params_off2 = curve_fit(func_off,new['g1'],gamma2_obs,p0=(0.,0.))
	m12, c12 = params_off1[0]
	m12_err, c12_err = np.sqrt(np.diagonal(params_off1[1]))
	m21, c21 = params_off2[0]
	m21_err, c21_err = np.sqrt(np.diagonal(params_off2[1]))

	print('off-diagonal cpomponents: ')
	print("m12="+str("%6.4f"% m12)+"+-"+str("%6.4f"% m12_err), "b12="+str("%6.6f"% c12)+"+-"+str("%6.6f"% c12_err))
	print("m21="+str("%6.4f"% m21)+"+-"+str("%6.4f"% m21_err), "b21="+str("%6.6f"% c21)+"+-"+str("%6.6f"% c21_err))
	print("before correction: ")
	print("m1="+str("%6.4f"% m5)+"+-"+str("%6.4f"% m5err), "b1="+str("%6.6f"% b5)+"+-"+str("%6.6f"% b5err))
	print("m2="+str("%6.4f"% m6)+"+-"+str("%6.4f"% m6err), "b2="+str("%6.6f"% b6)+"+-"+str("%6.6f"% b6err))

	return R11, R22, R12, R21, gamma1_obs, gamma2_obs

def residual_bias_correction(new, new1p, new1m, new2p, new2m):
    g = 0.01
    R11, R22, R12, R21, gamma1_obs, gamma2_obs = analyze_g12(new, new1p, new1m, new2p, new2m)

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
        
        #mask_1p = (new1p['hlr'] >= snr_binslist[i]) & (new1p['hlr'] < snr_binslist[i+1])
        #mask_1m = (new1m['hlr'] >= snr_binslist[i]) & (new1m['hlr'] < snr_binslist[i+1])
        #mask_2p = (new2p['hlr'] >= snr_binslist[i]) & (new2p['hlr'] < snr_binslist[i+1])
        #mask_2m = (new2m['hlr'] >= snr_binslist[i]) & (new2m['hlr'] < snr_binslist[i+1])
            
        #print("how many objects fall in each bin. ", len(mask_1p), len(mask_1m), len(mask_2p), len(mask_2m))
        
        R11_s[i] = (np.mean(new['e1'][mask_1p]) - np.mean(new['e1'][mask_1m]))/(2*g)
        R22_s[i] = (np.mean(new['e2'][mask_2p]) - np.mean(new['e2'][mask_2m]))/(2*g)
        R12_s[i] = (np.mean(new['e1'][mask_2p]) - np.mean(new['e1'][mask_2m]))/(2*g)
        R21_s[i] = (np.mean(new['e2'][mask_1p]) - np.mean(new['e2'][mask_1m]))/(2*g)

    ## total response
    tot_R11 = R11_g + R11_s
    tot_R22 = R22_g + R22_s
    tot_R12 = R12_g + R12_s
    tot_R21 = R21_g + R21_s
        
    ## get the m&b values for each bin
    from scipy.optimize import curve_fit
    def func(x,m,b):
      return (1+m)*x+b
    m1_val = []
    m1_err = []
    b1_val = []
    b1_err = []
    m2_val = []
    m2_err = []
    b2_val =[]
    b2_err = []

    for p in range(10):
        mask = (np.log(new['snr']) >= snr_binslist[p]) & (np.log(new['snr']) < snr_binslist[p+1])
        #mask = (new['hlr'] >= snr_binslist[p]) & (new['hlr'] < snr_binslist[p+1])

        params = curve_fit(func,new['g1'][mask],new['e1'][mask]/tot_R11[p],p0=(0.,0.))
        m1,b1=params[0]
        m1err,b1err=np.sqrt(np.diagonal(params[1]))

        params = curve_fit(func,new['g2'][mask],new['e2'][mask]/tot_R22[p],p0=(0.,0.))
        m2,b2=params[0]
        m2err,b2err=np.sqrt(np.diagonal(params[1]))
        
        # corrected
        m1_val.append(m1)
        m1_err.append(m1err)
        b1_val.append(b1)
        b1_err.append(b1err)
        m2_val.append(m2)
        m2_err.append(m2err)
        b2_val.append(b2)
        b2_err.append(b2err)

    print('corrected m, b: ')
    print("m1="+str("%6.4f"% np.mean(m1_val))+"+-"+str("%6.4f"% np.mean(m1_err)), "b1="+str("%6.6f"% np.mean(b1_val))+"+-"+str("%6.6f"% np.mean(b1_err)))
    print("m2="+str("%6.4f"% np.mean(m2_val))+"+-"+str("%6.4f"% np.mean(m2_err)), "b2="+str("%6.6f"% np.mean(b2_val))+"+-"+str("%6.6f"% np.mean(b2_err)))

    print(m1_val, m1_err, m2_val, m2_err)

    values=[m1_val,b1_val,m2_val,b2_val]
    errors=[m1_err,b1_err,m2_err,b2_err]
    return values, errors, snr_binslist



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
    
	def func(x,m,b):
		return (1+m)*x+b
	def func_off(x,m,b):
		return m*x+b

	## bootstrap covariance function. 
	def bootstrap_cov_m(N,data1,data2):
		fi = []
		for n in range(N):
			sample = np.random.choice(np.arange(len(data1)),len(data1),replace=True)
			fi.append((np.mean(data1[sample]) - np.mean(data2[sample]))/0.04)
		f_mean = np.sum(fi)/N 
		fi = np.array(fi)
		cov = np.sqrt(np.sum((fi-f_mean)**2)/(N-1))
		return cov

	def bootstrap_cov_c(N,m,data1,data2,data3,data4):
		fi = []
		for n in range(N):
			sample = np.random.choice(np.arange(len(data1)),len(data1),replace=True)
			function = ((data1[sample]-(1+m)*data2[sample]) + (data3[sample] - (1+m)*data4[sample]))/2
			fi.append(function)
		f_mean = np.sum(fi)/N 
		fi = np.array(fi)
		cov = np.sqrt(np.sum((fi-f_mean)**2)/(N-1))
		return cov

	## m1,c1 calculation
	m11 = ((np.mean(g1_obs[0])-np.mean(g1_obs[1]))/0.04) - 1
	m11_err = bootstrap_cov_m(200,g1_obs[0],g1_obs[1])
	c11p = np.mean(g1_obs[0] - (1+m11)*g1_true[0])
	c11n = np.mean(g1_obs[1] - (1+m11)*g1_true[1])
	c11 = (c11p + c11n)/2
	c11_err = bootstrap_cov_c(200,m11,g1_obs[0],g1_true[0],g1_obs[1],g1_true[1])

	## m2,c2 calculation
	m22 = ((np.mean(g2_obs[2])-np.mean(g2_obs[3]))/0.04) - 1
	m22_err = bootstrap_cov_m(200,g2_obs[2],g2_obs[3])
	c22p = np.mean(g2_obs[2] - (1+m22)*g2_true[2])
	c22n = np.mean(g2_obs[3] - (1+m22)*g2_true[3])
	c22 = (c22p + c22n)/2
	c22_err = bootstrap_cov_c(200,m22,g2_obs[2],g2_true[2],g2_obs[3],g2_true[3])

	## off-diagonal components
	m12 = ((np.mean(g1_obs[2])-np.mean(g1_obs[3]))/0.04) - 1
	m12_err = bootstrap_cov_m(200,g1_obs[2],g1_obs[3])
	c12p = np.mean(g1_obs[2] - (m12)*g1_true[2])
	c12n = np.mean(g1_obs[3] - (m12)*g1_true[3])
	c12 = (c12p + c12n)/2
	c12_err = bootstrap_cov_c(200,m12,g1_obs[2],g1_true[2],g1_obs[3],g1_true[3])

	m21 = ((np.mean(g2_obs[0])-np.mean(g2_obs[1]))/0.04) - 1
	m21_err = bootstrap_cov_m(200,g2_obs[0],g2_obs[1])
	c21p = np.mean(g2_obs[0] - (m21)*g2_true[0])
	c21n = np.mean(g2_obs[1] - (m21)*g2_true[1])
	c21 = (c21p + c21n)/2
	c21_err = bootstrap_cov_c(200,m21,g2_obs[0],g2_true[0],g2_obs[1],g2_true[1])

	#print(m11,c11,m22,c22,m12,c12,m21,c21)

	print('off-diagonal cpomponents: ')
	print("m12="+str("%6.4f"% m12)+"+-"+str("%6.4f"% m12_err), "b12="+str("%6.6f"% c12)+"+-"+str("%6.6f"% c12_err))
	print("m21="+str("%6.4f"% m21)+"+-"+str("%6.4f"% m21_err), "b21="+str("%6.6f"% c21)+"+-"+str("%6.6f"% c21_err))

	print("before correction: ")
	print("m1="+str("%6.4f"% m11)+"+-"+str("%6.4f"% m11_err), "b1="+str("%6.6f"% c11)+"+-"+str("%6.6f"% c11_err))
	print("m2="+str("%6.4f"% m22)+"+-"+str("%6.4f"% m22_err), "b2="+str("%6.6f"% c22)+"+-"+str("%6.6f"% c22_err))

	"""
	correction = sys.argv[1]
	if correction == True:

		start = 0
		object_number = 863305+863306+863306+863306
		for j in range(len(folder)):
			new_ = fio.FITS(folder[j]+dirr+model+'_noshear.fits')[-1].read()
			new1p_ = fio.FITS(folder[j]+dirr+model+'_1p.fits')[-1].read()
			new1m_ = fio.FITS(folder[j]+dirr+model+'_1m.fits')[-1].read()
			new2p_ = fio.FITS(folder[j]+dirr+model+'_2p.fits')[-1].read()
			new2m_ = fio.FITS(folder[j]+dirr+model+'_2m.fits')[-1].read()
			print(j,len(new_),len(new1p_),len(new1m_),len(new2p_),len(new2m_),start)
			#object_number = len(new_['ind'])
			new = np.zeros(object_number,dtype=new_.dtype)
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
		start+=len(new_)

	R11, R22, R12, R21, gamma1_obs, gamma2_obs = analyze_g12(new,new1p,new1m,new2p,new2m)
	values, errors, snr_binslist = residual_bias_correction(new,new1p,new1m,new2p,new2m)
	"""

	return None

if __name__ == "__main__":
    main(sys.argv)