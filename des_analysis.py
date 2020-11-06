import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def analyze_g1(new,new1p,new1m,new2p,new2m):

	R11 = (new1p["e1"] - new1m["e1"])/(2*g)
	R22 = (new2p["e2"] - new2m["e2"])/(2*g)
	R12 = (new2p["e1"] - new2m["e1"])/(2*g)
	R21 = (new1p["e2"] - new1m["e2"])/(2*g)

	avg_R11 = np.mean(R11)
	avg_R22 = np.mean(R22)
	avg_R12 = np.mean(R12)
	avg_R21 = np.mean(R21)

	gamma1_obs = new['e1']/avg_R11

    return new['g1'], gamma1_obs

def analyze_g2(new,new1p,new1m,new2p,new2m):

	R11 = (new1p["e1"] - new1m["e1"])/(2*g)
	R22 = (new2p["e2"] - new2m["e2"])/(2*g)
	R12 = (new2p["e1"] - new2m["e1"])/(2*g)
	R21 = (new1p["e2"] - new1m["e2"])/(2*g)

	avg_R11 = np.mean(R11)
	avg_R22 = np.mean(R22)
	avg_R12 = np.mean(R12)
	avg_R21 = np.mean(R21)

	gamma2_obs = new['e2']/avg_R22

    return new['g2'], gamma2_obs



def main(argv):

    g = 0.01
    old = None

    folder=['/hpc/group/cosmology/phy-lsst/my137/g1002/ngmix/','/hpc/group/cosmology/phy-lsst/my137/g1n002/ngmix/',
    '/hpc/group/cosmology/phy-lsst/my137/g2002/ngmix/','/hpc/group/cosmology/phy-lsst/my137/g2n002/ngmix/']
    dirr='fiducial_H158_mcal_'
    model='mcal'

    start = 0
    #object_number = 863305+863306+863306+863306
    for j in range(len(folder)):
        new_ = fio.FITS(folder[j]+dirr+model+'_noshear.fits')[-1].read()
        new1p_ = fio.FITS(folder[j]+dirr+model+'_1p.fits')[-1].read()
        new1m_ = fio.FITS(folder[j]+dirr+model+'_1m.fits')[-1].read()
        new2p_ = fio.FITS(folder[j]+dirr+model+'_2p.fits')[-1].read()
        new2m_ = fio.FITS(folder[j]+dirr+model+'_2m.fits')[-1].read()
        print(j,len(new_),len(new1p_),len(new1m_),len(new2p_),len(new2m_),start)
        if j==0:
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
        start+=len(new_)

        if j==0:
    		g1_true, g1_obs = analyze_g1(new,new1p,new1m,new2p,new2m)
    	elif j==1:
    		g1n_true, g1n_obs = analyze_g1(new,new1p,new1m,new2p,new2m)
    	elif j==2:
    		g2_true, g2_obs = analyze_g2(new,new1p,new1m,new2p,new2m)
    	elif j==3:
    		g2n_true, g2n_obs = analyze_g2(new,new1p,new1m,new2p,new2m)
    
    def func(x,m,b):
          return (1+m)*x+b
    def func_off(x,m,b):
        return m*x+b

    gamma1_true = np.concatenate((g1_true,g1n_true))
    gamma1_obs = np.concatenate((g1_obs,g1n_obs))
    params2 = curve_fit(func,gamma1_true,gamma1_obs,p0=(0.,0.))
    m5,b5=params2[0]
    m5err,b5err=np.sqrt(np.diagonal(params2[1]))

    gamma2_true = np.concatenate((g2_true,g2n_true))
    gamma2_obs = np.concatenate((g2_obs,g2n_obs))
    params2 = curve_fit(func,gamma2_true,gamma2_obs,p0=(0.,0.))
    m6,b6=params2[0]
    m6err,b6err=np.sqrt(np.diagonal(params2[1]))

    ## off-diagonal bias check
    params_off1 = curve_fit(func_off,gamma2_true,gamma1_obs,p0=(0.,0.))
    params_off2 = curve_fit(func_off,gamma1_true,gamma2_obs,p0=(0.,0.))
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

    return None

if __name__ == "__main__":
    main(sys.argv)