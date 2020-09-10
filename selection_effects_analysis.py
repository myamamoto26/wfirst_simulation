from __future__ import division
from __future__ import print_function

from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from past.builtins import basestring
from builtins import object
from past.utils import old_div
import numpy as np
#import healpy as hp
import sys, os, io
import math
import logging
import time
import yaml
import copy
import galsim as galsim
import galsim.wfirst as wfirst
#wfirst.pixel_scale /= 4
import galsim.config.process as process
import galsim.des as des
import ngmix
import fitsio as fio
#import pickle as pickle
#import pickletools
from astropy.time import Time
from astropy.table import Table
from mpi4py import MPI
#from mpi_pool import MPIPool
#import cProfile, pstats
#import glob
#import shutil
from ngmix.jacobian import Jacobian
from ngmix.observation import Observation, ObsList, MultiBandObsList,make_kobs
from ngmix.galsimfit import GalsimRunner,GalsimSimple,GalsimTemplateFluxFitter
from ngmix.guessers import R50FluxGuesser
from ngmix.bootstrap import PSFRunner
from ngmix import priors, joint_prior
import mof
import meds
#import psc

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

def residual_bias_quad(res_tot):

    g = 0.01

    new = res_tot[0]
    new1p = res_tot[1]
    new1m = res_tot[2]
    new2p = res_tot[3]
    new2m = res_tot[4]

    new = new[new['ra']!=0]
    new1p = new1p[new1p['ra']!=0]
    new1m = new1m[new1m['ra']!=0]
    new2p = new2p[new2p['ra']!=0]
    new2m = new2m[new2m['ra']!=0]

    R11 = (new1p["e1"] - new1m["e1"])/(2*g)
    R22 = (new2p["e2"] - new2m["e2"])/(2*g)
    R12 = (new2p["e1"] - new2m["e1"])/(2*g)
    R21 = (new1p["e2"] - new1m["e2"])/(2*g)

    avg_R11 = np.mean(R11)
    avg_R22 = np.mean(R22)
    avg_R12 = np.mean(R12)
    avg_R21 = np.mean(R21)

    g1_obs = new['e1']/avg_R11
    g2_obs = new['e2']/avg_R22

    ## some statistics
    print("Mean shear response: ")
    N=len(new1p['e1'])
    print(N)
    print("<R11> = "+str("%6.4f"% avg_R11)+"+-"+str("%6.4f"% (np.std(R11)/np.sqrt(N))))
    print("<R22> = "+str("%6.4f"% avg_R22)+"+-"+str("%6.4f"% (np.std(R22)/np.sqrt(N))))
    print("<R12> = "+str("%6.4f"% avg_R12)+"+-"+str("%6.4f"% (np.std(R12)/np.sqrt(N))))
    print("<R21> = "+str("%6.4f"% avg_R21)+"+-"+str("%6.4f"% (np.std(R21)/np.sqrt(N))))

    print(np.mean(g1_obs), np.std(g1_obs)/np.sqrt(N))
    def f(g_true, *coeffs):
        return (coeffs[2]*(g_true**3)
                +coeffs[1]*g_true
                +coeffs[0])
    #initalize coefficents to 1 except for c - set to zero.
    start = np.array([1.,1.,0.])

    coeffs_max, coeffs_cov = curve_fit(f, new['g1'], g1_obs, p0=start)
    print(coeffs_max, np.sqrt(np.diagonal(coeffs_cov)))
    coeffs_max2, coeffs_cov2 = curve_fit(f, new['g2'], g2_obs, p0=start)
    print(coeffs_max2, np.sqrt(np.diagonal(coeffs_cov2)))

    def func(x,m,b):
            return (1+m)*x+b

    params2 = curve_fit(func,new['g1'],g1_obs,p0=(0.,0.))
    m5,b5=params2[0]
    m5err,b5err=np.sqrt(np.diagonal(params2[1]))

    params2 = curve_fit(func,new['g2'],g2_obs,p0=(0.,0.))
    m6,b6=params2[0]
    m6err,b6err=np.sqrt(np.diagonal(params2[1]))

    #x=np.linspace(-0.1,0.1,1000)
    #y=coeffs_max[2]*(x**3)+coeffs_max[1]*x+coeffs_max[0]
    #y2=(1+m5)*x + b5
    #plt.plot(x,y)
    #plt.plot(x,y2)
    #plt.scatter(new['g1'], g1_obs, s=1)
    #plt.savefig('mcal_quad_fit.png')

    return None

def residual_bias(new, new1p, new1m, new2p, new2m, shape):

    if shape=='ngmix':
        R11=None
        R22=None
        R12=None
        R21=None

        def func(x,m,b):
            return (1+m)*x+b

        gamma1_obs = new['e1']
        #print(np.mean(gamma1_obs[0:N:2]), np.std(gamma1_obs[0:N:2])/np.sqrt(len(gamma1_obs[0:N:2])), np.mean(gamma1_obs[1:N:2]), np.std(gamma1_obs[1:N:2])/np.sqrt(len(gamma1_obs[1:N:2])))
        params2 = curve_fit(func,new['g1'],new['e1'],p0=(0.,0.))
        m5,b5=params2[0]
        m5err,b5err=np.sqrt(np.diagonal(params2[1]))

        gamma2_obs = new['e2']
        params2 = curve_fit(func,new['g2'],new['e2'],p0=(0.,0.))
        m6,b6=params2[0]
        m6err,b6err=np.sqrt(np.diagonal(params2[1]))

        print("before correction: ")
        print("m1="+str("%6.4f"% m5)+"+-"+str("%6.4f"% m5err), "b1="+str("%6.6f"% b5)+"+-"+str("%6.6f"% b5err))
        print("m2="+str("%6.4f"% m6)+"+-"+str("%6.4f"% m6err), "b2="+str("%6.6f"% b6)+"+-"+str("%6.6f"% b6err))

        return R11, R22, R12, R21, gamma1_obs, gamma2_obs

    elif shape=='mcal':
        g = 0.01

        """
        mask = (new['e1']>=-1) & (new['e1']<1)
        mask1p = (new1p['e1']>=-1) & (new1p['e1']<1)
        mask1m = (new1m['e1']>=-1) & (new1m['e1']<1)
        mask2p = (new2p['e2']>=-1) & (new2p['e2']<1)
        mask2m = (new2m['e2']>=-1) & (new2m['e2']<1)
        mask_all = (mask==True) & (mask1p==True) & (mask1m==True) & (mask2p==True) & (mask2m==True)
        print(len(mask), len(mask_all))
        #old = old[old['ra']!=0]
        new = new[mask_all]
        new1p = new1p[mask_all]
        new1m = new1m[mask_all]
        new2p = new2p[mask_all]
        new2m = new2m[mask_all]
        """
        
        R11 = (new1p["e1"] - new1m["e1"])/(2*g)
        R22 = (new2p["e2"] - new2m["e2"])/(2*g)
        R12 = (new2p["e1"] - new2m["e1"])/(2*g)
        R21 = (new1p["e2"] - new1m["e2"])/(2*g)

        avg_R11 = np.mean(R11)
        avg_R22 = np.mean(R22)
        avg_R12 = np.mean(R12)
        avg_R21 = np.mean(R21)

        ## some statistics
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

def residual_bias_correction(new, new1p, new1m, new2p, new2m, shape):
    g = 0.01
    R11, R22, R12, R21, gamma1_obs, gamma2_obs = residual_bias(new, new1p, new1m, new2p, new2m, shape)

    avg_R11 = np.mean(R11)
    avg_R22 = np.mean(R22)

    snr_binn = 10
    snr_min = np.log(15) #np.min(new['hlr']) #np.log(15) #np.log(min(new['snr']))
    snr_max = np.log(500) #np.max(new['hlr']) #np.log(max(new['snr']))
    snr_binslist = [snr_min+(x*((snr_max-snr_min)/10)) for x in range(11)]
    #print(snr_min, snr_max, snr_binslist)
    if snr_binslist[10] != snr_max:
        print("raise an error.")

    R11_g = []
    R22_g = []
    R12_g = []
    R21_g = []
    for a in range(10):
        mask = (np.log(new['snr']) >= snr_binslist[a]) & (np.log(new['snr']) < snr_binslist[a+1])

        R11_g.append(np.mean(R11[mask]))
        R22_g.append(np.mean(R22[mask]))
        R12_g.append(np.mean(R12[mask]))
        R21_g.append(np.mean(R21[mask]))

    ## getting cuts on the snr from the sheared catalogs and calculating selection response <R>selection
    R11_s = []
    R22_s = []
    R12_s = []
    R21_s = []
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
        
        R11_s.append((np.mean(new['e1'][mask_1p]) - np.mean(new['e1'][mask_1m]))/(2*g))
        R22_s.append((np.mean(new['e2'][mask_2p]) - np.mean(new['e2'][mask_2m]))/(2*g))
        R12_s.append((np.mean(new['e1'][mask_2p]) - np.mean(new['e1'][mask_2m]))/(2*g))
        R21_s.append((np.mean(new['e2'][mask_1p]) - np.mean(new['e2'][mask_1m]))/(2*g))

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

def plot_combined(g1values,g1errors,g2values,g2errors,snr_binslist):

    m1_val=g1values[0]
    b1_val=g1values[1]
    m2_val=g2values[2]
    b2_val=g2values[3]
    m3_val=g1values[4]
    b3_val=g1values[5]
    m4_val=g2values[6]
    b4_val=g2values[7]

    m1_err=g1errors[0]
    b1_err=g1errors[1]
    m2_err=g2errors[2]
    b2_err=g2errors[3]
    m3_err=g1errors[4]
    b3_err=g1errors[5]
    m4_err=g2errors[6]
    b4_err=g2errors[7]

    fig, ax3 = plt.subplots(figsize=(8,6))
    bins_loc = [(snr_binslist[x]+snr_binslist[x+1])/2 for x in range(10)]

    ax3.scatter(bins_loc, m1_val, c='b', marker='.', label='m1 w/ response correction')
    ax3.scatter(bins_loc, m2_val, c='r', marker='.',  label='m2 w/ response correction')
    ax3.errorbar(bins_loc, m1_val, c='b', yerr=m1_err, fmt='.')
    ax3.errorbar(bins_loc, m2_val, c='r', yerr=m2_err, fmt='.')

    
    ax3.scatter(bins_loc, m3_val, c='b', marker='^', label='m1 w/o response correction')
    ax3.scatter(bins_loc, m4_val, c='r', marker='^', label='m2 w/o response correction')
    ax3.errorbar(bins_loc, m3_val, c='b', yerr=m3_err, fmt='^')
    ax3.errorbar(bins_loc, m4_val, c='r', yerr=m4_err, fmt='^')
    

    ax3.legend(fontsize=9)
    ax3.set_xlabel("log(SNR)", fontsize=18)
    #ax3.set_xlabel("Half-Light Radius", fontsize=20)
    #ax3.set_ylabel(r"$R_{ij}$", fontsize=20)
    ax3.set_ylabel(r"$m_{i}$", fontsize=16)
    ax3.tick_params(labelsize=12)
    plt.savefig('v1_23comb_residualbias.png')

def main(argv):

    shape=sys.argv[1]
    if shape=='metacal_all':
        g = 0.01
        old = None
        """
        folder='/hpc/group/cosmology/phy-lsst/my137/roman_simple_mcal/v2/'
        dirr=['v2_3_', 'v2_3_seed2_', 'v2_3_seed3_', 'v2_3_seed4_', 'v2_3_seed5_']
        dirr2=[ 'v2_4_seed3_', 'v2_4_seed3_', 'v2_4_seed4_', 'v2_4_seed5_',
                'v2_4_seed6_', 'v2_4_seed7_', 'v2_4_seed8_', 'v2_4_seed9_']
        model='sim' # choice: metacal
        """
        folder='/hpc/group/cosmology/phy-lsst/my137/ngmix'
        dirr=['fiducial_H158_2290725_0_', 'fiducial_H158_2290725_1_', 'fiducial_H158_2290725_2_', 'fiducial_H158_2290725_3_', 'fiducial_H158_2290725_4_']
        model='mcal_'
        #f = open('meds_number.txt', 'r')
        #medsn = f.read().split('\n')

        start = 0
        """
        for j,pix in enumerate(medsn):
            for i in range(5):
                new_ = fio.FITS(dirr+'/fiducial_H158_'+str(pix)+'_'+str(i)+'_'+model+'_noshear.fits')[-1].read()
                new1p_ = fio.FITS(dirr+'/fiducial_H158_'+str(pix)+'_'+str(i)+'_'+model+'_1p.fits')[-1].read()
                new1m_ = fio.FITS(dirr+'/fiducial_H158_'+str(pix)+'_'+str(i)+'_'+model+'_1m.fits')[-1].read()
                new2p_ = fio.FITS(dirr+'/fiducial_H158_'+str(pix)+'_'+str(i)+'_'+model+'_2p.fits')[-1].read()
                new2m_ = fio.FITS(dirr+'/fiducial_H158_'+str(pix)+'_'+str(i)+'_'+model+'_2m.fits')[-1].read()
                print(j,i,len(new_),len(new1p_),len(new1m_),len(new2p_),len(new2m_),start,len(new))
                if (j==0)&(i==0):
                    new   = np.zeros(2500*len(medsn),dtype=new_.dtype)
                    new1p = np.zeros(2500*len(medsn),dtype=new_.dtype)
                    new1m = np.zeros(2500*len(medsn),dtype=new_.dtype)
                    new2p = np.zeros(2500*len(medsn),dtype=new_.dtype)
                    new2m = np.zeros(2500*len(medsn),dtype=new_.dtype)
                else:
                    for col in new.dtype.names:
                        new[col][start:start+len(new_)] += new_[col]
                        new1p[col][start:start+len(new_)] += new1p_[col]
                        new1m[col][start:start+len(new_)] += new1m_[col]
                        new2p[col][start:start+len(new_)] += new2p_[col]
                        new2m[col][start:start+len(new_)] += new2m_[col]
            start+=len(new_)
        """
        #object_number = 10000000
        for j in range(len(dirr)):
            new_ = fio.FITS(folder+dirr[j]+model+'_noshear.fits')[-1].read()
            new1p_ = fio.FITS(folder+dirr[j]+model+'_1p.fits')[-1].read()
            new1m_ = fio.FITS(folder+dirr[j]+model+'_1m.fits')[-1].read()
            new2p_ = fio.FITS(folder+dirr[j]+model+'_2p.fits')[-1].read()
            new2m_ = fio.FITS(folder+dirr[j]+model+'_2m.fits')[-1].read()
            print(j,len(new_),len(new1p_),len(new1m_),len(new2p_),len(new2m_),start)
            if j==0:
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


    if shape=='mcal':
        dirr=['/hpc/group/cosmology/phy-lsst/my137/ngmix/fiducial_H158_2290725_0']
        for i in range(len(dirr)):
            a=fio.FITS(dirr[i]+'_mcal_noshear.fits')[-1].read() 
            b=fio.FITS(dirr[i]+'_mcal_1p.fits')[-1].read()
            c=fio.FITS(dirr[i]+'_mcal_1m.fits')[-1].read()
            d=fio.FITS(dirr[i]+'_mcal_2p.fits')[-1].read()
            e=fio.FITS(dirr[i]+'_mcal_2m.fits')[-1].read()

            residual_bias_correction(a,b,c,d,e,shape)
            #g_values,g_errors,snr_binslist = residual_bias_correction(a,b,c,d,e, 'metacal')

    elif shape=='ngmix':
        for i in range(len(dirr)):
            a=fio.FITS(dirr[i]+'_ngmix_0.fits')[-1].read() 
            b=None
            c=None
            d=None
            e=None

            g_values,g_errors,snr_binslist = residual_bias([a,b,c,d,e], shape)

    elif shape=='metacal_all':
        g_values,g_errors,snr_binslist = residual_bias_correction(new,new1p,new1m,new2p,new2m,shape)
    
    return None

if __name__ == "__main__":
    main(sys.argv)