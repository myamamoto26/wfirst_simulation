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

    #new = new[new['ra']!=0]
    #new1p = new1p[new1p['ra']!=0]
    #new1m = new1m[new1m['ra']!=0]
    #new2p = new2p[new2p['ra']!=0]
    #new2m = new2m[new2m['ra']!=0]

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

def residual_bias(res_tot, shape):

    if shape=='ngmix':
        new = res_tot[0]

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

    elif shape=='metacal':
        g = 0.01

        new = res_tot[0]
        new1p = res_tot[1]
        new1m = res_tot[2]
        new2p = res_tot[3]
        new2m = res_tot[4]

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
        
        R11 = (new1p["e1"] - new1m["e1"])/(2*g)
        R22 = (new2p["e2"] - new2m["e2"])/(2*g)
        R12 = (new2p["e1"] - new2m["e1"])/(2*g)
        R21 = (new1p["e2"] - new1m["e2"])/(2*g)

        avg_R11 = np.mean(R11)
        avg_R22 = np.mean(R22)
        avg_R12 = np.mean(R12)
        avg_R21 = np.mean(R21)

        #g1 = new['e1']/avg_R11
        #g2 = new['e2']/avg_R22

        ## some statistics
        print("Mean shear response: ")
        N=len(new1p['e1'])
        print(N)
        print("<R11> = "+str("%6.4f"% avg_R11)+"+-"+str("%6.4f"% (np.std(R11)/np.sqrt(N))))
        print("<R22> = "+str("%6.4f"% avg_R22)+"+-"+str("%6.4f"% (np.std(R22)/np.sqrt(N))))
        print("<R12> = "+str("%6.4f"% avg_R12)+"+-"+str("%6.4f"% (np.std(R12)/np.sqrt(N))))
        print("<R21> = "+str("%6.4f"% avg_R21)+"+-"+str("%6.4f"% (np.std(R21)/np.sqrt(N))))

        """
        coeffs, coeff_cov = get_coeffs(new['g1'], new['e1']/avg_R11,
                                       g_cov=None, cubic=False)
        print("m = %f +- %f"%(coeffs[1]-1,
                                  np.sqrt(coeff_cov[1,1])))
        print("c = %f +- %f"%(coeffs[0], np.sqrt(coeff_cov[0,0])))
        exit()
        """

        def func(x,m,b):
          return (1+m)*x+b

        gamma1_obs = new['e1']/avg_R11
        #print(np.mean(gamma1_obs[0:N:2]), np.std(gamma1_obs[0:N:2])/np.sqrt(len(gamma1_obs[0:N:2])), np.mean(gamma1_obs[1:N:2]), np.std(gamma1_obs[1:N:2])/np.sqrt(len(gamma1_obs[1:N:2])))
        params2 = curve_fit(func,new['g1'],new['e1']/avg_R11,p0=(0.,0.))
        m5,b5=params2[0]
        m5err,b5err=np.sqrt(np.diagonal(params2[1]))

        gamma2_obs = new['e2']/avg_R22
        params2 = curve_fit(func,new['g2'],new['e2']/avg_R22,p0=(0.,0.))
        m6,b6=params2[0]
        m6err,b6err=np.sqrt(np.diagonal(params2[1]))

        print("before correction: ")
        print("m1="+str("%6.4f"% m5)+"+-"+str("%6.4f"% m5err), "b1="+str("%6.6f"% b5)+"+-"+str("%6.6f"% b5err))
        print("m2="+str("%6.4f"% m6)+"+-"+str("%6.4f"% m6err), "b2="+str("%6.6f"% b6)+"+-"+str("%6.6f"% b6err))

        return R11, R22, R12, R21, gamma1_obs, gamma2_obs

def residual_bias_correction(a, b, c, d, e, shape):
    g = 0.01
    new = a
    new1p = b
    new1m = c
    new2p = d
    new2m = e

    R11, R22, R12, R21, gamma1_obs, gamma2_obs = residual_bias([a,b,c,d,e], shape)

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
    R11_gerr = []
    R22_gerr = []
    R12_gerr = []
    R21_gerr = []
    for a in range(10):
        bin_R11 = []
        bin_R22 = []
        bin_R12 = []
        bin_R21 = []
        for b in range(len(R11)):
            if (np.log(new['snr'][b]) >= snr_binslist[a]) and (np.log(new['snr'][b]) < snr_binslist[a+1]):
            #if (new['hlr'][b] >= snr_binslist[a]) and (new['hlr'][b] < snr_binslist[a+1]):
                bin_R11 += [R11[b]]
                bin_R22 += [R22[b]]
                bin_R12 += [R12[b]]
                bin_R21 += [R21[b]]
        #print(len(bin_R11))
        R11_g += [np.mean(bin_R11)]
        R22_g += [np.mean(bin_R22)]
        R12_g += [np.mean(bin_R12)]
        R21_g += [np.mean(bin_R21)]
        R11_gerr += [np.std(bin_R11)/np.sqrt(len(bin_R11))]
        R22_gerr += [np.std(bin_R22)/np.sqrt(len(bin_R22))]
        R12_gerr += [np.std(bin_R12)/np.sqrt(len(bin_R12))]
        R21_gerr += [np.std(bin_R21)/np.sqrt(len(bin_R21))]

    ## getting cuts on the snr from the sheared catalogs and calculating selection response <R>selection
    R11_s = []
    R22_s = []
    R12_s = []
    R21_s = []
    R11_serr = []
    R22_serr = []
    R12_serr = []
    R21_serr = []
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
        
        R11_s += [(np.mean(new['e1'][mask_1p]) - np.mean(new['e1'][mask_1m]))/(2*g)]
        R22_s += [(np.mean(new['e2'][mask_2p]) - np.mean(new['e2'][mask_2m]))/(2*g)]
        R12_s += [(np.mean(new['e1'][mask_2p]) - np.mean(new['e1'][mask_2m]))/(2*g)]
        R21_s += [(np.mean(new['e2'][mask_1p]) - np.mean(new['e2'][mask_1m]))/(2*g)]

    #print("to check if there is no nan or inf", R11_s, R11_g)
    #print(R11_s)
    if len(R11_s) != 10:
        print('it is not 10 bins!')

    ## total response
    tot_R11 = []
    tot_R22 = []
    tot_R12 = []
    tot_R21 = []
    for k in range(10):
        tot_R11 += [R11_s[k] + R11_g[k]]
        tot_R22 += [R22_s[k] + R22_g[k]]
        tot_R12 += [R12_s[k] + R12_g[k]]
        tot_R21 += [R21_s[k] + R21_g[k]]
        
        
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
    m3_val = []
    m3_err = []
    b3_val = []
    b3_err = []
    m4_val = []
    m4_err = []
    b4_val =[]
    b4_err = []

    for p in range(10):
        mask = (np.log(new['snr']) >= snr_binslist[p]) & (np.log(new['snr']) < snr_binslist[p+1])
        #mask = (new['hlr'] >= snr_binslist[p]) & (new['hlr'] < snr_binslist[p+1])

        #gamma1_obs_corr[mask] = new['e1'][mask]/tot_R11[p]
        params = curve_fit(func,new['g1'][mask],new['e1'][mask]/tot_R11[p],p0=(0.,0.))
        m1,b1=params[0]
        m1err,b1err=np.sqrt(np.diagonal(params[1]))
        #gamma2_obs_corr[mask] = new['e2'][mask]/tot_R22[p]
        params = curve_fit(func,new['g2'][mask],new['e2'][mask]/tot_R22[p],p0=(0.,0.))
        m2,b2=params[0]
        m2err,b2err=np.sqrt(np.diagonal(params[1]))
        
        params = curve_fit(func,new['g1'][mask],new['e1'][mask]/R11_g[p],p0=(0.,0.))
        m3,b3=params[0]
        m3err,b3err=np.sqrt(np.diagonal(params[1]))
        params = curve_fit(func,new['g2'][mask],new['e2'][mask]/R22_g[p],p0=(0.,0.))
        m4,b4=params[0]
        m4err,b4err=np.sqrt(np.diagonal(params[1]))
        
        # corrected
        m1_val += [m1]
        m1_err += [m1err]
        b1_val += [b1]
        b1_err += [b1err]
        m2_val += [m2]
        m2_err += [m2err]
        b2_val += [b2]
        b2_err += [b2err]
        
        # not corrected
        m3_val += [m3]
        m3_err += [m3err]
        b3_val += [b3]
        b3_err += [b3err]
        m4_val += [m4]
        m4_err += [m4err]
        b4_val += [b4]
        b4_err += [b4err]

    print('corrected m, b: ')
    print("m1="+str("%6.4f"% np.mean(m1_val))+"+-"+str("%6.4f"% np.mean(m1_err)), "b1="+str("%6.6f"% np.mean(b1_val))+"+-"+str("%6.6f"% np.mean(b1_err)))
    print("m2="+str("%6.4f"% np.mean(m2_val))+"+-"+str("%6.4f"% np.mean(m2_err)), "b2="+str("%6.6f"% np.mean(b2_val))+"+-"+str("%6.6f"% np.mean(b2_err)))

    #t = Table([gamma1_obs, gamma2_obs, gamma1_obs_corr, gamma2_obs_corr], names=('gamma1_obs', 'gamma2_obs', 'gamma1_obs_corr', 'gamma2_obs_corr'))
    #t.write('delta_measuredshape_'+fname+'.fits', format=fits)
    #print()

    values=[m1_val,b1_val,m2_val,b2_val,m3_val,b3_val,m4_val,b4_val]
    errors=[m1_err,b1_err,m2_err,b2_err,m3_err,b3_err,m4_err,b4_err]
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
    #dirr=['v2_7_offset_0', 'v2_8_offset_0', 'v2_7_offset_10', 'v2_8_offset_10', 'v2_7_offset_45', 'v2_8_offset_45']
    #off=['g1_off0', 'g2_off0', 'g1_off10', 'g2_off10', 'g1_off45', 'g2_off45']
    #dirr=['../fiducial_H158'] #['v2_9_offset_0_rand20', 'v2_9_offset_0_rand360', 'v2_9_offset_45_rand20', 'v2_9_offset_45_rand360']
    dirr=sys.argv[1]
    shape=sys.argv[2]

    if shape=='metacal_quad':
        for i in range(len(dirr)):
            a=fio.FITS(dirr[i]+'_metacal_noshear.fits')[-1].read() 
            b=fio.FITS(dirr[i]+'_metacal_1p.fits')[-1].read()
            c=fio.FITS(dirr[i]+'_metacal_1m.fits')[-1].read()
            d=fio.FITS(dirr[i]+'_metacal_2p.fits')[-1].read()
            e=fio.FITS(dirr[i]+'_metacal_2m.fits')[-1].read()

            residual_bias_quad([a,b,c,d,e])
            #g_values,g_errors,snr_binslist = residual_bias_correction(a,b,c,d,e, 'metacal')

    elif shape=='ngmix':
        for i in range(len(dirr)):
            a=fio.FITS(dirr[i]+'_ngmix_0.fits')[-1].read() 
            b=None
            c=None
            d=None
            e=None

            g_values,g_errors,snr_binslist = residual_bias([a,b,c,d,e], shape)

    elif shape=='metacal':
        for i in range(len(dirr)):
            a=fio.FITS(dirr[i]+'_sim_0.fits')[-1].read() 
            b=fio.FITS(dirr[i]+'_sim_1.fits')[-1].read()
            c=fio.FITS(dirr[i]+'_sim_2.fits')[-1].read()
            d=fio.FITS(dirr[i]+'_sim_3.fits')[-1].read()
            e=fio.FITS(dirr[i]+'_sim_4.fits')[-1].read()

            g_values,g_errors,snr_binslist = residual_bias_correction(a,b,c,d,e, shape)
    return None

if __name__ == "__main__":
    main(sys.argv)