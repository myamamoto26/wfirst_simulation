

import numpy as np
from matplotlib import pyplot as plt 
import math
import os
import h5py
import scipy

import astropy
from astropy.utils.data import download_file
from astropy.io import fits
from astropy.table import Table, vstack
import fitsio as fio
from scipy.optimize import curve_fit

'''
def h5read(filename):

    f=h5py.File(filename,mode='r')
    return f

fileset=['/net/oit-nas-fe13.dscr.duke.local/phy-lsst/DES-Y3-Sims/desy3_combined_mcal_cat_g1-0.02.h5',
        '/net/oit-nas-fe13.dscr.duke.local/phy-lsst/DES-Y3-Sims/desy3_combined_mcal_cat_g10.02.h5', 
        '/net/oit-nas-fe13.dscr.duke.local/phy-lsst/DES-Y3-Sims/desy3_combined_mcal_cat_g2-0.02.h5', 
        '/net/oit-nas-fe13.dscr.duke.local/phy-lsst/DES-Y3-Sims/desy3_combined_mcal_cat_g20.02.h5']

len1=[]
len2=[]
len3=[]
len4=[]
len5=[]
for j in range(4):
    len1 += [len(h5read(fileset[j])['/catalog/metacal/unsheared/R11'][:])]
    len2 += [len(h5read(fileset[j])['/catalog/metacal/sheared_1p/e_1'][:])]
    len3 += [len(h5read(fileset[j])['/catalog/metacal/sheared_1m/e_1'][:])]
    len4 += [len(h5read(fileset[j])['/catalog/metacal/sheared_2p/e_1'][:])]
    len5 += [len(h5read(fileset[j])['/catalog/metacal/sheared_2m/e_1'][:])]
    
print(len1, len2, len3, len4, len5)


fileset=['/net/oit-nas-fe13.dscr.duke.local/phy-lsst/DES-Y3-Sims/desy3_combined_mcal_cat_g1-0.02.h5',
        '/net/oit-nas-fe13.dscr.duke.local/phy-lsst/DES-Y3-Sims/desy3_combined_mcal_cat_g10.02.h5', 
        '/net/oit-nas-fe13.dscr.duke.local/phy-lsst/DES-Y3-Sims/desy3_combined_mcal_cat_g2-0.02.h5', 
        '/net/oit-nas-fe13.dscr.duke.local/phy-lsst/DES-Y3-Sims/desy3_combined_mcal_cat_g20.02.h5']

    
start = 0
for j in range(4):
    new_ = h5read(fileset[j])['/catalog/metacal/unsheared']
    new1p_ = h5read(fileset[j])['/catalog/metacal/sheared_1p']
    new1m_ = h5read(fileset[j])['/catalog/metacal/sheared_1m']
    new2p_ = h5read(fileset[j])['/catalog/metacal/sheared_2p']
    new2m_ = h5read(fileset[j])['/catalog/metacal/sheared_2m']
    if j==0:
        new   = np.zeros(len1,dtype=new_.dtype)
        new1p = np.zeros(len2,dtype=new1p_.dtype)
        new1m = np.zeros(len3,dtype=new1m_.dtype)
        new2p = np.zeros(len4,dtype=new2p_.dtype)
        new2m = np.zeros(len5,dtype=new2m_.dtype)
    else:
        for col in new.dtype.names:
            new[col][start:start+len(new_['R11'][:])] += new_[col]
        for col in new1p.dtype.names:
            new1p[col][start:start+len(new_['R11'][:])] += new1p_[col]
            new1m[col][start:start+len(new_['R11'][:])] += new1m_[col]
            new2p[col][start:start+len(new_['R11'][:])] += new2p_[col]
            new2m[col][start:start+len(new_['R11'][:])] += new2m_[col]
    start+=len(new_['R11'][:])
'''


def h5read(filename):

    f=h5py.File(filename,mode='r')
    return f

def find_length(fileset):
    len1=[]
    len2=[]
    len3=[]
    len4=[]
    len5=[]
    for j in range(4):
        len1 += [len(h5read(fileset[j])['/catalog/metacal/unsheared/R11'][:])]
        len2 += [len(h5read(fileset[j])['/catalog/metacal/sheared_1p/e_1'][:])]
        len3 += [len(h5read(fileset[j])['/catalog/metacal/sheared_1m/e_1'][:])]
        len4 += [len(h5read(fileset[j])['/catalog/metacal/sheared_2p/e_1'][:])]
        len5 += [len(h5read(fileset[j])['/catalog/metacal/sheared_2m/e_1'][:])]
    return len1, len2, len3, len4, len5

def combine_data(file1, file2, file3, file4):

    unsheared = [h5read(file1)['/catalog/metacal/unsheared'], h5read(file2)['/catalog/metacal/unsheared'], h5read(file3)['/catalog/metacal/unsheared'], h5read(file4)['/catalog/metacal/unsheared']]
    sheared_1p = [h5read(file1)['/catalog/metacal/sheared_1p'], h5read(file2)['/catalog/metacal/sheared_1p'], h5read(file3)['/catalog/metacal/sheared_1p'], h5read(file4)['/catalog/metacal/sheared_1p']]
    sheared_1m = [h5read(file1)['/catalog/metacal/sheared_1m'], h5read(file2)['/catalog/metacal/sheared_1m'], h5read(file3)['/catalog/metacal/sheared_1m'], h5read(file4)['/catalog/metacal/sheared_1m']]
    sheared_2p = [h5read(file1)['/catalog/metacal/sheared_2p'], h5read(file2)['/catalog/metacal/sheared_2p'], h5read(file3)['/catalog/metacal/sheared_2p'], h5read(file4)['/catalog/metacal/sheared_2p']]
    sheared_2m = [h5read(file1)['/catalog/metacal/sheared_2m'], h5read(file2)['/catalog/metacal/sheared_2m'], h5read(file3)['/catalog/metacal/sheared_2m'], h5read(file4)['/catalog/metacal/sheared_2m']]


    for i in range(4):
        if i==0:
            newe1 = unsheared[i]['e_1'][:]
            newe2 = unsheared[i]['e_2'][:]
            new1pe1 = sheared_1p[i]['e_1'][:]
            new1pe2 = sheared_1p[i]['e_2'][:]
            new1me1 = sheared_1m[i]['e_1'][:]
            new1me2 = sheared_1m[i]['e_2'][:]
            new2pe1 = sheared_2p[i]['e_1'][:]
            new2pe2 = sheared_2p[i]['e_2'][:]
            new2me1 = sheared_2m[i]['e_1'][:]
            new2me2 = sheared_2m[i]['e_2'][:]
            newsnr = unsheared[i]['snr'][:]
            newflags = unsheared[i]['flags'][:]
            newT = unsheared[i]['T'][:]
            newmcalT = unsheared[i]['mcal_psf_T'][:]
            new1psnr = sheared_1p[i]['snr'][:]
            new1msnr = sheared_1m[i]['snr'][:]
            new2psnr = sheared_2p[i]['snr'][:]
            new2msnr = sheared_2m[i]['snr'][:]
        else:
            newe1 = np.append(newe1, unsheared[i]['e_1'][:])
            newe2 = np.append(newe2, unsheared[i]['e_2'][:])
            new1pe1 = np.append(new1pe1, sheared_1p[i]['e_1'][:])
            new1pe2 = np.append(new1pe2, sheared_1p[i]['e_2'][:])
            new1me1 = np.append(new1me1, sheared_1m[i]['e_1'][:])
            new1me2 = np.append(new1me2, sheared_1m[i]['e_2'][:])
            new2pe1 = np.append(new2pe1, sheared_2p[i]['e_1'][:])
            new2pe2 = np.append(new2pe2, sheared_2p[i]['e_2'][:])
            new2me1 = np.append(new2me1, sheared_2m[i]['e_1'][:])
            new2me2 = np.append(new2me2, sheared_2m[i]['e_2'][:])
            newsnr = np.append(newsnr, unsheared[i]['snr'][:])
            newflags = np.append(newflags, unsheared[i]['flags'][:])
            newT = np.append(newT, unsheared[i]['T'][:])
            newmcalT = np.append(newmcalT, unsheared[i]['mcal_psf_T'][:])
            new1psnr = np.append(new1psnr, sheared_1p[i]['snr'][:])
            new1msnr = np.append(new1msnr, sheared_1m[i]['snr'][:])
            new2psnr = np.append(new2psnr, sheared_2p[i]['snr'][:])
            new2msnr = np.append(new2msnr, sheared_2m[i]['snr'][:])

    g11=-0.02*np.ones_like(unsheared[0]['e_1'][:])
    g12=0.02*np.ones_like(unsheared[1]['e_1'][:])
    g13=np.zeros_like(unsheared[2]['e_1'][:])
    g14=np.zeros_like(unsheared[3]['e_1'][:])
    g21=np.zeros_like(unsheared[0]['e_1'][:])
    g22=np.zeros_like(unsheared[1]['e_1'][:])
    g23=-0.02*np.ones_like(unsheared[2]['e_1'][:])
    g24=0.02*np.ones_like(unsheared[3]['e_1'][:])

    g1 = np.concatenate((g11,g12,g13,g14))
    g2 = np.concatenate((g21,g22,g23,g24))

    # make selections
    # flags==0, 0<snr<1000, T/mcal_psf_T>0.5
    mask =  (newflags == 0) & (newsnr > 0) & (newsnr < 1000) & (newT/newmcalT > 0.5) 

    newe1 = newe1[mask]
    newe2 = newe2[mask]
    new1pe1 = new1pe1[mask]
    new1pe2 = new1pe2[mask]
    new1me1 = new1me1[mask]
    new1me2 = new1me2[mask]
    new2pe1 = new2pe1[mask]
    new2pe2 = new2pe2[mask]
    new2me1 = new2me1[mask]
    new2me2 = new2me2[mask]
    newsnr = newsnr[mask]
    new1psnr = new1psnr[mask]
    new1msnr = new1msnr[mask]
    new2psnr = new2psnr[mask]
    new2msnr = new2msnr[mask]
    g1 = g1[mask]
    g2 = g2[mask]

    return newe1, newe2, new1pe1, new1pe2, new1me1, new1me2, new2pe1, new2pe2, new2me1, new2me2, newsnr, new1psnr, new1msnr, new2psnr, new2msnr, g1, g2


def residual_bias(newe1, newe2, new1pe1, new1pe2, new1me1, new1me2, new2pe1, new2pe2, new2me1, new2me2, g1, g2):
    g = 0.01

    R11 = (new1pe1 - new1me1)/(2*g)
    R22 = (new2pe2 - new2me2)/(2*g)
    R12 = (new2pe1 - new2me1)/(2*g)
    R21 = (new1pe2 - new1me2)/(2*g)

    avg_R11 = np.mean(R11)
    avg_R22 = np.mean(R22)
    avg_R12 = np.mean(R12)
    avg_R21 = np.mean(R21)

    #g1 = new['e1']/avg_R11
    #g2 = new['e2']/avg_R22

    ## some statistics
    print("Mean shear response: ")
    N=len(new1pe1)
    print(N)
    print("<R11> = "+str("%6.4f"% avg_R11)+"+-"+str("%6.4f"% (np.std(R11)/np.sqrt(N))))
    print("<R22> = "+str("%6.4f"% avg_R22)+"+-"+str("%6.4f"% (np.std(R22)/np.sqrt(N))))
    print("<R12> = "+str("%6.4f"% avg_R12)+"+-"+str("%6.4f"% (np.std(R12)/np.sqrt(N))))
    print("<R21> = "+str("%6.4f"% avg_R21)+"+-"+str("%6.4f"% (np.std(R21)/np.sqrt(N))))


    def func(x,m,b):
      return (1+m)*x+b
      
    #params2 = curve_fit(func,new['g1'],new['e1']/avg_R11,p0=(0.,0.))
    params2 = curve_fit(func,g1,newe1/avg_R11,p0=(0.,0.))
    m5,b5=params2[0]
    m5err,b5err=np.sqrt(np.diagonal(params2[1]))
    #params2 = curve_fit(func,new['g2'],new['e2']/avg_R22,p0=(0.,0.))
    params2 = curve_fit(func,g2,newe2/avg_R22,p0=(0.,0.))
    m6,b6=params2[0]
    m6err,b6err=np.sqrt(np.diagonal(params2[1]))

    print("before correction: ")
    print("m1="+str("%6.4f"% m5)+"+-"+str("%6.4f"% m5err), "b1="+str("%6.6f"% b5)+"+-"+str("%6.6f"% b5err))
    print("m2="+str("%6.4f"% m6)+"+-"+str("%6.4f"% m6err), "b2="+str("%6.6f"% b6)+"+-"+str("%6.6f"% b6err))

    return R11, R22, R12, R21

def residual_bias_correction(len1, len2, len3, len4, len5):

    newe1, newe2, new1pe1, new1pe2, new1me1, new1me2, new2pe1, new2pe2, new2me1, new2me2, newsnr, new1psnr, new1msnr, new2psnr, new2msnr, g1, g2 = combine_data('/net/oit-nas-fe13.dscr.duke.local/phy-lsst/DES-Y3-Sims/desy3_combined_mcal_cat_g1-0.02.h5',
                                                                                                                                                                '/net/oit-nas-fe13.dscr.duke.local/phy-lsst/DES-Y3-Sims/desy3_combined_mcal_cat_g10.02.h5', 
                                                                                                                                                                '/net/oit-nas-fe13.dscr.duke.local/phy-lsst/DES-Y3-Sims/desy3_combined_mcal_cat_g2-0.02.h5', 
                                                                                                                                                                '/net/oit-nas-fe13.dscr.duke.local/phy-lsst/DES-Y3-Sims/desy3_combined_mcal_cat_g20.02.h5')

    #test 
    if len(newe1)!=len(new2me2):
        print("ERROR!!")

    g = 0.01

    R11, R22, R12, R21 = residual_bias(newe1, newe2, new1pe1, new1pe2, new1me1, new1me2, new2pe1, new2pe2, new2me1, new2me2, g1, g2)

    avg_R11 = np.mean(R11)
    avg_R22 = np.mean(R22)

    snr_binn = 10
    snr_min = np.log(10) #np.min(new['hlr']) #np.log(15) #np.log(min(new['snr']))
    snr_max = np.log(1000) #np.max(new['hlr']) #np.log(max(new['snr']))
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
            if (np.log(newsnr[b]) >= snr_binslist[a]) and (np.log(newsnr[b]) < snr_binslist[a+1]):
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
        mask_1p = (np.log(new1psnr) >= snr_binslist[i]) & (np.log(new1psnr) < snr_binslist[i+1])
        mask_1m = (np.log(new1msnr) >= snr_binslist[i]) & (np.log(new1msnr) < snr_binslist[i+1])
        mask_2p = (np.log(new2psnr) >= snr_binslist[i]) & (np.log(new2psnr) < snr_binslist[i+1])
        mask_2m = (np.log(new2msnr) >= snr_binslist[i]) & (np.log(new2msnr) < snr_binslist[i+1])
        
        #mask_1p = (new1p['hlr'] >= snr_binslist[i]) & (new1p['hlr'] < snr_binslist[i+1])
        #mask_1m = (new1m['hlr'] >= snr_binslist[i]) & (new1m['hlr'] < snr_binslist[i+1])
        #mask_2p = (new2p['hlr'] >= snr_binslist[i]) & (new2p['hlr'] < snr_binslist[i+1])
        #mask_2m = (new2m['hlr'] >= snr_binslist[i]) & (new2m['hlr'] < snr_binslist[i+1])
            
        #print("how many objects fall in each bin. ", len(mask_1p), len(mask_1m), len(mask_2p), len(mask_2m))
        
        R11_s += [(np.mean(newe1[mask_1p]) - np.mean(newe1[mask_1m]))/(2*g)]
        R22_s += [(np.mean(newe2[mask_2p]) - np.mean(newe2[mask_2m]))/(2*g)]
        R12_s += [(np.mean(newe1[mask_2p]) - np.mean(newe1[mask_2m]))/(2*g)]
        R21_s += [(np.mean(newe2[mask_1p]) - np.mean(newe2[mask_1m]))/(2*g)]

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
        mask = (np.log(newsnr) >= snr_binslist[p]) & (np.log(newsnr) < snr_binslist[p+1])
        #mask = (new['hlr'] >= snr_binslist[p]) & (new['hlr'] < snr_binslist[p+1])

        params = curve_fit(func,g1[mask],newe1[mask]/tot_R11[p],p0=(0.,0.))
        m1,b1=params[0]
        m1err,b1err=np.sqrt(np.diagonal(params[1]))
        params = curve_fit(func,g2[mask],newe2[mask]/tot_R22[p],p0=(0.,0.))
        m2,b2=params[0]
        m2err,b2err=np.sqrt(np.diagonal(params[1]))
        
        params = curve_fit(func,g1[mask],newe1[mask]/R11_g[p],p0=(0.,0.))
        m3,b3=params[0]
        m3err,b3err=np.sqrt(np.diagonal(params[1]))
        params = curve_fit(func,g2[mask],newe2[mask]/R22_g[p],p0=(0.,0.))
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

    print(m1,b1)
    print(m1_val, b1_val)
    print('corrected m, b: ')
    print("m1="+str("%6.4f"% np.mean(m1_val))+"+-"+str("%6.4f"% np.mean(m1_err)), "b1="+str("%6.6f"% np.mean(b1_val))+"+-"+str("%6.6f"% np.mean(b1_err)))
    print("m2="+str("%6.4f"% np.mean(m2_val))+"+-"+str("%6.4f"% np.mean(m2_err)), "b2="+str("%6.6f"% np.mean(b2_val))+"+-"+str("%6.6f"% np.mean(b2_err)))

    values=[m1_val,b1_val,m2_val,b2_val,m3_val,b3_val,m4_val,b4_val]
    errors=[m1_err,b1_err,m2_err,b2_err,m3_err,b3_err,m4_err,b4_err]
    return values, errors, snr_binslist





fileset=['/net/oit-nas-fe13.dscr.duke.local/phy-lsst/DES-Y3-Sims/desy3_combined_mcal_cat_g1-0.02.h5',
        '/net/oit-nas-fe13.dscr.duke.local/phy-lsst/DES-Y3-Sims/desy3_combined_mcal_cat_g10.02.h5', 
        '/net/oit-nas-fe13.dscr.duke.local/phy-lsst/DES-Y3-Sims/desy3_combined_mcal_cat_g2-0.02.h5', 
        '/net/oit-nas-fe13.dscr.duke.local/phy-lsst/DES-Y3-Sims/desy3_combined_mcal_cat_g20.02.h5']
len1, len2, len3, len4, len5 = find_length(fileset)
residual_bias_correction(len1,len2,len3,len4,len5)
