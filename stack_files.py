

import numpy as np
import math
import os
from matplotlib import pyplot as plt
import fitsio as fio
from scipy.optimize import curve_fit
from astropy.table import vstack


def readinfiles(dirr):
    a=fio.FITS(dirr+'_sim_0.fits')[-1].read() 
    b=fio.FITS(dirr+'_sim_1.fits')[-1].read()
    c=fio.FITS(dirr+'_sim_2.fits')[-1].read()
    d=fio.FITS(dirr+'_sim_3.fits')[-1].read()
    e=fio.FITS(dirr+'_sim_4.fits')[-1].read()

    return a, b, c, d, e

def stack_files(f1, f2, f3, f4, f5, f6):
    a1, b1, c1, d1, e1 = readinfiles(f1)
    a2, b2, c2, d2, e2 = readinfiles(f2)
    a3, b3, c3, d3, e3 = readinfiles(f3)
    a4, b4, c4, d4, e4 = readinfiles(f4)
    a5, b5, c5, d5, e5 = readinfiles(f5)
    a6, b6, c6, d6, e6 = readinfiles(f6)

    noshear = np.append(a1,a2,a3,a4,a5,a6)
    shear1p = np.append(b1,b2,b3,b4,b5,b6)
    shear1m = np.append(c1,c2,c3,c4,c5,c6)
    shear2p = np.append(d1,d2,d3,d4,d5,d6)
    shear2m = np.append(e1,e2,e3,e4,e5,e6)

    return noshear, shear1p, shear1m, shear2p, shear2m

def residual_bias(res_tot, gal_num):
    g = 0.01

    new = res_tot[0]
    new1p = res_tot[1]
    new1m = res_tot[2]
    new2p = res_tot[3]
    new2m = res_tot[4]

    print(len(new1p["e1"]))
    #old = old[old['ra']!=0]
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

    g1 = new['e1']/avg_R11
    g2 = new['e2']/avg_R22

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

    #params2 = curve_fit(func,new['g1'],new['e1']/avg_R11,p0=(0.,0.))
    params2 = curve_fit(func,new['g1'],new['e1']/avg_R11,p0=(0.,0.))
    m5,b5=params2[0]
    m5err,b5err=np.sqrt(np.diagonal(params2[1]))
    #params2 = curve_fit(func,new['g2'],new['e2']/avg_R22,p0=(0.,0.))
    params2 = curve_fit(func,new['g2'],new['e2']/avg_R22,p0=(0.,0.))
    m6,b6=params2[0]
    m6err,b6err=np.sqrt(np.diagonal(params2[1]))

    print("before correction: ")
    print("m1="+str("%6.4f"% m5)+"+-"+str("%6.4f"% m5err), "b1="+str("%6.6f"% b5)+"+-"+str("%6.6f"% b5err))
    print("m2="+str("%6.4f"% m6)+"+-"+str("%6.4f"% m6err), "b2="+str("%6.6f"% b6)+"+-"+str("%6.6f"% b6err))

    return R11, R22, R12, R21

def residual_bias_correction(a, b, c, d, e, gal_num):
    g = 0.01
    new = a
    new1p = b
    new1m = c
    new2p = d
    new2m = e

    R11, R22, R12, R21 = residual_bias([a,b,c,d,e], gal_num)

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

        params = curve_fit(func,new['g1'][mask],new['e1'][mask]/tot_R11[p],p0=(0.,0.))
        m1,b1=params[0]
        m1err,b1err=np.sqrt(np.diagonal(params[1]))
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

    print(m1,b1)
    print(m1_val, b1_val)
    print('corrected m, b: ')
    print("m1="+str("%6.4f"% np.mean(m1_val))+"+-"+str("%6.4f"% np.mean(m1_err)), "b1="+str("%6.6f"% np.mean(b1_val))+"+-"+str("%6.6f"% np.mean(b1_err)))
    print("m2="+str("%6.4f"% np.mean(m2_val))+"+-"+str("%6.4f"% np.mean(m2_err)), "b2="+str("%6.6f"% np.mean(b2_val))+"+-"+str("%6.6f"% np.mean(b2_err)))

    values=[m1_val,b1_val,m2_val,b2_val,m3_val,b3_val,m4_val,b4_val]
    errors=[m1_err,b1_err,m2_err,b2_err,m3_err,b3_err,m4_err,b4_err]
    return values, errors


new, new1p, new1m, new2p, new2m = stack_files('v1_6', 'v1_7', 'v1_17', 'v1_18', 'v1_19', 'v1_20')
gal_num = 15000000
bias_means, bias_errors = residual_bias_correction(new, new1p, new1m, new2p, new2m, gal_num)










