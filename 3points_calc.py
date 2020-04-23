import numpy as np
import math
import os
from matplotlib import pyplot as plt
import fitsio as fio
from scipy.optimize import curve_fit


def readinfiles(dirr):
    a=fio.FITS(dirr+'_sim_0.fits')[-1].read() 
    b=fio.FITS(dirr+'_sim_1.fits')[-1].read()
    c=fio.FITS(dirr+'_sim_2.fits')[-1].read()
    d=fio.FITS(dirr+'_sim_3.fits')[-1].read()
    e=fio.FITS(dirr+'_sim_4.fits')[-1].read()

    return a, b, c, d, e

def plot_3points(num, dirr1, dirr2, dirr3, dirr4, dirr5, dirr6):

    unsheared1, sheared1p1, sheared1m1, sheared2p1, sheared2m1 = readinfiles(dirr1)
    unsheared2, sheared1p2, sheared1m2, sheared2p2, sheared2m2 = readinfiles(dirr3)
    unsheared3, sheared1p3, sheared1m3, sheared2p3, sheared2m3 = readinfiles(dirr2)
    unsheared4, sheared1p4, sheared1m4, sheared2p4, sheared2m4 = readinfiles(dirr4)

    mask1 = (sheared1p1['g1'] == -0.02)
    g002=[-0.01, 0, 0.01]
    e002=[np.mean(sheared1m1['e1'][mask1]), np.mean(unsheared1['e1'][mask1]), np.mean(sheared1p1['e1'][mask1])]
    e002err=[np.std(sheared1m1['e1'][mask1])/np.sqrt(len(sheared1m1['e1'][mask1])), np.std(unsheared1['e1'][mask1])/np.sqrt(len(unsheared1['e1'][mask1])), np.std(sheared1p1['e1'][mask1])/np.sqrt(len(sheared1p1['e1'][mask1]))]

    mask2 = (sheared1p2['g1'] == -0.05)
    g005=[-0.01, 0, 0.01]
    e005=[np.mean(sheared1m2['e1'][mask2])-0.03, np.mean(unsheared2['e1'][mask2])-0.03, np.mean(sheared1p2['e1'][mask2])-0.03]
    e005err=[np.std(sheared1m2['e1'][mask2])/np.sqrt(len(sheared1m2['e1'][mask2])), np.std(unsheared2['e1'][mask2])/np.sqrt(len(unsheared2['e1'][mask2])), np.std(sheared1p2['e1'][mask2])/np.sqrt(len(sheared1p2['e1'][mask2]))]

    mask3 = (sheared2p3['g2'] == 0.02)
    g2002=[-0.01, 0, 0.01]
    e2002=[np.mean(sheared2m3['e2'][mask3]), np.mean(unsheared3['e2'][mask3]), np.mean(sheared2p3['e2'][mask3])]
    e2002err=[np.std(sheared2m3['e2'][mask3])/np.sqrt(len(sheared2m3['e2'][mask3])), np.std(unsheared3['e2'][mask3])/np.sqrt(len(unsheared3['e2'][mask3])), np.std(sheared2p3['e2'][mask3])/np.sqrt(len(sheared2p3['e2'][mask3]))]

    mask4 = (sheared2p4['g2'] == 0.05)
    g2005=[-0.01, 0, 0.01]
    e2005=[np.mean(sheared2m4['e2'][mask4])-0.03, np.mean(unsheared4['e2'][mask4])-0.03, np.mean(sheared2p4['e2'][mask4])-0.03]
    e2005err=[np.std(sheared2m4['e2'][mask4])/np.sqrt(len(sheared2m4['e2'][mask4])), np.std(unsheared4['e2'][mask4])/np.sqrt(len(unsheared4['e2'][mask4])), np.std(sheared2p4['e2'][mask4])/np.sqrt(len(sheared2p4['e2'][mask4]))]



    deltae1 = [e005[i]-e002[i] for i in range(3)]
    deltae2 = [e2005[i]-e2002[i] for i in range(3)]
    deltae1_err = [e002err[i]-e005err[i] for i in range(3)]
    deltae2_err = [e2002err[i]-e2005err[i] for i in range(3)]

    #print(deltae1, deltae2)
    #print(deltae1_err, deltae2_err)

    def func(x,m,n):
        return (1+m)*x+n

    def quadratic_function(x,a,b,c):    
        B = (a*(x**2.0)) + (b*x) + c
        return B

    # line and quadratic fit for e1, +-0.02
    params = curve_fit(func,g002,e002,p0=(0.,0.))
    m1,n1=params[0]
    params2 = curve_fit(quadratic_function,g002,e002,p0=(-1.,1.,0.), sigma=e002err)
    a1,b1,c1=params2[0]

    print(params[1])

    # same for e1, +-0.05
    params = curve_fit(func,g005,e005,p0=(0.,0.), sigma=e005err)
    m2,n2=params[0]
    params2 = curve_fit(quadratic_function,g005,e005,p0=(-1.,1.,0.), sigma=e005err)
    a2,b2,c2=params2[0]

    
    x = np.linspace(-0.01, 0.01, 100)
    g_x = np.array([-0.01,0,0.01])
    
    linefit1 = func(g_x,m1,n1)
    quadfit1 =  quadratic_function(g_x,a1,b1,c1)
    linefit2 = func(g_x,m2,n2)
    quadfit2 = quadratic_function(g_x,a2,b2,c2)

    print(e002, linefit1, e002err, np.std(sheared1m1['e1'][mask1]), np.std(unsheared1['e1'][mask1]), np.std(sheared1p1['e1'][mask1]))

    chilin1=0
    chiquad1=0
    chilin2=0
    chiquad2=0
    for j in range(3):
        chilin1 += ((e002[j] - linefit1[j])**2)/(e002err[j]**2)
        chiquad1 += ((e002[j] - quadfit1[j])**2)/(e002err[j]**2)
        chilin2 += ((e005[j] - linefit2[j])**2)/(e005err[j]**2)
        chiquad2 += ((e005[j] - quadfit2[j])**2)/(e005err[j]**2)
    print(chilin1, chilin2, chiquad1, chiquad2)
    #print(chisquare(e002, f_exp=linefit1), chisquare(e002, f_exp=quadfit1))
    #print(chisquare(e005, f_exp=linefit2), chisquare(e005, f_exp=quadfit2))
    
    """
    fig, ax1 = plt.subplots(figsize=(8,6))
    ax1.scatter(g002, e002, label='g1=+-0.02')
    ax1.errorbar(g002, e002, yerr=e002err, fmt='o')
    ax1.plot(x, func(x, m1, n1))
    ax1.plot(x, quadratic_function(x, a1, b1, c1))

    ax1.scatter(g005, e005, label='g1=+-0.05')
    ax1.errorbar(g005, e005, yerr=e005err, fmt='o')
    ax1.plot(x, func(x, m2, n2))
    ax1.plot(x, quadratic_function(x, a2, b2, c2))
    #ax1.scatter(g2002, deltae2, label='g2')
    #ax1.errorbar(g2002, deltae2, yerr=deltae2_err, fmt='o')
    #ax1.hlines(y=deltae1[1], xmin=-0.01, xmax=0.01, linestyles='dashed')
    #ax1.hlines(y=deltae2[1], xmin=-0.01, xmax=0.01, linestyles='solid')
    
    ax1.plot(g002, e002, marker='o', c='b', label='g1=+0.02')
    ax1.errorbar(g002, e002, yerr=e002err, c='b', fmt='o')

    ax1.plot(g005, e005, marker='o', c='g', label='g1=+0.05')
    ax1.errorbar(g005, e005, yerr=e005err, c='g', fmt='o')

    ax1.plot(g2002, e2002, marker='o',c='r', label='g2=+0.02')
    ax1.errorbar(g2002, e2002, yerr=e2002err, c='r', fmt='o')

    ax1.plot(g2005, e2005, marker='o', c='m', label='g2=+0.05')
    ax1.errorbar(g2005, e2005, yerr=e2005err, c='m', fmt='o')
    

    ax1.set_xlabel('g', fontsize=16)
    ax1.set_ylabel('e', fontsize=16)
    #ax1.set_ylim(-0.0012,-0.0010)
    ax1.legend(fontsize=11)
    ax1.tick_params(labelsize=10)
    plt.savefig('metacal_3points_fit.png')
    """

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

def plot_biasvsg(dir1, dir2, dir3, dir4, dir5, dir6):
    num = 5000000
    a,b,c,d,e = readinfiles(dirr1)
    values1, errors1 = residual_bias_correction(a,b,c,d,e,num)
    a,b,c,d,e = readinfiles(dirr2)
    values2, errors2 = residual_bias_correction(a,b,c,d,e,num)
    a,b,c,d,e = readinfiles(dirr3)
    values3, errors3 = residual_bias_correction(a,b,c,d,e,num)
    a,b,c,d,e = readinfiles(dirr4)
    values4, errors4 = residual_bias_correction(a,b,c,d,e,num)
    a,b,c,d,e = readinfiles(dirr5)
    values5, errors5 = residual_bias_correction(a,b,c,d,e,num)
    a,b,c,d,e = readinfiles(dirr6)
    values6, errors6 = residual_bias_correction(a,b,c,d,e,num)

    app_shear = [0.02, 0.05, 0.1]
    m1bias = [np.mean(values1[0]), np.mean(values3[0]), np.mean(values5[0])]
    m1biaserr = [np.mean(errors1[0]), np.mean(errors3[0]), np.mean(errors5[0])]
    m2bias = [np.mean(values2[2]), np.mean(values4[2]), np.mean(values6[2])]
    m2biaserr = [np.mean(errors2[2]), np.mean(errors4[2]), np.mean(errors6[2])]

    fig, ax1 = plt.subplots(figsize=(8,6))
    ax1.scatter(app_shear, m1bias, label='m1')
    ax1.errorbar(app_shear, m1bias, yerr=m1biaserr, fmt='o')
    ax1.scatter(app_shear, m2bias, label='m2')
    ax1.errorbar(app_shear, m2bias, yerr=m2biaserr, fmt='o')
    ax1.set_xlabel('Applied Shear, g', fontsize=15)
    ax1.set_ylabel('Multicative Bias, m', fontsize=15)
    plt.legend(fontsize=14)
    ax1.tick_params(labelsize=10)
    plt.savefig('metacal_bias_shear.png')


dirr1='v1_6' # g1=+-0.02
dirr2='v1_7' # g1=+-0.05
dirr3='v1_10'
dirr4='v1_11'
dirr5='v1_13'
dirr6='v1_14'
#plot_3points(num, dirr1, dirr2, dirr3, dirr4)
plot_biasvsg(dirr1, dirr2, dirr3, dirr4, dirr5, dirr6)