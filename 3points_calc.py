import numpy as np
import math
import os
from matplotlib import pyplot as plt
import fitsio as fio


def readinfiles(dirr):
    a=fio.FITS(dirr+'_sim_0.fits')[-1].read() 
    b=fio.FITS(dirr+'_sim_1.fits')[-1].read()
    c=fio.FITS(dirr+'_sim_2.fits')[-1].read()
    d=fio.FITS(dirr+'_sim_3.fits')[-1].read()
    e=fio.FITS(dirr+'_sim_4.fits')[-1].read()

    return a, b, c, d, e

def plot_3points(num, dirr1, dirr2, dirr3, dirr4):

    unsheared1, sheared1p1, sheared1m1, sheared2p1, sheared2m1 = readinfiles(dirr1)
    unsheared2, sheared1p2, sheared1m2, sheared2p2, sheared2m2 = readinfiles(dirr2)
    unsheared3, sheared1p3, sheared1m3, sheared2p3, sheared2m3 = readinfiles(dirr3)
    unsheared4, sheared1p4, sheared1m4, sheared2p4, sheared2m4 = readinfiles(dirr4)

    mask1 = (sheared1p1['g1'] == 0.02)
    g002=[-0.01, 0, 0.01]
    e002=[np.mean(sheared1m1['e1'][mask1]), np.mean(unsheared1['e1'][mask1]), np.mean(sheared1p1['e1'][mask1])]
    e002err=[np.std(sheared1m1['e1'][mask1])/np.sqrt(len(sheared1m1['e1'][mask1])), np.std(unsheared1['e1'][mask1])/np.sqrt(len(unsheared1['e1'][mask1])), np.std(sheared1p1['e1'][mask1])/np.sqrt(len(sheared1p1['e1'][mask1]))]

    mask2 = (sheared1p2['g1'] == 0.05)
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


    from scipy.optimize import curve_fit
    from scipy.stats import chisquare
    def func(x,m,n):
        return (1+m)*x+n

    def quadratic_function(x,a,b,c):    
        B = (a*(x**2.0)) + (b*x) + c
        return B

    # line and quadratic fit for e1, +-0.02
    params = curve_fit(func,g002,e002,p0=(0.,0.))
    m1,n1=params[0]
    params2 = curve_fit(quadratic_function,g002,e002,p0=(0.,0.,0.))
    a1,b1,c1=params2[0]

    # same for e1, +-0.05
    params = curve_fit(func,g005,e005,p0=(0.,0.))
    m2,n2=params[0]
    params2 = curve_fit(quadratic_function,g005,e005,p0=(0.,0.,0.))
    a2,b2,c2=params2[0]

    
    x = np.linspace(-0.01, 0.01, 100)
    g_x = np.array([-0.01,0,0.01])
    
    linefit1 = func(g_x,m1,n1)
    quadfit1 =  quadratic_function(g_x,a1,b1,c1)
    linefit2 = func(g_x,m2,n2)
    quadfit2 = quadratic_function(g_x,a2,b2,c2)

    chiline1=0
    chiquad1=0
    chilin2=0
    chiquad2=0
    for j in range(3):
        chilin1 += ((e002[j] - linefit1[j])**2)/linefit1[j]
        chiquad1 += ((e002[j] - quadfit1[j])**2)/quadfit1[j]
        chilin2 += ((e005[j] - linefit2[j])**2)/linefit2[j]
        chiquad2 += ((e005[j] - quadfit2[j])**2)/quadfit2[j]
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


num = 5000000
dirr1='v1_6' # g1=+-0.02
dirr2='v1_10' # g1=+-0.05
dirr3='v1_7'
dirr4='v1_11'
plot_3points(num, dirr1, dirr2, dirr3, dirr4)