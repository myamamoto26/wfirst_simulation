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

def plot_3points(num, dirr1, dirr2):

    unsheared1, sheared1p1, sheared1m1, sheared2p1, sheared2m1 = readinfiles(dirr1)
    unsheared2, sheared1p2, sheared1m2, sheared2p2, sheared2m2 = readinfiles(dirr2)

    mask1 = (sheared1p1['g1'] == 0.02)
    g002=[-0.01, 0, 0.01]
    e002=[np.mean(sheared1m1['e1'][mask1]), np.mean(unsheared1['e1'][mask1]), np.mean(sheared1p1['e1'][mask1])]

    mask2 = (sheared1p2['g1'] == 0.05)
    g005=[-0.01, 0, 0.01]
    e005=[np.mean(sheared1m2['e1'][mask2]), np.mean(unsheared2['e1'][mask2]), np.mean(sheared1p2['e1'][mask2])]

    print(e002, e005)

    fig, ax1 = plt.subplots(figsize=(8,6))
    ax1.plot(g002, e002)
    ax1.plot(g005, e005)
    plt.savefig('metacal_3points.png')


num = 5000000
dirr1='v1_6' # g1=+-0.02
dirr2='v1_10' # g1=+-0.05
plot_3points(num, dirr1, dirr2)