import os, sys
import numpy as np
from matplotlib import pyplot as plt

import astropy
from astropy.utils.data import download_file
from astropy.io import fits
from astropy.table import Table, vstack
import fitsio as fio

def main(argv):
    g = 0.01
    old = None
    dirr=sys.argv[1] # example: /hpc/group/cosmology/phy-lsst/my137/ngmix
    model=sys.argv[2] # example: mcal, mcal_coadd
    filter_=sys.argv[3]
    if sys.argv[4] == 'drizzle':
        medsn = np.arange(0,501)
    else:
        f = open('/hpc/group/cosmology/masaya/roman_imsim/meds_number.txt', 'r')
        medsn = f.read().split('\n')

    obj_num = int(sys.argv[4])
    start = 0
    for j,pix in enumerate(medsn):
        for i in range(1): #range(5):
            if not os.path.exists(dirr+'/fiducial_'+filter_+'_'+str(pix)+'_'+str(i)+'_'+model+'_noshear.fits'):
                continue
            new_ = fio.FITS(dirr+'/fiducial_'+filter_+'_'+str(pix)+'_'+str(i)+'_'+model+'_noshear.fits')[-1].read()
            new1p_ = fio.FITS(dirr+'/fiducial_'+filter_+'_'+str(pix)+'_'+str(i)+'_'+model+'_1p.fits')[-1].read()
            new1m_ = fio.FITS(dirr+'/fiducial_'+filter_+'_'+str(pix)+'_'+str(i)+'_'+model+'_1m.fits')[-1].read()
            new2p_ = fio.FITS(dirr+'/fiducial_'+filter_+'_'+str(pix)+'_'+str(i)+'_'+model+'_2p.fits')[-1].read()
            new2m_ = fio.FITS(dirr+'/fiducial_'+filter_+'_'+str(pix)+'_'+str(i)+'_'+model+'_2m.fits')[-1].read()

            #mask = (new_['flags']==0) #(new_['ind']!=0)
            #new_ = new_[mask]
            #new1p_ = new1p_[mask]
            #new1m_ = new1m_[mask]
            #new2p_ = new2p_[mask]
            #new2m_ = new2m_[mask]
            print(j,i,len(new_),len(new1p_),len(new1m_),len(new2p_),len(new2m_),start)
            if (j==0)&(i==0):
                new   = np.zeros(obj_num,dtype=new_.dtype)
                new1p = np.zeros(obj_num,dtype=new_.dtype)
                new1m = np.zeros(obj_num,dtype=new_.dtype)
                new2p = np.zeros(obj_num,dtype=new_.dtype)
                new2m = np.zeros(obj_num,dtype=new_.dtype)

            for col in new.dtype.names:
                new[col][start:start+len(new_)] += new_[col]
                new1p[col][start:start+len(new_)] += new1p_[col]
                new1m[col][start:start+len(new_)] += new1m_[col]
                new2p[col][start:start+len(new_)] += new2p_[col]
                new2m[col][start:start+len(new_)] += new2m_[col]
            start+=len(new_)
    print('number of objects is, ', start)

    fio.write(dirr+'/fiducial_'+filter_+'_mcal_noshear.fits', new)
    fio.write(dirr+'/fiducial_'+filter_+'_mcal_1p.fits', new1p)
    fio.write(dirr+'/fiducial_'+filter_+'_mcal_1m.fits', new1m)
    fio.write(dirr+'/fiducial_'+filter_+'_mcal_2p.fits', new2p)
    fio.write(dirr+'/fiducial_'+filter_+'_mcal_2m.fits', new2m)

if __name__ == "__main__":
    main(sys.argv)
