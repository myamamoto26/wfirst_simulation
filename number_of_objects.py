import os, sys
import numpy as np
from matplotlib import pyplot as plt

import astropy
from astropy.utils.data import download_file
from astropy.io import fits
from astropy.table import Table, vstack
import fitsio as fio

def main(argv):
    old = None
    dirr=sys.argv[1] # example: /hpc/group/cosmology/phy-lsst/my137/ngmix
    model=sys.argv[2] # example: mcal, mcal_coadd
    filter_=sys.argv[3]
    if sys.argv[4] == 'drizzle':
        medsn = np.arange(0,500)
    else:
        f = open('/hpc/group/cosmology/masaya/roman_imsim/meds_number.txt', 'r')
        medsn = f.read().split('\n')

    start = np.zeros(5)
    outlier = np.zeros(5)
    for j,pix in enumerate(medsn):
        for i in range(1):
            if not os.path.exists(dirr+'/fiducial_'+filter_+'_'+str(pix)+'_'+str(i)+'_'+model+'_noshear.fits'):
                continue
            new_ = fio.FITS(dirr+'/fiducial_'+filter_+'_'+str(pix)+'_'+str(i)+'_'+model+'_noshear.fits')[-1].read()
            new1p_ = fio.FITS(dirr+'/fiducial_'+filter_+'_'+str(pix)+'_'+str(i)+'_'+model+'_1p.fits')[-1].read()
            new1m_ = fio.FITS(dirr+'/fiducial_'+filter_+'_'+str(pix)+'_'+str(i)+'_'+model+'_1m.fits')[-1].read()
            new2p_ = fio.FITS(dirr+'/fiducial_'+filter_+'_'+str(pix)+'_'+str(i)+'_'+model+'_2p.fits')[-1].read()
            new2m_ = fio.FITS(dirr+'/fiducial_'+filter_+'_'+str(pix)+'_'+str(i)+'_'+model+'_2m.fits')[-1].read()

            lengths = [len(new_), len(new1p_), len(new1m_), len(new2p_), len(new2m_)]
            start = [start[k]+lengths[k] for k in range(5)]

            mask = (new_['flags']!=0)
            new_ = new_[mask]
            new1p_ = new1p_[mask]
            new1m_ = new1m_[mask]
            new2p_ = new2p_[mask]
            new2m_ = new2m_[mask]
            lengths2 = [len(new_), len(new1p_), len(new1m_), len(new2p_), len(new2m_)]
            outlier = [outlier[k]+lengths2[k] for k in range(5)]

            
    print('the number of objects is, ', start)
    print('the outliers are, ', outlier)

if __name__ == "__main__":
    main(sys.argv)