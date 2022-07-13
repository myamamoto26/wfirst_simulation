import glob
import numpy as np
import fitsio as fio

mcal_keys = ['noshear', '1p', '1m', '2p', '2m']
for k in mcal_keys:
    fs = glob.glob('/hpc/group/cosmology/phy-lsst/my137/roman_H158/g1002/ngmix/new_coadd_no_oversampling_psf/fiducial_H158_*_0_mcal_coadd_'+k+'.fits')
    flag_fail = 0
    total = 0
    for f in fs:
        d = fio.read(f)
        total += len(d)
        mask = (d['flags']==0 & d['ind']!=0)
        nonzero_flag = d[mask]
        flag_fail += len(nonzero_flag)
    print(flag_fail, total)