import glob
import numpy as np
import fitsio as fio

fs = glob.glob('/hpc/group/cosmology/phy-lsst/my137/roman_H158/g1002/ngmix/new_coadd_no_oversampling_psf/fiducial_H158_*_0_mcal_coadd_noshear.fits')
flag_fail = 0
total = 0
for f in fs:
    d = fio.read(f)
    total += len(d)
    nonzero_flag = (d['coadd_e1']==0)
    flag_fail += len(nonzero_flag)
print(flag_fail, total)