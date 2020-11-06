import numpy as np
import fitsio as fio

a=fio.FITS('/hpc/group/cosmology/phy-lsst/my137/g1002/ngmix/fiducial_H158_mcal_noshear.fits')[-1].read()
b=fio.FITS('/hpc/group/cosmology/phy-lsst/my137/g1n002/ngmix/fiducial_H158_mcal_noshear.fits')[-1].read()

print(len(a['ind'], len(b['ind']))
for i in range(len(b['ind'])):
	if b['ind'][i] not in a['ind']:
		print(b['ind'][i])
	else:
		print('clear',i)