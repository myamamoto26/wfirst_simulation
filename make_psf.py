import numpy
import sys, os
import math
import logging
import time
import galsim as galsim
import galsim.wfirst as wfirst


stamp_size = [4088, 32, 256]
SCA = 1
filter_ = 'H158'
bpass = wfirst.getBandpasses(AB_zeropoint=True)[filter_]
psf = wfirst.getPSF(SCA, 
                    filter_,
                    pupil_bin=1,
                    n_waves=10,
                    wavelength=bpass.effective_wavelength
                    )


for stamp in stamp_size:
    st_model = galsim.DeltaFunction(flux=1.)
    st_model = st_model.evaluateAtWavelength(bpass.effective_wavelength)
    # reassign correct flux
    starflux=1.
    st_model = st_model.withFlux(starflux)
    st_model = galsim.Convolve(st_model, psf)

    xyI = galsim.PositionI(int(stamp), int(stamp))
    b = galsim.BoundsI( xmin=1,
                        xmax=xyI.x,
                        ymin=1,
                        ymax=xyI.y)
    psf_stamp = galsim.Image(b, scale=wfirst.pixel_scale)
    st_model.drawImage(image=psf_stamp)
    psf_stamp.write('psf_size_pupil1_'+str(stamp)+'.fits')

