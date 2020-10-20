import numpy
import sys, os
import math
import logging
import time
import galsim as galsim
import galsim.wfirst as wfirst

stamp_size1 = 4088
stamp_size2 = 32
SCA = 1
filter_ = 'H158'
bpass = wfirst.getBandpasses(AB_zeropoint=True)[filter_]
psf = wfirst.getPSF(SCA, 
                    filter_,
                    pupil_bin=4,
                    n_waves=10,
                    wavelength=bpass.effective_wavelength
                    )

point = galsim.DeltaFunction(flux=1.)
star_sed = galsim.SED(lambda x:1, 'nm', 'flambda').withFlux(1.,filter_)  # Give it unit flux in this filter.
star1 = galsim.Convolve(point*star_sed, psf)
star2 = galsim.Convolve(point*star_sed, psf)
img_psf1 = galsim.ImageF(stamp_size1,stamp_size1)
img_psf2 = galsim.ImageF(stamp_size2, stamp_size2)
star1.drawImage(bandpass=filter_, image=img_psf1, scale=wfirst.pixel_scale)
star2.drawImage(bandpass=filter_, image=img_psf2, scale=wfirst.pixel_scale)
img_psf1.write('psf_size4088.fits')
img_psf1.write('psf_size32.fits')

if __name__ == "__main__":
    main(sys.argv)