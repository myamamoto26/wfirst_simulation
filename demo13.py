import numpy
import sys, os
import math
import logging
import time
import galsim as galsim
import galsim.roman as wfirst

# This is a setting that decides whether or not to output differences images showing what each
# detector effect does.  Since they take up quite a bit of space, we set this to False by default,
# but users who want to see the difference images can change it to True.
diff_mode = False

def main(argv):
    # Where to find and output data.
    path, filename = os.path.split(__file__)
    outpath = os.path.abspath(os.path.join(path, "output/"))

    # Just use a few galaxies, to save time.  Note that we are going to put 4000 galaxy images into
    # our big image, so if we have n_use=10, each galaxy will appear 400 times.  Users who want a
    # more interesting image with greater variation in the galaxy population can change `n_use` to
    # something larger (but it should be <=100, the number of galaxies in this small example
    # catalog).  With 4000 galaxies in a 4k x 4k image with the WFIRST pixel scale, the effective
    # galaxy number density is 74/arcmin^2.  This is not the number density that is expected for a
    # sample that is so bright (I<23.5) but it makes the image more visually interesting.  One could
    # think of it as what you'd get if you added up several images at once, making the images for a
    # sample that is much deeper have the same S/N as that for an I<23.5 sample in a single image.
    n_use = 10
    n_tot = 4000

    # Default is to use all filters.  Specify e.g. 'YJH' to only do Y106, J129, and H158.
    use_filters = None

    # quick and dirty command line parsing.
    for var in argv:
        if var.startswith('data='): datapath = var[5:]
        if var.startswith('out='): outpath = var[4:]
        if var.startswith('nuse='): n_use = int(var[5:])
        if var.startswith('ntot='): n_tot = int(var[5:])
        if var.startswith('filters='): use_filters = var[8:].upper()

    # Make output directory if not already present.
    if not os.path.isdir(outpath):
        os.mkdir(outpath)

    # In non-script code, use getLogger(__name__) at module scope instead.
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo13")

    # Initialize (pseudo-)random number generator.
    random_seed = 123456
    rng = galsim.BaseDeviate(random_seed)

    # Generate a Poisson noise model.
    poisson_noise = galsim.PoissonNoise(rng)
    logger.info('Poisson noise model created.')

    # Read in the WFIRST filters, setting an AB zeropoint appropriate for this telescope given its
    # diameter and (since we didn't use any keyword arguments to modify this) using the typical
    # exposure time for WFIRST images.  By default, this routine truncates the parts of the
    # bandpasses that are near 0 at the edges, and thins them by the default amount.
    filters = wfirst.getBandpasses(AB_zeropoint=True)
    logger.debug('Read in WFIRST imaging filters.')

    # Here we carry out the initial steps that are necessary to get a fully chromatic PSF.  We use
    # the getPSF() routine in the WFIRST module, which knows all about the telescope parameters
    # (diameter, bandpasses, obscuration, etc.).  Note that we arbitrarily choose a single SCA
    # (Sensor Chip Assembly) rather than all of them, for faster calculations, and use a simple
    # representation of the struts for faster calculations.  To do a more exact calculation of the
    # chromaticity and pupil plane configuration, remove the `approximate_struts` and the `n_waves`
    # keyword from the call to getPSF():
    
    PA=[20,50]
    for i in range(2):
        use_SCA = 7 # This could be any number from 1...18
        logger.info('Doing expensive pre-computation of PSF.')
        t1 = time.time()
        logger.setLevel(logging.DEBUG)

        # We choose a particular (RA, dec) location on the sky for our observation.
        ra_targ = 90.*galsim.degrees
        dec_targ = -10.*galsim.degrees
        targ_pos = galsim.CelestialCoord(ra=ra_targ, dec=dec_targ)
        # Get the WCS for an observation at this position.  We are not supplying a date, so the routine
        # will assume it's the vernal equinox.  We are also not supplying a position angle for the
        # observatory, which means that it will just find the optimal one (the one that has the solar
        # panels pointed most directly towards the Sun given this targ_pos and date).  The output of
        # this routine is a dict of WCS objects, one for each SCA.  We then take the WCS for the SCA
        # that we are using.
        wcs_dict = wfirst.getWCS(world_pos=targ_pos, PA=PA[i]*galsim.radians, SCAs=use_SCA)
        wcs = wcs_dict[use_SCA]
        # We need to find the center position for this SCA.  We'll tell it to give us a CelestialCoord
        # corresponding to (X, Y) = (wfirst.n_pix/2, wfirst.n_pix/2).
        SCA_cent_pos = wcs.toWorld(galsim.PositionD(wfirst.n_pix/2, wfirst.n_pix/2))

        # Need to make a separate PSF for each filter.  We are, however, ignoring the
        # position-dependence of the PSF within each SCA, just using the PSF at the center of the SCA
        # (default kwargs).
        PSFs = {}
        for filter_name, filter_ in filters.items():
            logger.info('PSF pre-computation for SCA %d, filter %s.'%(use_SCA, filter_name))
            PSFs[filter_name] = roman.getPSF(use_SCA, filter_name, n_waves=10, wcs=wcs, pupil_bin=4)
        t2 = time.time()
        logger.info('Done PSF precomputation in %.1f seconds!'%(t2-t1))

        # Define the size of the postage stamp that we use for each individual galaxy within the larger
        # image, and for the PSF images.
        stamp_size = 256

        # We randomly distribute points in (X, Y) on the CCD.
        # If we had a real galaxy catalog with positions in terms of RA, dec we could use wcs.toImage()
        # to find where those objects should be in terms of (X, Y).
        pos_rng = galsim.UniformDeviate(random_seed)

        # Calculate the sky level for each filter, and draw the PSF and the galaxies through the
        # filters.
        for filter_name, filter_ in filters.items():
            if use_filters is not None and filter_name[0] not in use_filters:
                logger.info('Skipping filter {0}.'.format(filter_name))
                continue

            logger.info('Beginning work for {0}.'.format(filter_name))

            # Drawing PSF.  Note that the PSF object intrinsically has a flat SED, so if we convolve it
            # with a galaxy, it will properly take on the SED of the galaxy.  For the sake of this demo,
            # we will simply convolve with a 'star' that has a flat SED and unit flux in this band, so
            # that the PSF image will be normalized to unit flux. This does mean that the PSF image
            # being drawn here is not quite the right PSF for the galaxy.  Indeed, the PSF for the
            # galaxy effectively varies within it, since it differs for the bulge and the disk.  To make
            # a real image, one would have to choose SEDs for stars and convolve with a star that has a
            # reasonable SED, but we just draw with a flat SED for this demo.
            out_filename = os.path.join(outpath, 'demo13_PSF_'+str(i)+'.fits')
            # Generate a point source.
            point = galsim.DeltaFunction(flux=1.)
            # Use a flat SED here, but could use something else.  A stellar SED for instance.
            # Or a typical galaxy SED.  Depending on your purpose for drawing the PSF.
            star_sed = galsim.SED(lambda x:1, 'nm', 'flambda').withFlux(1.,filter_)  # Give it unit flux in this filter.
            star = galsim.Convolve(point*star_sed, PSFs[filter_name])
            img_psf = galsim.ImageF(64,64)
            star.drawImage(bandpass=filter_, image=img_psf, scale=wfirst.pixel_scale)
            img_psf.write(out_filename)
            logger.debug('Created PSF with flat SED for {0}-band'.format(filter_name))

            # Set up the full image that will contain all the individual galaxy images, with information
            # about WCS:
            final_image = galsim.ImageF(wfirst.n_pix,wfirst.n_pix, wcs=wcs)


if __name__ == "__main__":
    main(sys.argv)