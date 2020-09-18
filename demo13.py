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

    logger.info('Reading from a parametric COSMOS catalog.')
    # Read in a galaxy catalog - just a random subsample of 100 galaxies for F814W<23.5 from COSMOS.
    cat_file_name = 'Simulated_WFIRST+LSST_photometry_catalog_CANDELSbased.fits'
    dir = 'data'
    # Use the routine that can take COSMOS real or parametric galaxy information, and tell it we
    # want parametric galaxies that represent an I<23.5 sample.
    cat = galsim.COSMOSCatalog(cat_file_name, dir=dir, use_real=False)
    logger.info('Read in %d galaxies from catalog'%cat.nobjects)

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
        # Make a list of (X, Y, F814W magnitude, n_rot, flip) values.
        # (X, Y) give the position of the galaxy centroid (or the center of the postage stamp into which
        # we draw the galaxy) in the big image.
        # F814W magnitudes are randomly drawn from the catalog, and are used to create a more realistic
        # flux distribution for the galaxies instead of just having the 10 flux values for the galaxies
        # we have chosen to draw.
        # n_rot says how many 90 degree rotations to include for a given realization of each galaxy, so
        # it doesn't appear completely identical each time we put it in the image.
        # flip is a random number that will determine whether we include an x-y flip for this appearance
        # of the galaxy or not.
        x_stamp = []
        y_stamp = []
        mag_stamp = []
        n_rot_stamp = []
        flip_stamp = []
        for i_gal in range(n_tot):
            x_stamp.append(pos_rng()*wfirst.n_pix)
            y_stamp.append(pos_rng()*wfirst.n_pix)
            # Note that we could use wcs.toWorld() to get the (RA, dec) for these (x, y) positions.  Or,
            # if we had started with (RA, dec) positions, we could have used wcs.toImage() to get the
            # CCD coordinates for those positions.
            mag_stamp.append(cat.param_cat['mag_auto'][int(pos_rng()*cat.nobjects)])
            n_rot_stamp.append(int(4*pos_rng()))
            flip_stamp.append(pos_rng())

        # Make the 2-component parametric GSObjects for each object, including chromaticity (roughly
        # appropriate SEDs per galaxy component, at the appropriate galaxy redshift).  Note that since
        # the PSF is position-independent within the SCA, we can simply do the convolution with that PSF
        # now instead of using a different one for each position.  We also have to include the correct
        # flux scaling: The catalog returns objects that would be observed by HST in 1 second
        # exposures. So for our telescope we scale up by the relative area and exposure time.  Note that
        # what is important is the *effective* area after taking into account obscuration.
        logger.info('Processing the objects in the catalog to get GSObject representations')
        # Choose a random set of unique indices in the catalog (will be the same each time script is
        # run, due to use of the same random seed):
        rand_indices = []
        while len(rand_indices)<n_use:
            tmp_ind = int(pos_rng()*cat.nobjects)
            if tmp_ind not in rand_indices:
                rand_indices.append(tmp_ind)
        obj_list = cat.makeGalaxy(rand_indices, chromatic=True, gal_type='parametric')
        hst_eff_area = 2.4**2 * (1.-0.33**2)
        wfirst_eff_area = galsim.wfirst.diameter**2 * (1.-galsim.wfirst.obscuration**2)
        flux_scaling = (wfirst_eff_area/hst_eff_area) * wfirst.exptime
        mag_list = []
        for ind in range(len(obj_list)):
            # First, let's check what magnitude this object has in F814W.  We want to do this because
            # (to inject some variety into our images) we are going to rescale the fluxes in all bands
            # for different instances of this galaxy in the final image in order to get a reasonable S/N
            # distribution.  So we need to save the original magnitude in F814W, to compare with a
            # randomly drawn one from the catalog.  This is not something that most users would need to
            # do.
            mag_list.append(cat.param_cat['mag_auto'][cat.orig_index[rand_indices[ind]]])

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