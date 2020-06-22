# wfirst_simulation

1. For the main simulation, use wfirst_simple_sim.py (includes multiple parameters; half-light radius, galaxy number,
   input shear, galaxy model, PSF model, shear measurement methods, etc... )
   Main simulation returns the ngmix output or the metacalibration output (noshear, 1p, 1m, 2p, 2m) in fits format. 

2. To analyze the shear response and observed shear, use selection_effects.py, which prints out the before and after
   we take care of selection effects on shear response. 
   To use this, "python selection_effects.py 'shear measurement methods' [paths to the files]"
   shear measurement methods: 'metacal' or 'ngmix'. 
   An example of the path name, [v2_7_offset_45]. This path will take care of 5 files from the metacal output. 

3. To make a plot of the difference of observed shear (delta g_obs), use plot_obsg_inputg. This can make a plot of 
   delta g vs (input g or angle offset).
   To use this, "python plot_obsg_inputg.py 'shear measurement methods'"