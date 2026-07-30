Some things to do by Mrugank / Nathan:
- Test and optimize on MAMO and other subjects, make sure the CPU and GPU fits are comparable if not better relative to MrVista
- Figure out an optimal grid size, maybe adaptive grid fitting (based on MruALFF or some other measure of SNR, does that help?)
- Can we do something like latent analyses to find common space voxels to do a grid fit few number of times for these and simply do final fit after?
- Can we do adaptive fitting for runs and figure out "bad" runs and use only good data to get better fits?
- Do fine resolution for the exponent maybe?
- Add an option to do differential sigma for minor and major axes?

- Aside:
- Are there anatomical and functional measures (noise correlations, functional connectivity estimates, measures of SNR like ALFF), that can give an estimate of prf fitting R^2 and maybe memory performance down the line?
- can we design an optimal task, and predict number of runs for a given subject that can give us reasonable fits depending on the measures above?


To do for Zhengang:
- Can we do an exhaustive list of tasks that other labs use, can we create stimuli sequences for each of them (the most common use cases)? Some way to visualize for "non-traditional" tasks so users can visualize the stimuli seq reliably?
- 