# umbr_tools
## Author: Reilly Osadchey Brown
## Created: June 7, 2023
For running and reweighting umbrella sampling simulations

This package is meant to be used for umbrella sampling simulations using PLUMED to run the simualtions and PyMBAR to reweight them. It can be used for umbrella sampling which uses harmonic restraints, a static external biasing potential via PLUMED EXTERNAL, or a combination of them both. This is implemented for both 1D and 2D umbrella sampling simulations. 

I have defined classes to hold data and make reweighting these simulations easy. I define an umbrella set as any simulations which may differ in their harmonic restraints but share a common static bias potential. I define an umbrella collection to be a group of umbrella sets which will all be reweighted together. Each set may have different harmonic restraints and differnt extenal biases from the other sets in the collection.

IMPORTANT: Harmonic restraints are of the form y = 0.*5k*(x-a) where k is the force constant you supply. PLUMED harmonic restraints also include this factor, but be aware in case you use this package for a different program.

In the case where you have no external bias, you can still reweight using an Umbrella Set because the images all share a constant (zero) external bias.
In the case where you have no harmonic biases and just varied external biases: each external bias gets its own set and each set can be given an arbitrary harmonic restraint location with force constant zero. These harmonic restraints will not add any bias.
If you have just a single external bias and no harmonic biases, you can use a single umbrella set in a single umbrella collection to reweight. The MBAR will work, but this is overkill becasue you could do normal reweighting since you only have one window. 

Some potential issues:
1. If the gradient during MBAR initialization is nan, so will all the f_k free energies for each state. This will crash the FES histogramming downstream. This happens if the data has EVEN A SINGLE nan value. This is why the read_colvar has functions has the nan_handling option: "exception" raises an exception if nan are found (default), "drop" drops rows with any nan values. However, you should probably go back and look at your data manually. If this functions options are insufficient, please load the data into a Pandas dataframe manually and use that for MBAR once you have rectified the issue. One reason this occured for me: a simulation with PLUMED ended due to the job's allotted time being reached and being killed by the job scheduler. This left a half written line. In cases where the simualtion has not been restarted, this should be easy ti see at the end of the file. If the simualtion has been restarted, the PLUMED column name line will continue on this incomplete data line, making it harder to see.

Future work:
1. Seperate histogramming from the reweighting states & generalize the biasing and histogramming inputs. This will more easily allow 1d biases to be projected onto two dimensions, or 2d bias projected onto one dimensions.