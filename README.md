# umbr_tools
## Author: Reilly Osadchey Brown
## Created: June 7, 2023
For running and reweighting umbrella sampling simulations

This package is meant to be used for umbrella sampling simulations using PLUMED to run the simualtions and PyMBAR to reweight them. It can be used for umbrella sampling which uses harmonic restraints, a static external biasing potential via PLUMED EXTERNAL, or a combintaion of them both. This is implemented for both 1D and 2D umbrella sampling simulations. 

I have defined classes to hold data and make reweighting these simulations easy. I define an umbrella set as any simulations which may differ in their harmonic restraints but share a common static bias potential. I define an umbrella collection to be a group of umbrella sets which will all be reweighted together. Each set may have different harmonic restraints and differnt extenal biases from the other sets in the collection.

IMPORTANT: Harmoniuc restraints are of the form y = 0.*5k*(x-a) where k is the force constant you supply. PLUMED harmonic restraints also include this factor, but be aware in case you use this package for a differnt program.

In the case where you have no external bias, you can still reweight using an Umbrella Set because the images all share a constant (zero) external bias.
In the case where you have no harmonic biases and just varied external biases: each external bias gets its own set and each set can be given an arbitrary harmonic restraint location with force constant zero. These harmonic restraints will not add any bias.
If you have just a single external bias and no harmonic biases, you can use a single umbrella set in a single umbrella collection to reweight. The MBAR will work, but this is overkill becasue you could do normal reweighting since you only have one window. 