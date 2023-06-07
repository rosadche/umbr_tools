# umbr_tools
## Author: Reilly Osadchey Brown
## Created: June 7, 2023
For running and reweighting umbrella sampling simulations

This package is meant to be used for umbrella sampling simulations using PLUMED to run the simualtions and PyMBAR to reweight them. It can be used for umbrella sampling which uses harmonic restraints, a static external biasing potential via PLUMED EXTERNAL, or a combintaion of them both. This is implemented for both 1D and 2D umbrella sampling simulations. 

I have defined classes to hold data and make reweighting these simulations easy. I define an umbrella set as any simulations which may differ in their harmonic restraints but share a common static bias potential. I define an umbrella collection to be a group of umbrella sets which will all be reweighted together. Each set may have different harmonic restraints and differnt extenal biases from the other sets in the collection.
