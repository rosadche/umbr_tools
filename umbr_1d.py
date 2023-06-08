# Umbrella 1D
# Reilly Osadchey Brown

import pandas as pd
import numpy as np

import pymbar

import os
import shutil

from umbr_tools.misc_fxs import read_colvar
from umbr_tools.reweighting_fxs import harmonic_umbrella_bias, eval_reduced_pot_energies_1d

class umbrella_set_1d(object):

    """
    An Umbrella Set is defined as:
        any number of umbrella windows that share a static bias term
        they may differ in harmonic umbrella terms
    """

    def __init__(self, name):
        self.name = name

    def umbr_harm(self, umbr_harm_locs, k):
        
        if isinstance(umbr_harm_locs, list) or isinstance(umbr_harm_locs, np.ndarray):
            self.umbr_harm_locs = umbr_harm_locs
            self.num_im = len(self.umbr_harm_locs)
        else:
            raise Exception("umbr_harm_locs must be an array/list!")
                
        
        if isinstance(k, list) or isinstance(k, np.ndarray):
            if len(k) != self.num_im:
                self.umbr_harm_ks = np.asarray(k)
            else:
                raise Exception("List/Array is the wrong length!")
        else:
            self.umbr_harm_ks = np.full(self.num_im, k)

    def umbr_static(self, file=None, need_to_invert=False, static_bias_col=None):
        """
        If performing MBAR on the set only, please supply static_bias_col
        If performing MBAR on a collection, please supply file, need_to_invert
        
        This just stores information, does not compute spline
        File must be the file used by PLUMED as external bias
        """
        self.umbr_static_file = file
        self.static_bias_col = static_bias_col
        
        if need_to_invert == True:
            self.static_scale = -1.0
        else:
            self.static_scale = 1.0

    def sim_temp(self, temp):
        if isinstance(temp, list) or isinstance(temp, np.ndarray):
            if len(temp) != self.num_im:
                self.temps = np.asarray(temp)
            else:
                raise Exception("List/Array is the wrong length!")
        else:
            self.temps = np.full(self.num_im, temp)

    def cv_file(self, files, file_type):
        """
        files can be:
            - a string with {i} for format (range up to num_im),
            - a list/array of lenght num_images
        file_type can be:
            "colvar": a plumed COLVAR file
            "df": a pandas dataframe directly
            "df_pkl": a pickle of a pandas dataframe
        """

        if isinstance(files, str):
            self.cv_files = [files.format(i=x) for x in range(self.num_im)]
        elif isinstance(files, list) or isinstance(files, np.ndarray):
            if len(files) == self.num_im:
                self.cv_files = files
            else:
                raise Exception("List/Array is the wrong length!")
        else:
            raise Exception("Files not provided as correct type!")

        filetype_options = ["colvar", "df", "df_pkl"]
        if file_type.lower() not in filetype_options:
            raise Exception(
                "file_type muse be  not one of: {}!".format(", ".join(filetype_options))
            )
        else:
            self.file_type = file_type
        
    def do_mbar(self, umbr_options=None, mbar_options=None):
        """ """

        # start with options
        needed_umbr_options = (
            "cv_col",
            "N_max",
            "nbins",
            "outtemp",
            "pot_ener_col",
            "g",
            "verbose",
            "outfile",
            "kB",
            "units"
            "static_bias_col"
        )

        if umbr_options == None:
            umbr_options = dict()
        for option in needed_umbr_options:
            if option not in needed_umbr_options:
                umbr_options[option] = None

        # Required Arguements (No Default Value)
        if umbr_options["cv_col"] == None:
            raise Exception("Must provide cv_col!")
        else:
            self.cv_col = umbr_options["cv_col"]

        if umbr_options["N_max"] == None:
            raise Exception("Must provide N_max!")
        else:
            self.N_max = umbr_options["N_max"]

        if umbr_options["outtemp"] == None:
            raise Exception("Must provide outtemp (in Kelvin)!")
        else:
            self.outtemp = umbr_options["outtemp"]
        
        if umbr_options["kB"] == None:
            raise Exception("Must provide kB!")
        else:
            if isinstance(umbr_options["kB"], float):
                self.kB = umbr_options["kB"]
            else:
                raise Exception("kB must be a float value!")
            
        if umbr_options["units"] == None:
            raise Exception("Must provide units!")
        else:
            if isinstance(umbr_options["units"], str):
                self.units = umbr_options["units"]
            else:
                raise Exception(
                    "units must be a string (just for clarifying in output files)!"
                )
            
        if umbr_options["nbins"] == None:
            self.nbins = 100
        else:
            self.nbins = umbr_options["nbins"]

        if umbr_options["pot_ener_col"] == None:
            self.pot_ener_col = None
            # if all temperatures are not all the same, must provide a potential energy column in COLVAR fikle
            if not (np.all(self.temps == self.temps[0])):
                raise Exception(
                    "A column containing potential energy must be given when reweighting with simulations at different temperatures!"
                )
        else:
            self.pot_ener_col = umbr_options["pot_ener_col"]

        if umbr_options["verbose"] == None:
            self.verbose = True
        else:
            self.verbose = umbr_options["verbose"]

        if umbr_options["outfile"] == None:
            self.outfile = "fes.dat"
        else:
            self.outfile = umbr_options["outfile"]
        
        # we cannot procced if static_bias_col was not provided and an external bias file was
        if self.static_bias_col == None and self.umbr_static_file == None:
            if self.verbose:
                print("No static bias provided for this umbrella set, so no external bias used")
            elif self.static_bias_col == None and self.umbr_static_file != None:
                raise Exception(
                            "A static bias column name must be provided in the case of reweighting an Umbrella Set. Will NOT use an external bias file directly."
                        )
            elif self.static_bias_col != None and self.umbr_static_file == None:
                if self.verbose:
                    print("A static bias column name was provided - will be used for reweighting.")
            else:
                if self.verbose:
                    print("Both a static bias column and an external bias file provided. No conflict, but will ONLY use static bias column name.")
                
        
        # need to setup KT at outtemp
        self.kT = (self.kB * self.outtemp)
        
        # need to get beta_k array:
        self.beta_k = 1.0 / (self.temps * self.kB)
        # we want self.K but have self.num_im
        self.K = self.num_im
        
        # if g is a constant, lets fill the g_k array with it
        # if g is a list or array, make sure it is correct length and if so make that the g_k array
        # if g is None, then later we will use pymbar's timeseries module to calculate it
        self.g = umbr_options["g"]
        self.g_k = np.zeros([self.K])  # statistical inefficiency of simulation k
        if self.g != None:
            if isinstance(self.g, list) or isinstance(self.g, np.ndarray):
                if len(self.g) != self.K:
                    raise Exception(
                        "g as an array/list must be of same length as number of total windows!"
                    )
                else:
                    self.g_k[:] = self.g
            else:
                if not isinstance(self.g, int):
                    raise Exception(
                        "g must bean integer, an list/array of integers, or None"
                    )
                else:
                    self.g_k[:] = self.g

        # ------ Allocate Storage --------------------------------------------------------------
        self.N_k = np.zeros(
            [self.K], dtype=int
        )  # N_k[k] is the number of snapshots from umbrella simulation k

        self.restraint_k = np.zeros(
            [self.K, 2]
        )  # Retraint_k[k] is the Umbrella spring constant and center vals for simualtion k: r1, k1

        self.cv_mat_kn = np.zeros(
            [self.K, self.N_max]
        )  # cv_mat_kn[k,n] is the CV value for snapshot n from umbrella simulation k
        self.cv_mat_kn[:] = np.nan

        self.u_kn = np.zeros(
            [self.K, self.N_max]
        )  # u_kn[k,n] is the reduced potential energy without umbrella restraints of snapshot n of umbrella simulation k (only needed if T is not constant)
        self.u_kln = np.zeros(
            [self.K, self.K, self.N_max]
        )  # u_kln[k,l,n] is the reduced potential energy of snapshot n from umbrella simulation k evaluated at umbrella l
        self.b_kln = np.zeros(
            [self.K, self.K, self.N_max]
        )  # b_kln[k,l,n] is the static bias of snapshot n (from umbrella simulation k) evaluated at umbrella l

        # ------ Load Data --------------------------------------------------------------
        if self.verbose:
            print("Loading Data...")
        self.cv_min = np.inf
        self.cv_max = -np.inf

        # we will first just load in the data for all images
        for k in range(self.K):

            cv_details = np.asarray(
                [
                    self.umbr_harm_locs[k],
                    self.umbr_harm_ks[k],
                ]
            )
            self.restraint_k[k, :] = cv_details

            if self.file_type == "colvar":
                df = read_colvar(
                    self.cv_files[k],
                    keep_zero=False,
                )
            if self.file_type == "df":
                df = self.cv_files[k]
            if self.file_type == "df_pkl":
                df = pd.read_pickle(
                    self.cv_files[k]
                )
            else:
                raise Exception("This CV filetype has not been implemented!")

            cv_vals = df[self.cv_col].to_numpy()
            self.cv_mat_kn[k, 0 : len(cv_vals)] = cv_vals

            if self.pot_ener_col:
                self.u_kn[k, 0 : len(cv_vals)] = df[self.pot_ener_col].to_numpy()
            
            if self.static_bias_col:
                self.b_kln[k, :, 0 : len(cv_vals)] = df[self.static_bias_col].to_numpy()
                
            # Subsample our data
            # If g not provided, calculate statistical inefficiency
            if self.g == None:
                self.g_k[k] = pymbar.timeseries.statistical_inefficiency(cv_vals)
            # get indices of subsampled timeseries
            indices = pymbar.timeseries.subsample_correlated_data(
                self.cv_mat_kn[k, 0 : len(cv_vals)], g=self.g_k[k]
            )

            # Subsample data.
            self.N_k[k] = len(indices)
            self.u_kn[k, 0 : self.N_k[k]] = self.u_kn[k, indices]
            self.cv_mat_kn[k, 0 : self.N_k[k]] = self.cv_mat_kn[k, indices]

            if self.verbose:
                print(
                    f"image {k}: stat_ineff={self.g_k[k]} for {self.N_k[k]} frames"
                )

            if np.nanmin(self.cv_mat_kn[k, 0 : self.N_k[k]]) < self.cv_min:
                self.cv_min = np.nanmin(self.cv_mat_kn[k, 0 : self.N_k[k]])

            if np.nanmax(self.cv_mat_kn[k, 0 : self.N_k[k]]) > self.cv_max:
                self.cv_max = np.nanmax(self.cv_mat_kn[k, 0 : self.N_k[k]])

        hist, bins = np.histogram(
            self.cv_mat_kn,
            bins=self.nbins,
            range=(self.cv_min, self.cv_max),
            density=False,
        )

        # ------ Set Up For Final MBAR --------------------------------------------------------------
        # Evaluate reduced energies in all umbrellas
        if self.verbose:
            print("Evaluating reduced potential energies...")
        # Set zero of u_kn -- this is arbitrary.
        self.u_kn -= self.u_kn.min()  # arbitrary up to a constant
        eval_reduced_pot_energies_1d(
            self.N_k,
            self.u_kln,
            self.u_kn,
            self.beta_k,
            self.cv_mat_kn,
            self.restraint_k,
            self.b_kln,
        )

        # compute bin centers
        bin_center_i = np.zeros([self.nbins])
        bin_edges = np.linspace(self.cv_min, self.cv_max, self.nbins + 1)
        for i in range(self.nbins):
            bin_center_i[i] = 0.5 * (bin_edges[i] + bin_edges[i + 1])

        N = np.sum(self.N_k)
        cv_n = pymbar.utils.kn_to_n(self.cv_mat_kn, N_k=self.N_k)

        # initialize free energy profile with the data collected
        if self.verbose:
            print("Creating FES Object...")

        fes = pymbar.FES(self.u_kln, self.N_k, mbar_options=mbar_options)

        # Compute free energy profile in unbiased potential (in units of kT).
        histogram_parameters = {}
        histogram_parameters["bin_edges"] = bin_edges

        if self.verbose:
            print("Generating FES...")
        fes.generate_fes(
            self.u_kn,
            cv_n,
            fes_type="histogram",
            histogram_parameters=histogram_parameters,
        )
        results = fes.get_fes(
            bin_center_i, reference_point="from-lowest", uncertainty_method="analytical"
        )
        center_f_i = self.kT * results["f_i"]
        center_df_i = self.kT * results["df_i"]

        # Write out free energy profile
        text = f"# free energy profile from histogramming" + "\n"
        text += f"# provided units: {self.units}" + "\n"
        text += f"# provided value for kB: {self.kB} {self.units}/K" + "\n"
        text += f"# provided T={self.outtemp} K, resulitng in kT={self.kT} {self.units}" + "\n"
        text += f"{'bin':>8s} {'f':>8s} {'df':>8s}" + "\n"
        for i in range(self.nbins):
            text += (
                f"{bin_center_i[i]:8.3f} {center_f_i[i]:8.3f} {center_df_i[i]:8.3f}"
                + "\n"
            )

        if self.verbose:
            print(text)

        with open(self.outfile, "w") as f:
            f.write(text)


class umbrella_collection_1d(object):

    """
    An Umbrella Collection is defined as any number of umbrella sets.
    Simulations within a set share a static bias
    An umbrella collection therefore is used for when you have different static biases
    """

    def __init__(self, dict_umbrella_sets, kB, units):
        # takes a dictionary of umbrella sets
        if not isinstance(dict_umbrella_sets, dict):
            raise Exception(
                "Collection of umbrella sets must be provided as dictionary!"
            )
        else:
            # ----- Parse Key Data from the dictionary data of Umbrella Sets ---------
            self.dict_umbr_sets = dict_umbrella_sets
            self.umbr_sets_keys = list(self.dict_umbr_sets.keys())

            if isinstance(kB, float):
                self.kB = kB
            else:
                raise Exception("kB must be a float value!")

            if isinstance(units, str):
                self.units = units
            else:
                raise Exception(
                    "units must be a string (just for clarifying in output files)!"
                )

            self.K_per_set = np.asarray(
                [self.dict_umbr_sets[x].num_im for x in self.umbr_sets_keys]
            )  # many harmonic umbrellas are in each set for all sets
            self.K = np.sum(self.K_per_set)  # K is total number of images.
            self.K_digitize = np.digitize(
                np.arange(0, self.K), np.cumsum(self.K_per_set)
            )  # which set does every image belong to

            # pull initial temperatures from umbrella sets
            self.T_k = np.concatenate(
                [self.dict_umbr_sets[x].temps for x in self.umbr_sets_keys]
            )
            self.kT_k = kB * self.T_k
            # inverse temperature of simulations
            self.beta_k = 1.0 / self.kT_k

    def get_plumed_external_biases(self):
        # write a fake COLVAR file concatenating all CVs from all simulations
        # but in order so we can use the biases easily
        colvar_file = "COLVAR_fake"
        colvar_txt = f"#! FIELDS time {self.cv_col}" + "\n"
        for k in range(self.K):
            for n in range(self.N_k[k]):
                colvar_txt += f"0.0 {self.cv_mat_kn[k,n]}" + "\n"

        with open(colvar_file, "w") as f:
            f.write(colvar_txt)

        # write plumed file to actually calcualte all external biases for all frames
        plumed_txt = ""
        plumed_txt += (
            f"{self.cv_col}: READ FILE={colvar_file} VALUES={self.cv_col} IGNORE_TIME IGNORE_FORCES"
            + "\n"
        )

        # we need to rename the input GRID ext.bias files becasue PLUMED won't accept
        # a different label in the plumed.dat file from the label in grid
        # create scratch dir for this
        if not os.path.exists("./reweight_scratch"):
            os.mkdir("./reweight_scratch")

        for m in range(len(self.umbr_sets_keys)):
            this_umbr_set = self.dict_umbr_sets[self.umbr_sets_keys[m]]
            if this_umbr_set.umbr_static_file != None:
                # copy this grid file to scratch dir, edit the name, and reference that file
                copied_name = f"./reweight_scratch/grid_{m}.dat"
                shutil.copyfile(this_umbr_set.umbr_static_file, copied_name)
                os.popen(f'sed -i "s/ext/ext{m}/g" {copied_name}')
                plumed_txt += (
                    f"ext{m}: EXTERNAL ARG={self.cv_col} FILE={copied_name} SCALE={this_umbr_set.static_scale}"
                    + "\n"
                )

        plumed_txt += "PRINT ARG=* FILE=colvar_all_bias" + "\n"
        with open("plumed.dat", "w") as f:
            f.write(plumed_txt)

        # run the plumed command to do calculation
        print("\nRunning Plumed Driver To Get External Biases")
        message = os.popen(f"plumed driver --noatoms --plumed plumed.dat").readlines()
        if self.verbose:
            print("".join(message))
        # load the data back in
        bias_df = read_colvar("colvar_all_bias", keep_zero=True)
        bias_df.drop(labels=["time"], axis="columns", inplace=True)
        self.bias_df = bias_df

        # delete the files we created for this purpose
        os.popen(f"rm {colvar_file}").readlines()
        os.popen("rm plumed.dat").readlines()
        os.popen("rm colvar_all_bias").readlines()
        shutil.rmtree("./reweight_scratch")

    def do_mbar(self, umbr_options=None, mbar_options=None):
        """ """

        # start with options
        needed_umbr_options = (
            "cv_col",
            "N_max",
            "nbins",
            "outtemp",
            "pot_ener_col",
            "g",
            "verbose",
            "outfile",
        )

        if umbr_options == None:
            umbr_options = dict()
        for option in needed_umbr_options:
            if option not in needed_umbr_options:
                umbr_options[option] = None

        # Required Arguements (No Default Value)
        if umbr_options["cv_col"] == None:
            raise Exception("Must provide cv_col!")
        else:
            self.cv_col = umbr_options["cv_col"]

        if umbr_options["N_max"] == None:
            raise Exception("Must provide N_max!")
        else:
            self.N_max = umbr_options["N_max"]

        if umbr_options["outtemp"] == None:
            raise Exception("Must provide outtemp (in Kelvin)!")
        else:
            self.outtemp = umbr_options["outtemp"]

        # Optional Arguements (Default Values)
        if umbr_options["nbins"] == None:
            self.nbins = 100
        else:
            self.nbins = umbr_options["nbins"]

        if umbr_options["pot_ener_col"] == None:
            self.pot_ener_col = None
            # if all temperatures are not all the same, must provide a potential energy column in COLVAR fikle
            if not (np.all(self.T_k == self.T_k[0])):
                raise Exception(
                    "A column containing potential energy must be given when reweighting with simulations at different temperatures!"
                )
        else:
            self.pot_ener_col = umbr_options["pot_ener_col"]

        if umbr_options["verbose"] == None:
            self.verbose = True
        else:
            self.verbose = umbr_options["verbose"]

        if umbr_options["outfile"] == None:
            self.outfile = "fes.dat"
        else:
            self.outfile = umbr_options["outfile"]
                    
        # if g is a constant, lets fill the g_k array with it
        # if g is a list or array, make sure it is correct length and if so make that the g_k array
        # if g is None, then later we will use pymbar's timeseries module to calculate it
        self.g = umbr_options["g"]
        self.g_k = np.zeros([self.K])  # statistical inefficiency of simulation k
        if self.g != None:
            if isinstance(self.g, list) or isinstance(self.g, np.ndarray):
                if len(self.g) != self.K:
                    raise Exception(
                        "g as an array/list must be of same length as number of total windows!"
                    )
                else:
                    self.g_k[:] = self.g
            else:
                if not isinstance(self.g, int):
                    raise Exception(
                        "g must bean integer, an list/array of integers, or None"
                    )
                else:
                    self.g_k[:] = self.g
        
        # need to setup kT at outtemp
        self.kT = (self.kB * self.outtemp)
        
        # ------ Allocate Storage --------------------------------------------------------------
        self.N_k = np.zeros(
            [self.K], dtype=int
        )  # N_k[k] is the number of snapshots from umbrella simulation k

        self.restraint_k = np.zeros(
            [self.K, 2]
        )  # Retraint_k[k] is the Umbrella spring constant and center vals for simualtion k: r1, k1

        self.cv_mat_kn = np.zeros(
            [self.K, self.N_max]
        )  # cv_mat_kn[k,n] is the CV value for snapshot n from umbrella simulation k
        self.cv_mat_kn[:] = np.nan

        self.u_kn = np.zeros(
            [self.K, self.N_max]
        )  # u_kn[k,n] is the reduced potential energy without umbrella restraints of snapshot n of umbrella simulation k (only needed if T is not constant)
        self.u_kln = np.zeros(
            [self.K, self.K, self.N_max]
        )  # u_kln[k,l,n] is the reduced potential energy of snapshot n from umbrella simulation k evaluated at umbrella l
        self.b_kln = np.zeros(
            [self.K, self.K, self.N_max]
        )  # b_kln[k,l,n] is the static bias of snapshot n (from umbrella simulation k) evaluated at umbrella l

        # ------ Load Data --------------------------------------------------------------
        if self.verbose:
            print("Loading Data...")
        self.cv_min = np.inf
        self.cv_max = -np.inf

        # we will first just load in the data for all images
        for k in range(self.K):
            set_id = self.K_digitize[k]
            set_min_loc_in_K_digitize = np.where(self.K_digitize == set_id)[0][0]
            image_in_set = k - set_min_loc_in_K_digitize
            which_set = self.umbr_sets_keys[set_id]

            cv_details = np.asarray(
                [
                    self.dict_umbr_sets[which_set].umbr_harm_locs[image_in_set],
                    self.dict_umbr_sets[which_set].umbr_harm_ks[image_in_set],
                ]
            )
            self.restraint_k[k, :] = cv_details

            if self.dict_umbr_sets[which_set].file_type == "colvar":
                df = read_colvar(
                    self.dict_umbr_sets[which_set].cv_files[image_in_set],
                    keep_zero=False,
                )
            if self.dict_umbr_sets[which_set].file_type == "df":
                df = self.dict_umbr_sets[which_set].cv_files[image_in_set]
            if self.dict_umbr_sets[which_set].file_type == "df_pkl":
                df = pd.read_pickle(
                    self.dict_umbr_sets[which_set].cv_files[image_in_set]
                )
            else:
                raise Exception("This CV filetype has not been implemented!")

            cv_vals = df[self.cv_col].to_numpy()
            self.cv_mat_kn[k, 0 : len(cv_vals)] = cv_vals

            if self.pot_ener_col:
                self.u_kn[k, 0 : len(cv_vals)] = df[self.pot_ener_col].to_numpy()

            # Subsample our data
            # If g not provided, calculate statistical inefficiency
            if self.g == None:
                self.g_k[k] = pymbar.timeseries.statistical_inefficiency(cv_vals)
            # get indices of subsampled timeseries
            indices = pymbar.timeseries.subsample_correlated_data(
                self.cv_mat_kn[k, 0 : len(cv_vals)], g=self.g_k[k]
            )

            # Subsample data.
            self.N_k[k] = len(indices)
            self.u_kn[k, 0 : self.N_k[k]] = self.u_kn[k, indices]
            self.cv_mat_kn[k, 0 : self.N_k[k]] = self.cv_mat_kn[k, indices]

            if self.verbose:
                print(
                    f"set {which_set} image {image_in_set}: stat_ineff={self.g_k[k]} for {self.N_k[k]} frames"
                )
            
            # need to make sure the external bias file are provided. 
            # probably could check earlier in this loop, but I wantit to print after the information on this window
            if self.dict_umbr_sets[which_set].static_bias_col == None and self.dict_umbr_sets[which_set].umbr_static_file == None:
                if self.verbose:
                    print("No static bias provided for this umbrella set, so no external bias used.")
            elif self.dict_umbr_sets[which_set].static_bias_col == None and self.dict_umbr_sets[which_set].umbr_static_file != None:
                if self.verbose:
                    print("An external bias file was provided - will be used for reweighting.")
            elif self.dict_umbr_sets[which_set].static_bias_col != None and self.dict_umbr_sets[which_set].umbr_static_file == None:
                raise Exception(
                            "An external bias file must be provided in the case of reweighting an Umbrella Set. Will NOT use an static bais column name."
                        )
            else:
                if self.verbose:
                    print("Both a static bias column and an external bias file provided. No conflict, but will ONLY use the external bias file.")
                

            if np.nanmin(self.cv_mat_kn[k, 0 : self.N_k[k]]) < self.cv_min:
                self.cv_min = np.nanmin(self.cv_mat_kn[k, 0 : self.N_k[k]])

            if np.nanmax(self.cv_mat_kn[k, 0 : self.N_k[k]]) > self.cv_max:
                self.cv_max = np.nanmax(self.cv_mat_kn[k, 0 : self.N_k[k]])

        # the above has not yet calculated external bias contribution from all external baises for all subsampled cv_values
        # we do this here
        self.get_plumed_external_biases()  # creates the self.bias_df attribute

        # loop over all the external biases
        for i in range(len(self.umbr_sets_keys)):
            # which external bias is this
            this_umbr_set = self.dict_umbr_sets[self.umbr_sets_keys[i]]
            # which images all share this external bias
            locs = np.where(self.K_digitize == i)[0]

            # if there is no external bias you can just not fill those matrix elements b/c they initialize at 0.0
            if this_umbr_set.umbr_static_file != None:
                ext_bias_vals = self.bias_df[f"ext{i}.bias"].to_numpy()
                N = np.sum(self.N_k)
                # this loops over all images of all sets (whole collection)
                counter = 0
                for k in range(self.K):
                    self.b_kln[k, locs, 0 : self.N_k[k]] = ext_bias_vals[
                        counter : counter + self.N_k[k]
                    ]
                    counter += self.N_k[k]

        hist, bins = np.histogram(
            self.cv_mat_kn,
            bins=self.nbins,
            range=(self.cv_min, self.cv_max),
            density=False,
        )

        # ------ Set Up For Final MBAR --------------------------------------------------------------
        # Evaluate reduced energies in all umbrellas
        if self.verbose:
            print("Evaluating reduced potential energies...")
        # Set zero of u_kn -- this is arbitrary.
        self.u_kn -= self.u_kn.min()  # arbitrary up to a constant
        eval_reduced_pot_energies_1d(
            self.N_k,
            self.u_kln,
            self.u_kn,
            self.beta_k,
            self.cv_mat_kn,
            self.restraint_k,
            self.b_kln,
        )

        # compute bin centers
        bin_center_i = np.zeros([self.nbins])
        bin_edges = np.linspace(self.cv_min, self.cv_max, self.nbins + 1)
        for i in range(self.nbins):
            bin_center_i[i] = 0.5 * (bin_edges[i] + bin_edges[i + 1])

        N = np.sum(self.N_k)
        cv_n = pymbar.utils.kn_to_n(self.cv_mat_kn, N_k=self.N_k)

        # initialize free energy profile with the data collected
        if self.verbose:
            print("Creating FES Object...")

        fes = pymbar.FES(self.u_kln, self.N_k, mbar_options=mbar_options)

        # Compute free energy profile in unbiased potential (in units of kT).
        histogram_parameters = {}
        histogram_parameters["bin_edges"] = bin_edges

        if self.verbose:
            print("Generating FES...")
        fes.generate_fes(
            self.u_kn,
            cv_n,
            fes_type="histogram",
            histogram_parameters=histogram_parameters,
        )
        results = fes.get_fes(
            bin_center_i, reference_point="from-lowest", uncertainty_method="analytical"
        )
        center_f_i = self.kT * results["f_i"]
        center_df_i = self.kT * results["df_i"]

        # Write out free energy profile
        text = f"# free energy profile from histogramming" + "\n"
        text += f"# provided units: {self.units}" + "\n"
        text += f"# provided value for kB: {self.kB} {self.units}/K" + "\n"
        text += f"# provided T={self.outtemp} K, resulitng in kT={self.kT} {self.units}" + "\n"
        text += f"{'bin':>8s} {'f':>8s} {'df':>8s}" + "\n"
        for i in range(self.nbins):
            text += (
                f"{bin_center_i[i]:8.3f} {center_f_i[i]:8.3f} {center_df_i[i]:8.3f}"
                + "\n"
            )

        if self.verbose:
            print(text)

        with open(self.outfile, "w") as f:
            f.write(text)
