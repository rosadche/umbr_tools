# Umbrella 1D
# Reilly Osadchey Brown

import pandas as pd
import numpy as np

import pymbar

import os
import shutil


class umbrella_set(object):

    """
    An Umbrella Set is defined as:
        any number of umbrella windows that share a static bias term
        they may differ in harmonic umbrella terms
    """

    def __init__(self, name):
        self.name = name

    def umbr_harm(self, min_val, max_val, num_im, endpoint, k):
        self.num_im = num_im
        self.umbr_harm_locs = np.linspace(
            min_val, max_val, num=num_im, endpoint=endpoint
        )

        if isinstance(k, list) or isinstance(k, np.ndarray):
            if len(k) != self.num_im:
                self.umbr_harm_ks = np.asarray(k)
            else:
                raise Exception("List/Array is the wrong length!")
        else:
            self.umbr_harm_ks = np.full(self.num_im, k)

    def umbr_static(self, file=None, need_to_invert=False):
        """
        This just stores information, does not compute spline
        File must be the file used by PLUMED as external bias
        """
        self.umbr_static_file = file
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

    def cv_file(self, file_base):
        self.cv_files = [file_base.format(i=x) for x in range(self.num_im)]


class adaptive_umbrella_collection(object):

    """
    An Umbrella Collection is defined as any number of umbrella sets.
    Simulations within a set share a static bias
    An umbrella collection therefore is used for when you have different static biases
    """

    def __init__(self, dict_umbrella_sets):
        # takes a dictionary of umbrella sets
        if not isinstance(dict_umbrella_sets, dict):
            raise Exception(
                "Collection of umbrella sets must be provided as dictionary!"
            )
        else:
            # ----- Parse Key Data from the dictionary data of Umbrella Sets ---------
            self.dict_umbr_sets = dict_umbrella_sets
            self.umbr_sets_keys = list(self.dict_umbr_sets.keys())

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
        # print( "".join(message) )
        # load the data back in
        bias_df = read_colvar("colvar_all_bias", keep_zero=True)
        print(bias_df)
        bias_df.drop(labels=["time"], axis="columns", inplace=True)
        self.bias_df = bias_df

        # delete the files we created for this purpose
        os.popen(f"rm {colvar_file}").readlines()
        os.popen("rm plumed.dat").readlines()
        os.popen("rm colvar_all_bias").readlines()
        shutil.rmtree("./reweight_scratch")

    def do_mbar(self, umbr_options=None, mbar_options=None):
        needed_umbr_options = (
            "cv_col",
            "N_max",
            "nbins",
            "outtemp",
            "pot_ener_col",
            "g",
            "verbose",
            "outfile",
            "units",
            "kB",
        )

        if umbr_options == None:
            umbr_options = dict()
        for option in needed_umbr_options:
            if option not in needed_umbr_options:
                umbr_options[option] = None

        if umbr_options["cv_col"] == None:
            raise Exception("Must provide cv_col!")
        else:
            self.cv_col = cv_col

        if umbr_options["verbose"] == None:
            self.verbose = True
        else:
            self.verbose = umbr_options["verbose"]

        if umbr_options["outfile"] == None:
            self.outfile = "fes.dat"
        else:
            self.outfile = umbr_options["outfile"]

        # if all temperatures are not all the same, must provide a potential energy column in COLVAR fikle
        if (self.pot_ener_col == None) and not (np.all(self.T_k == self.T_k[0])):
            raise Exception(
                "A column containing potential energy must be given when reweighting with simulations at different temperatures!"
            )

        # if g is a constant, lets fill the g_k array with it
        # if g is a list or array, make sure it is correct length and if so make that the g_k array
        # if g=None, make sure to initialize the array anyway
        self.g_k = np.zeros([self.K])  # statistical inefficiency of simualtion k
        if g != None:
            if isinstance(g, list) or isinstance(g, np.ndarray):
                if len(g) != self.K:
                    raise Exception(
                        "g as an array must be list/array of integers with len of total windows!"
                    )
                else:
                    self.g_k[:] = g
            else:
                if not isinstance(g, int):
                    raise Exception(
                        "g must bean integer, an list/array of integers, or None"
                    )
                else:
                    self.g_k[:] = g

        # ------ Allocate Storage --------------------------------------------------------------
        self.N_k = np.zeros(
            [self.K], dtype=int
        )  # N_k[k] is the number of snapshots from umbrella simulation k

        self.restraint_k = np.zeros(
            [self.K, 2]
        )  # Retraint_k[k] is the Umbrella spring constant and center vals for simualtion k: r1, k1

        self.cv_mat_kn = np.zeros(
            [self.K, N_max]
        )  # cv_mat_kn[k,n] is the CV value for snapshot n from umbrella simulation k
        self.cv_mat_kn[:] = np.nan

        self.u_kn = np.zeros(
            [self.K, N_max]
        )  # u_kn[k,n] is the reduced potential energy without umbrella restraints of snapshot n of umbrella simulation k (only needed if T is not constant)
        self.u_kln = np.zeros(
            [self.K, self.K, N_max]
        )  # u_kln[k,l,n] is the reduced potential energy of snapshot n from umbrella simulation k evaluated at umbrella l
        self.b_kln = np.zeros(
            [self.K, self.K, N_max]
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

            file = self.dict_umbr_sets[which_set].cv_files[image_in_set]
            df = pd.read_pickle(file)

            cv_vals = df[cv_col].to_numpy()
            self.cv_mat_kn[k, 0 : len(cv_vals)] = cv_vals

            if pot_ener_col:
                self.u_kn[k, 0 : len(cv_vals)] = df[pot_ener_col].to_numpy()

            # Subsample our data
            # If g not provided, calculate statistical inefficiency
            if g == None:
                self.g_k[k] = timeseries.statistical_inefficiency(cv_vals)
            # get indices of subsampled timeseries
            indices = timeseries.subsample_correlated_data(
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
            self.cv_mat_kn, bins=nbins, range=(self.cv_min, self.cv_max), density=False
        )
        cv_min, cv_max = check_histogram(hist, bins)

        # ------ Set Up For Final MBAR --------------------------------------------------------------
        # Evaluate reduced energies in all umbrellas
        if self.verbose:
            print("Evaluating reduced potential energies...")
        # Set zero of u_kn -- this is arbitrary.
        self.u_kn -= self.u_kn.min()  # arbitrary up to a constant
        eval_reduced_pot_energies(
            self.N_k,
            self.u_kln,
            self.u_kn,
            self.beta_k,
            self.cv_mat_kn,
            self.restraint_k,
            self.b_kln,
        )

        # compute bin centers
        bin_center_i = np.zeros([nbins])
        bin_edges = np.linspace(cv_min, cv_max, nbins + 1)
        for i in range(nbins):
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
        center_f_i = kBT * results["f_i"]
        center_df_i = kBT * results["df_i"]

        # Write out free energy profile
        text = f"# free energy profile from histogramming" + "\n"
        text += f"# provided units: {self.units}" + "\n"
        text += f"# provided value for kB: {self.kB}"
        text += f"# provided T={self.outtemp} K, resulitng in KT={self.kT} {self.units}"
        text += f"{'bin':>8s} {'f':>8s} {'df':>8s}" + "\n"
        for i in range(nbins):
            text += (
                f"{bin_center_i[i]:8.3f} {center_f_i[i]:8.3f} {center_df_i[i]:8.3f}"
                + "\n"
            )

        if self.verbose:
            print(text)

        with open(self.outfile, "w") as f:
            f.write(text)
