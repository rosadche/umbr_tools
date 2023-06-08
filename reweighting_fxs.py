# Reweighting Functions
# Reilly Osadchey Brown

import pandas as pd
import numpy as np
import numba


@numba.jit(nopython=True)
def harmonic_umbrella_bias(cv_val, cv_details):
    """ """
    temp1 = 0.5 * cv_details[1]
    temp2 = np.power((cv_val - cv_details[0]), 2)
    U = temp1 * temp2

    return U


@numba.jit(nopython=True)
def eval_reduced_pot_energies_1d(N_k, u_kln, u_kn, beta_k, cv_mat_kn, restraint_k, b_kln):
    """ """
    for k in range(len(N_k)):
        for n in range(N_k[k]):
            for l in range(len(N_k)):
                # Compute energy of snapshot n from simulation k in umbrella potential l
                u_kln[k, l, n] = u_kn[k, n] + beta_k[k] * (
                    harmonic_umbrella_bias(cv_mat_kn[k, n], restraint_k[l])
                    + b_kln[k, l, n]
                )


@numba.jit(nopython=True)
def eval_reduced_pot_energies_2d(N_k, u_kln, u_kn, beta_k, cv_x_kn, restraint_x_k, cv_y_kn, restraint_y_k, b_kln):
    """ """
    for k in range(K):
        for n in range(N_k[k]):
            for l in range(K):
                # Compute energy of snapshot n from simulation k in umbrella potential l
                u_kln[k, l, n] = u_kn[k, n] + beta_k[k] * (harmonic_umbrella_bias(cv_x_kn[k,n], restraint_x_k[l]) + harmonic_umbrella_bias(cv_y_kn[k,n], restraint_y_k[l]) + b_kln[k,l,n] )
    