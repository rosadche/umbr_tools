# Misc. Functions
# Reilly Osadchey Brown

import pandas as pd
import numpy as np


def read_colvar(filename, keep_zero=True):
    with open(filename, "r") as f:
        columns = f.readline()

    columns = columns.rstrip("\n").split()

    df = pd.read_csv(filename, sep="\s+", comment="#", header=None)

    df.columns = columns[2:]

    # Removes the zero time rows which occur in Plumed file every run with CHARMM
    if keep_zero == False:
        df = df.loc[df["time"] != 0.0]

    # reset times to start at zero
    df["time"] = df["time"] - df["time"].min()

    # remove any duplicate times. Would only occur if a simualtion started, then fialed, then was restarted without backing up the
    # COLVAR file to before the failed run. This means you never have to worry about this failed simualtion issue
    df.drop_duplicates(subset="time", keep="last", inplace=True, ignore_index=True)

    return df
