from os.path import join, basename, splitext
import os
from os import mkdir
from pathlib import Path
import random

import numpy as np
import scipy as sp
import argparse

sizes_toeplitz = [100, 1000]

# Erzeugt Matrix mit -2 auf der Hauptdiagonale und 1 auf den Nebendiagonalen
diag_vals = [1, -2, 1]
diag_pos = [-1, 0, 1]

pfad = "matrix_instances/generated/toeplitz"





###################

def write_mtx_coo(fname, B):
    with open(fname, "w") as f:
        f.write("%d %d %d\n" % (B.shape[0], B.shape[1], B.data.shape[0]))
        for i, j, x in zip(B.coords[0], B.coords[1], B.data):
            f.write("%d %d %.20f\n" %(i+1,j+1,x))


###################



# A = sp.sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n,n)).tocoo()
# B = sp.sparse.diags_array([1, -2, 1], offsets=[-1, 0, 1], shape=(n,n))


Path(pfad).mkdir(parents=True, exist_ok=True)

for n in sizes_toeplitz:
    A = sp.sparse.diags(diag_vals, diag_pos, shape=(n,n)).tocoo()
    
    fname = join(pfad, "toeplit_" + str(n) + "_A.mtx")
    
    write_mtx_coo(fname, A)
    











