from os.path import join, basename, splitext
import random

import numpy as np
import scipy as sp


# RANDOM MATRIX GENERATOR FOR TESTING PURPOSES
# Generated matrices A, B, and C = A*B for `N_matrices` test case
# and saves them in specified directory.
# A and B have max shize `max_size` in each dimension


# TODO: "Echte" Sparse-Matrizen generieren!


N_matrices = 2
base_dir = "matrix_instances/generated"

max_size = 5


#############################

def write_mtx(fname, B):
    with open(fname, "w") as f:
        f.write("%d %d %d\n" % (B.shape[0], B.shape[1], B.shape[0]*B.shape[1]))
        for i in range(0, B.shape[0]):
            for j in range(0, B.shape[1]):
                f.write("%d %d %.20f\n" %(i+1,j+1, B[i,j]))
                
#############################

for testcase in range(0, N_matrices):
    n = random.randint(1, max_size)
    m = random.randint(1, max_size)
    l = random.randint(1, max_size)
    
    A = np.random.rand(n,m)
    B = np.random.rand(m,l)
    C = A.dot(B)
    
    fname_A = join(base_dir, "case_" + str(testcase).zfill(4) + "_A.mtx")
    fname_B = join(base_dir, "case_" + str(testcase).zfill(4) + "_B.mtx")
    fname_C = join(base_dir, "case_" + str(testcase).zfill(4) + "_C.mtx")
    
    write_mtx(fname_A, A)
    
    
    
    
    
    
    
    
