from os.path import join, basename, splitext
import random

import numpy as np
import scipy as sp
import argparse



# RANDOM MATRIX GENERATOR FOR TESTING PURPOSES
# Generated matrices A, B, and C = A*B for `N_matrices` test case
# and saves them in specified directory.
# A and B have max size `max_size` in each dimension
# and a maximum density of max_density



####### CLI when not run in Spyder or so #######

parser = argparse.ArgumentParser(description="Generate sparse matrices.")
parser.add_argument("fname", type=str, help="Name of the mtx-files")
parser.add_argument("--n_matrices", type=int, default=10, help="Number of matrices")
parser.add_argument("--max_size", type=int, default=100, help="Maximum matrix size")
parser.add_argument("--max_density", type=float, default=0.2, help="Maximum matrix density (0=all zero, 1=dense)")
parser.add_argument("--basedir", type=str, default="./matrix_instances/generated", help="Base directory for output, defaults to matrix_instances/generated")

args = parser.parse_args()

print("Input file:", args.fname)
print("Number of matrices:", args.n_matrices)
print("Maximum size:", args.max_size)
print("Maximum density:", args.max_density)
print("Base directory:", args.basedir)


fname = args.fname
N_matrices = args.n_matrices
max_size = args.max_size
max_density = args.max_density
base_dir = args.basedir


print("Generating %d sparse matrices of maximum size %d and density %f with name %s in %s" % (N_matrices, max_size, max_density, fname, base_dir))

## When run in Spyder, hardcoded

# N_matrices = 2
# base_dir = "matrix_instances/generated"

# max_size = 100
# max_density = 0.2


#############################

# Writes dense matrix B into file fname as mtx-file
def write_mtx_coo(fname, B):
    with open(fname, "w") as f:
        f.write("%d %d %d\n" % (B.shape[0], B.shape[1], B.data.shape[0]))
        for i, j, x in zip(B.coords[0], B.coords[1], B.data):
            f.write("%d %d %.20f\n" %(i+1,j+1,x))
                
#############################

for testcase in range(0, N_matrices):
    n = random.randint(1, max_size)
    m = random.randint(1, max_size)
    l = random.randint(1, max_size)
    density = random.random() * max_density
    
    
    A = sp.sparse.random_array((n,m), density=density)
    B = sp.sparse.random_array((m,l), density=density)
    C = A.dot(B).tocoo()
    
    fname_A = join(base_dir, fname + "_" + str(testcase).zfill(4) + "_A.mtx")
    fname_B = join(base_dir, fname + "_" + str(testcase).zfill(4) + "_B.mtx")
    fname_C = join(base_dir, fname + "_" + str(testcase).zfill(4) + "_C.mtx")

    write_mtx_coo(fname_A, A)
    write_mtx_coo(fname_B, B)
    write_mtx_coo(fname_C, C)
    
    
    
    
    
    
    
    
    
