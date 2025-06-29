from os.path import join, basename, splitext
import os
from os import mkdir
from pathlib import Path
import random

import numpy as np
import scipy as sp
import argparse

########### PARAMETER HIER ANPASSEN

### Ordner, wo die Sachen reinsollen
# Erstellt drei Unterordner: dense, sparse und sparse_vs_dense
# In denen sind jeweils die generierten Matrizen
pfad = "matrix_instances/generated"


### Anzahl der Exemplare
number_of_examples = 1

### DENSE
# Größe variieren
sizes_dense = [1] # Dimensionen
#sizes_dense = [x for x in range(5,251, 5)] # Dimensionen
#sizes_dense = [x for x in range(250,500, 10)] # Dimensionen
#sizes_dense = [x for x in range(600,2500, 100)] # Dimensionen
#sizes_dense = [10, 20,30,40,50,60,70,80,90,100,120,140,160,180,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000] # Dimensionen
#sizes_dense = [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2200,2400,2600,2800,3000] # Dimensionen
#sizes_dense = list(range(10,601, 10)) # Dimensionen


## SPARSE
# Größe variieren
sparsity = 0.001 # Fixe Sparsity
sizes_sparse = [1] # Dimensionen
#sizes_sparse = list(range (500, 3001, 100))
# sparsity = 0.15 # Fixe Sparsity
# sizes_sparse = [10, 20,30,40,50,60,70,80,90,100,120,140,160,180,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000] # Dimensionen



## DENSE vs SPARSE
# Sparsity variieren
#sparsities = [x/1000 for x in range(5, 101, 5)] # Sparsity
#sparsities = [1] # Sparsity
#size_sparse_vs_dense = 10 # Fixe Dimension
#size_sparse_vs_dense = 10 # Fixe Dimension
sparsities = [0.05, 0.1, 0.2, 0.3, 0.4,0.5,0.6, 0.7,0.8,0.9, 0.15, 0.25, 0.35, 0.45,0.55,0.65, 0.75,0.85,0.95]
size_sparse_vs_dense = 500 #  Fixe Dimension


########### PARAMETER ENDE

#############################

# Writes matrix B into file fname as mtx-file

def write_mtx_coo(fname, B):
    with open(fname, "w") as f:
        f.write("%d %d %d\n" % (B.shape[0], B.shape[1], B.data.shape[0]))
        for i, j, x in zip(B.coords[0], B.coords[1], B.data):
            f.write("%d %d %.20f\n" %(i+1,j+1,x))

def write_mtx_dense(fname, B):
    with open(fname, "w") as f:
        f.write("%d %d %d\n" % (B.shape[0], B.shape[1], B.shape[0]*B.shape[1]))
        for i in range(0, B.shape[0]):
            for j in range(0, B.shape[1]):
                f.write("%d %d %.20f\n" %(i+1,j+1, B[i,j]))
#############################

for example_count in range(0, number_of_examples):
    example_count = str(example_count).zfill(len(str(number_of_examples)))
    ### DENSE
    for n in sizes_dense:
        # Generate
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)


        # Write
        Path(join(pfad, "dense")).mkdir(parents=True, exist_ok=True)

        fname = join(pfad, "dense", "dense_" + str(n) + "_A" + example_count + ".mtx")
        write_mtx_dense(fname, A)
        fname = join(pfad, "dense", "dense_" + str(n) + "_B" + example_count + ".mtx")
        write_mtx_dense(fname, B)


    ### Sparse
    for n in sizes_sparse:
        print("generate matrix")
        # Paths
        Path(join(pfad, "sparse")).mkdir(parents=True, exist_ok=True)
        fname_A = join(pfad, "sparse", "sparse_" + str(n) + "_A" + example_count + ".mtx")
        fname_B = join(pfad, "sparse", "sparse_" + str(n) + "_B" + example_count + ".mtx")

        with open(fname_A, "w") as f:
            N = int(n*n*sparsity)
            f.write("%d %d %d\n" % (n, n, N))

            # Ensure (1,1) is non-zero
            f.write("%d %d %.20f\n" %(1, 1, np.random.rand() + 0.1)) # Add a small offset to ensure it's not zero

            for _ in range(0, N - 1):
                i = np.random.randint(1, n)
                j = np.random.randint(1, n)
                x = np.random.rand()
                f.write("%d %d %.20f\n" %(i+1,j+1,x))

        with open(fname_B, "w") as f:
            N = int(n*n*sparsity)
            f.write("%d %d %d\n" % (n, n, N))

            # Ensure (1,1) is non-zero
            f.write("%d %d %.20f\n" %(1, 1, np.random.rand() + 0.1)) # Add a small offset to ensure it's not zero

            for _ in range(0, N - 1): # Generate N-1 additional entries
                i = np.random.randint(1, n)
                j = np.random.randint(1, n)
                x = np.random.rand()
                f.write("%d %d %.20f\n" %(i+1,j+1,x))


    ### Sparse vs Dense
    for spa in sparsities:
        # Paths
        Path(join(pfad, "sparse-vs-dense")).mkdir(parents=True, exist_ok=True)
        fname_A = join(pfad, "sparse-vs-dense", "s-vs-d_" + str(spa) + "_A" + example_count + ".mtx")
        fname_B = join(pfad, "sparse-vs-dense", "s-vs-d_" + str(spa) + "_B" + example_count + ".mtx")

        n = size_sparse_vs_dense

        with open(fname_A, "w") as f:
            N = int(size_sparse_vs_dense*size_sparse_vs_dense*spa)
            f.write("%d %d %d\n" % (n, n, N))

            # Ensure (1,1) is non-zero
            f.write("%d %d %.20f\n" %(1, 1, np.random.rand() + 0.1)) # Add a small offset to ensure it's not zero

            for _ in range(0, N - 1): # Generate N-1 additional entries
                i = np.random.randint(1, n)
                j = np.random.randint(1, n)
                x = np.random.rand()
                f.write("%d %d %.20f\n" %(i+1,j+1,x))

        with open(fname_B, "w") as f:
            N = int(size_sparse_vs_dense*size_sparse_vs_dense*spa)
            f.write("%d %d %d\n" % (n, n, N))

            # Ensure (1,1) is non-zero
            f.write("%d %d %.20f\n" %(1, 1, np.random.rand() + 0.1)) # Add a small offset to ensure it's not zero

            for _ in range(0, N - 1): # Generate N-1 additional entries
                i = np.random.randint(1, n)
                j = np.random.randint(1, n)
                x = np.random.rand()
                f.write("%d %d %.20f\n" %(i+1,j+1,x))
    

    
