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
number_of_examples = 2

### DENSE
# Größe variieren
sizes_dense = [1] # Dimensionen


## SPARSE
# Größe variieren
sparsity = 0.01 # Fixe Sparsity
sizes_sparse = [1] # Dimensionen
sizes_sparse.sort()


## DENSE vs SPARSE
# Sparsity variieren
sparsities = [0.01] # Sparsity
size_sparse_vs_dense = 2000 # Fixe Dimension


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
        # Paths
        Path(join(pfad, "sparse")).mkdir(parents=True, exist_ok=True)
        fname_A = join(pfad, "sparse", "sparse_" + str(n) + "_A" + example_count + ".mtx")
        fname_B = join(pfad, "sparse", "sparse_" + str(n) + "_B" + example_count + ".mtx")
        
        with open(fname_A, "w") as f:
            N = int(n*n*sparsity) #  Anzahl Einträge gesamt
            f.write("%d %d %d\n" % (n, n, N))
            
            for _ in range(0, N):
                i = np.random.randint(1, n)
                j = np.random.randint(1, n)
                x = np.random.rand()
                f.write("%d %d %.20f\n" %(i+1,j+1,x))
                
        with open(fname_B, "w") as f:
            N = int(n*n*sparsity) #  Anzahl Einträge gesamt
            f.write("%d %d %d\n" % (n, n, N))
            
            for _ in range(0, N):
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
            N = int(size_sparse_vs_dense*size_sparse_vs_dense*spa) #  Anzahl Einträge gesamt
            f.write("%d %d %d\n" % (n, n, N))
            
            for _ in range(0, N):
                i = np.random.randint(1, n)
                j = np.random.randint(1, n)
                x = np.random.rand()
                f.write("%d %d %.20f\n" %(i+1,j+1,x))
                
        with open(fname_B, "w") as f:
            N = int(size_sparse_vs_dense*size_sparse_vs_dense*spa) #  Anzahl Einträge gesamt
            f.write("%d %d %d\n" % (n, n, N))
            
            for _ in range(0, N):
                i = np.random.randint(1, n)
                j = np.random.randint(1, n)
                x = np.random.rand()
                f.write("%d %d %.20f\n" %(i+1,j+1,x))
    

    
