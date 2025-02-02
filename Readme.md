# FAKSGPU Scientific Computing Project

## Running the Benchmark  

Before starting, ensure that all paths in `./cusparse/build.rs` match your local CUDA installation.  
Install `libopenblas-dev`
```bash
sudo apt-get install libopenblas-dev
```

### Steps to Run:  

1. Navigate to the project's root directory.  
2. Run the following command:  

   ```bash
   cargo run --release <repeat_count> <path_to_matrices>
   ```  

   - Replace `<repeat_count>` with the number of times you want the benchmark to run.  
   - Replace `<path_to_matrices>` with the directory containing your matrix instances.  

   **Example:**  
   ```bash
   cargo run --release 1 ./matrix_instances/
   ```

This runs the benchmark once using the matrices in `./matrix_instances/`.


## Struktur

### Code

`cpu`: Vergleichsimplementierung: pur CPU
`blas`: Vergleichsimplementierung: cudaBLAS und CPU BLAS
`gpu`: Implementierung auf GPU

### Daten

`matrix_instances`: Matrizen im MTX-Format, eher kleine Dateien
`matrix_instances/generated`: Matrizen im MTX-Format, eher größere. Nicht unter Versionskontrolle

## Skripte

### Sparse Matrix Generation

`generate_matrices_sparse.py`: Generiert zufällige Matrizen zum Test und schreibt sie als MTX-Datei aus. Dateiname muss angegeben werden, Rest sind optionale Parameter. Beispiel:

`python3 generate_matrices_sparse.py --n_matrices=10 --max_size=20 --max_density=0.3 name`: Erzeugt im Ordner `matrix_instances/generated` 10 Matrizen der maximalen Größe 10 (in jeder Dimension) mit einer maximalen Density von 0.3. Die Größe und Density werden zufallsgeneriert. 

Die Dateien haben den Namen `name_0000_A.mtx`, `name_0001_A.mtx`, etc. Es werden zu jeder Nummer drei Matrizen erzeugt: A, B, C, wobei C = A*B.

Der Standard-Output-Ordner kann festgelegt werden mit `--basedir=matrix_instances/lieber/anderer/ordner`.

Wenn kein Produkt gewünscht ist, kann der `--noproduct`-Parameter angegeben werden.

### Dense Matrix Generation

`generate_matrices_sparse.py`: In Arbeit, bitte ignorieren.

