# FAKSGPU Scientific Computing Project

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

### Dense Matrix Generation

`generate_matrices_sparse.py`: In Arbeit, bitte ignorieren.

