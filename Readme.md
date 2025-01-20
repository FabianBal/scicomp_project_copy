# FAKSGPU Scientific Computing Project

## Struktur

### Code

`cpu`: Vergleichsimplementierung: pur CPU
`blas`: Vergleichsimplementierung: cudaBLAS und CPU BLAS
`gpu`: Implementierung auf GPU

### Daten

`matrix_instances`: Matrizen im MTX-Format, eher kleine Dateien
`matrix_instances/generated`: Matrizen im MTX-Format, eher größere. Nicht unter Versionskontrolle

### Skripte

`generate_matrices.py`: Generiert zufällige Matrizen zum Test und schreibt sie als MTX-Datei aus

