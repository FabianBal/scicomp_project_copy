# Crate matrix_base

Provides base types for matrices: Dense and COO (sparse).

## How to use

### Cargo.toml
Under `[dependencies]`, add `matrix_base = { path = "../matrix_base" }`.

### Code example

```rust
use matrix_base::{Dense, COO};

fn main() {
    // Read MTX file into sparse COO format
    // coo is of type COO
    let coo = COO::read_mtx("path/to/matrix.mtx", false).expect("Failed reading file");

    // Create dense matrix from COO matrix
    // dense is of type Dense
    let dense = coo.to_dense();
}
```
