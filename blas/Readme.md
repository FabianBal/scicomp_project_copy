# blas\_dense crate

Crate to multiply two matrices given in COO format using the dgemm BLAS function

## How to use
install libopenblas-dev
```bash
sudo apt-get install libopenblas-dev
```

### Cargo.toml
Under `[dependencies]`, add `blas_dense = { path = "../blas" }`.

### The library

The library contains the struct `BlasDense` that stores the data and shape of a matrix.  
**IMPORTANT NOTE:** other than in `Dense` the matrix is flattened in **column-major order**!! E.g. the matrix [[1,2],[3,4]] is stored as [1,3,2,4]  
Below a code example, which demonstrates the functionality of this crate

```rust
use blas_dense::*;
use matrix_base::COO;
use std::path::Path;

fn main() {
    let sp_a = COO::read_mtx(Path::new("../matrix_instances/a_blas.mtx"), false)
        .expect("Failed reading file");
    let sp_b = COO::read_mtx(Path::new("../matrix_instances/b_blas.mtx"), false).expect("Failed reading file");
    let a = BlasDense::from_coo(&sp_a);
    let b = BlasDense::from_coo(&sp_b);
    let result = a.prod(&b);
}
