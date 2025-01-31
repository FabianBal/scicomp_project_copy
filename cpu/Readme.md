# CPU crate

Crate for matrix multiplication on CPU, currently only defined on CSR matrices

## How to use

### Cargo.toml
Under `[dependencies]`, add `fakscpu = { path = "../cpu" }`.

**IMPORTANT NOTE:** The crate name is `fakscpu`, while the directory name is only `cpu`!

### The library

Below a code example, which demonstrates the functionality of this crate

```rust
use matrix_base::{Dense, COO};

fn main() {
    // Load MTX file into COO format
    // IMPORTANT: Set sorting flag to true!
    let A = COO::read_mtx("../matrix_instances/generated/case_0000_A.mtx", true).expect("Failed reading file");
    let B = COO::read_mtx("../matrix_instances/generated/case_0000_B.mtx", true).expect("Failed reading file");

    // Create CSR matrix from COO matrix
    let A = CSR::from_coo(A);
    let B = CSR::from_coo(B);

    // Calculate product. Three different ways
    // C1 is of type Dense. C2 and C3 are CSR
    // C1 and C2 are calculated serially. C3 is calculated parallely.
    let C1 = A.product(&B);
    let C2 = A.product_sparse(&B);
    let C3 = A.product_sparse_par(&B);
}
```
