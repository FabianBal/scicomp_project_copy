# GPU Matrix Multiplication

Add to cargo via `gpu = { path = "../gpu" }`.


## Sparse


Import sparse part, then call. See example. DOES NOT WORK CORRECTLY FOR LARGE MATRICES! Try too keep them below 256 rows for now.

```rust
use gpu::gpu_sparse_multiplyer::GPUSparseMultiplyer;

use matrix_base::{COO, CSR};


#[tokio::main]
async fn main() {
    let batch_size = 18;  // Choose not large than 256


    let a = COO::read_mtx(Path::new("matrix_instances/generated/case_0000_A.mtx"), true).expect("Failed reading matrix file.");
    let a = CSR::from_coo(a); 
    let b = COO::read_mtx(Path::new("matrix_instances/generated/case_0000_B.mtx"), true).expect("Failed reading matrix file.");
    let b = CSR::from_coo(b);


    // Initialize
    let mut gpusm = GPUSparseMultiplyer::new(a, b, batch_size).await;

    // Load data to GPU (can time this)
    gpusm.create_and_load_buffer();

    // Calculate product (can time this)
    let mut res = gpusm.doit().await;
}
```


## Dense

Import dense stuff on demand

```rust
use gpu::dense::???

// Call respecitve functions...

```
