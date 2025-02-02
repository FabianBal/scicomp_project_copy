use bindgen;

fn main() {
    //adjust path to CUDA installation!
    let cuda_path = "/usr/local/cuda-12.8";
    // Generate bindings to CUDA's C API because no cuSPARSE rust wrapper exists as in cuBLAS
    let bindings = bindgen::Builder::default()
        .header(format!("{}/include/cusparse.h", cuda_path))
        .blocklist_item("FP_NAN")
        .blocklist_item("FP_INFINITE")
        .blocklist_item("FP_ZERO")
        .blocklist_item("FP_SUBNORMAL")
        .blocklist_item("FP_NORMAL")
        .blocklist_item("FP_INFINITE")
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to a file
    bindings
        .write_to_file("src/bindings.rs")
        .expect("Couldn't write bindings!");

    // Tell Cargo to rerun this script if the CUDA header changes
    println!("cargo:rerun-if-changed={}/include/cusparse.h", cuda_path);

    // Tell Cargo to link to CUDA libraries
    println!("cargo:rustc-link-lib=cusparse"); // Link against cusparse
    println!("cargo:rustc-link-lib=cudart");   // Link against CUDA runtime library

    // Specify the path to the CUDA libraries
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);  // Modify this if your CUDA is in a different location
}