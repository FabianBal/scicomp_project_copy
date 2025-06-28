use bindgen;

fn main() {
    // Sicherstellen, dass der Pfad korrekt ist
    let cuda_path = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6".to_string();

    println!("Using CUDA path: {}", cuda_path);

    // Generiere die Bindings für cuSPARSE
    let bindings = bindgen::Builder::default()
        .header(format!("{}/include/cusparse.h", cuda_path))
        .clang_args(&[
            format!("-I{}/include", cuda_path),
            format!("-I{}/include/crt", cuda_path), // Falls notwendig
        ])
        .blocklist_item("FP_NAN")
        .blocklist_item("FP_INFINITE")
        .blocklist_item("FP_ZERO")
        .blocklist_item("FP_SUBNORMAL")
        .blocklist_item("FP_NORMAL")
        .blocklist_item("FP_INFINITE")
        .generate()
        .expect("Unable to generate bindings");

    // Bindings in Datei schreiben
    bindings
        .write_to_file("src/bindings.rs")
        .expect("Couldn't write bindings!");

    // Cargo anweisen, das Skript erneut auszuführen, falls sich die Header ändern
    println!("cargo:rerun-if-changed={}/include/cusparse.h", cuda_path);

    // CUDA-Bibliotheken verlinken
    println!("cargo:rustc-link-lib=cusparse"); // cuSPARSE
    println!("cargo:rustc-link-lib=cudart"); // CUDA Runtime

    // Den richtigen Library-Pfad setzen
    println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
}
