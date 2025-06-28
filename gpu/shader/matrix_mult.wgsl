@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec3<u32>; // x=matrix_a_rows, y=matrix_a_cols, z=matrix_b_cols

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.y;
    let col = id.x;

    // Verwende die Dimensionen aus dem `dims`-Buffer
    let matrix_a_rows = dims.x;
    let matrix_a_cols = dims.y;
    let matrix_b_cols = dims.z; // Die dritte Dimension des Uniform-Buffers

    if (row < matrix_a_rows && col < matrix_b_cols) { // matrix_b_size.y wird zu matrix_b_cols
        var sum: f32 = 0.0;
        for (var k = 0u; k < matrix_a_cols; k = k + 1u) { // matrix_a_size.y wird zu matrix_a_cols
            sum = sum + a[row * matrix_a_cols + k] * b[k * matrix_b_cols + col]; // matrix_a_size.y & matrix_b_size.y
        }
        c[row * matrix_b_cols + col] = sum; // matrix_b_size.y
    }
}