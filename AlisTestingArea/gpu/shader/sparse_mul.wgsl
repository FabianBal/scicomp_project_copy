// struct CSR {
//     row_pos: array<u32>,
//     col_pos: array<u32>,
//     values: array<f32>,
//     shape: vec2<u32>
// }


// @group(0) @binding(0) var<storage, read> a: CSR;

@group(0) @binding(0) var<storage, read> a_row_pos: array<u32>;
@group(0) @binding(1) var<storage, read> a_col_pos: array<u32>;
@group(0) @binding(2) var<storage, read> a_values: array<f32>;
@group(0) @binding(3) var<storage, read> a_shape: vec2<u32>;


@group(1) @binding(0) var<storage, read> b_row_pos: array<u32>;
@group(1) @binding(1) var<storage, read> b_col_pos: array<u32>;
@group(1) @binding(2) var<storage, read> b_values: array<f32>;
@group(1) @binding(3) var<storage, read> b_shape: vec2<u32>;

@group(2) @binding(0) var<storage, read_write> c_row_pos: array<u32>;
@group(2) @binding(1) var<storage, read_write> c_col_pos: array<u32>;
@group(2) @binding(2) var<storage, read_write> c_values: array<f32>;
@group(2) @binding(3) var<storage, read_write> c_shape: vec2<u32>;


// @group(0) @binding(1) var<storage, read> b: array<f32>;
// @group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // let N: u32 = 4u; // Matrixgröße
    // let row = id.y;
    // let col = id.x;

    // if (row < N && col < N) {
    //     var sum: f32 = 0.0;
    //     for (var k = 0u; k < N; k = k + 1u) {
    //         sum = sum + a[row * N + k] * b[k * N + col];
    //     }
    //     c[row * N + col] = sum;
    // }

    c_shape.x = 20;
    c_shape.y = 20;
}