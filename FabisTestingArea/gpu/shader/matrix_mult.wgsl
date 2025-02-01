@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> matrix_a_size: vec2<u32>;
@group(0) @binding(4) var<uniform> matrix_b_size: vec2<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.y;
    let col = id.x;

    if (row < matrix_a_size.x && col < matrix_b_size.y) {
        var sum: f32 = 0.0;
        for (var k = 0u; k < matrix_a_size.y; k = k + 1u) {
            sum = sum + a[row * matrix_a_size.y + k] * b[k * matrix_b_size.y + col];
        }
        c[row * matrix_b_size.y + col] = sum;
    }
}