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

@compute @workgroup_size(18,1,1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = a_shape.x;
    let m = b_shape.y;

    let i = id.x;

    let aa: array<u32> = [1,2,3];

    // for (col_pos_pos = a_row_pos[i]; i < a_row_pos[i+1]; i++) {
    //     let k = a_col_pos[col_pos_pos];

    //     for (other_col_pos_pos = b_row_pos[k]; other_col_pos_pos < b_row_pos[k+1]; other_col_pos_pos++) {
    //         let j = b_col_pos[other_col_pos_pos];


    //     }

    // }



    c_shape.x = 20;
    c_shape.y = 20;
}


// for col_pos_pos in self.row_pos[i]..self.row_pos[i+1] {
//     let k = self.col_pos[col_pos_pos];

//     for other_col_pos_pos in other.row_pos[k]..other.row_pos[k+1] {
//         let j = other.col_pos[other_col_pos_pos];

//         // C_{i*} = \sum_{k \in I_i (A)} a_{ik} b_{i*}
//         // a_{ik} = self.values[col_pos_pos]
//         // b_{kj} = other.values[other_col_pos_pos]
//         // mat[i][j] += self.values[col_pos_pos] * other.values[other_col_pos_pos]
//         mat.set(i,j , mat.get(i, j) +   self.values[col_pos_pos] * other.values[other_col_pos_pos]);

//     }
// }

// if (row < N && col < N) {
//         var sum: f32 = 0.0;
//         for (var k = 0u; k < N; k = k + 1u) {
//             sum = sum + a[row * N + k] * b[k * N + col];
//         }
//         c[row * N + col] = sum;
//     }