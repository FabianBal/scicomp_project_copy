use super::{Dense, COO};


// CSR format from "Two Fast Algorithms for Sparse Matrices: Multiplication and Permuted Transposition", Rice, Gustavson
// https://dl.acm.org/doi/pdf/10.1145/355791.355796
// Notation relation with paper:
// row_pos = IA, col_pos = JA, values = A
pub struct CSR {
    pub row_pos: Vec<usize>,
    pub col_pos: Vec<usize>,
    pub values: Vec<f64>,
    pub shape: (usize, usize)
}


impl CSR {
    pub fn from_coo(coo: COO) -> Self {
        let mut row_pos: Vec<usize> = vec![];
        let mut col_pos: Vec<usize> = vec![];
        let mut values: Vec<f64> = vec![];

        let shape = coo.shape;


        // Go through all entries of the COO matrix.
        // Keep track of the current row in curr_row.
        // When i changes, we know we are done with the last row. Increment.
        // global_idx that is pushed into row_pos is the index of the
        // first entry of the corresponsing row (see paper on CSR).
        // Write the column-index and corresponding value of the current COO-entry
        // into col_pos and values, repsectively.
        let mut curr_row = 0;
        
        // Push 0-th row 
        row_pos.push(0);

        // Main loop
        for (global_idx, (i, j, x)) in coo.data.iter().enumerate() {
            if curr_row < *i {
                // If diff > 1, then zero rows are encountererd
                let diff = i - curr_row;
                curr_row = *i;

                // This ahdnles directly the case of (multiple) zero rows
                // If there is no zero row, only push global_idx once.
                for _ in 0..diff {
                    row_pos.push(global_idx);
                }                
            }

            col_pos.push(*j);
            values.push(*x);
        }
        // One more entry to mark the end of the last row.
        // This handles also the edge case if the there are zero-rows in the end
        let diff = shape.0 -curr_row;
        for _ in 0..diff {
            row_pos.push(values.len());
        }

        CSR{row_pos, col_pos, values, shape}
    }




    pub fn print(&self) {        
        println!("Sparse ({},{})-matrix in CSR format with {} entries", self.shape.0, self.shape.1, self.values.len());
        println!("Row Pos {:?}", self.row_pos);
        println!("Col Pos {:?}", self.col_pos);
        println!("Values {:?}", self.values);
    }



    pub fn get_row_nnz(&self, k: usize) -> usize {
        self.row_pos[k+1]-self.row_pos[k]
    }



    // Matrix/Matrix product, see seciton 3.2 from "A Systematic Survey of General Sparse Matrix-Matrix Multiplication", Gao et al.
    // https://doi.org/10.1145/3571157
    // Notation also from Paper
    // C = A*B
    // I_i(A) is the set of column indices of all non-zero entries of the i-th row of A
    // Returns dense matrix
    pub fn product(&self, other: &CSR) -> Dense {
        // let data = vec![];
        let m = self.shape.0;
        let n = other.shape.1;
        // let mut mat = vec![vec![0.;n];m];
        let mut mat = Dense::new_zeros((m,n));

        for i in 0..m {
            // iterate over all non-zero cols of A_{i*}
            // let cols = self.col_pos[i]..self.col_pos[i+1];
            for col_pos_pos in self.row_pos[i]..self.row_pos[i+1] {
                let k = self.col_pos[col_pos_pos];

                for other_col_pos_pos in other.row_pos[k]..other.row_pos[k+1] {
                    let j = other.col_pos[other_col_pos_pos];

                    // C_{i*} = \sum_{k \in I_i (A)} a_{ik} b_{i*}
                    // a_{ik} = self.values[col_pos_pos]
                    // b_{kj} = other.values[other_col_pos_pos]
                    // mat[i][j] += self.values[col_pos_pos] * other.values[other_col_pos_pos]
                    mat.set(i,j , mat.get(i, j) +   self.values[col_pos_pos] * other.values[other_col_pos_pos]);

                }
            }
        }

        mat
    }


    pub fn to_dense(&self) -> Dense {
        let m = self.shape.0;
        let n = self.shape.1;
        // let mut mat = vec![vec![0.;n];m];
        let mut mat = Dense::new_zeros((m,n));

        for i in 0..m {
            for col_pos_pos in self.row_pos[i]..self.row_pos[i+1] {
                let k = self.col_pos[col_pos_pos];
                mat.set(i,k, self.values[col_pos_pos]);
            }
        }

        mat
    }




    // For the general algorithm see above
    // This is a modification for directly saving CSR
    // via dense intermediate results, suited for
    // parallel execution
    pub fn product_sparse(&self, other: &CSR) -> CSR {
        let m = self.shape.0;
        let n = other.shape.1;
        // let mut mat = Dense::new_zeros((m,n));

        let mut res_rows: Vec<Vec<f64>> = vec![];
        let mut res_col_idxs: Vec<Vec<usize>> = vec![];

        for i in 0..m {
            // Iterate over all non-zero cols of A_{i*}

            // Create a dense row for the result matrix, C_{i*}
            // and also a bool array that flags if some non-zero
            // entry for the k-th (i.e. for C_{ik}) is calculated.            
            let mut nz_row_marker = vec![false;n];
            let mut res_curr_row = vec![0.;n];


            for col_pos_pos in self.row_pos[i]..self.row_pos[i+1] {
                let k = self.col_pos[col_pos_pos];

                for other_col_pos_pos in other.row_pos[k]..other.row_pos[k+1] {
                    let j = other.col_pos[other_col_pos_pos];

                    // C_{i*} = \sum_{k \in I_i (A)} a_{ik} b_{i*}
                    // a_{ik} = self.values[col_pos_pos]
                    // b_{kj} = other.values[other_col_pos_pos]
                    // mat[i][j] += self.values[col_pos_pos] * other.values[other_col_pos_pos]
                    // mat.set(i,j , mat.get(i, j) +   self.values[col_pos_pos] * other.values[other_col_pos_pos]);
                    res_curr_row[j] += self.values[col_pos_pos] * other.values[other_col_pos_pos];
                    nz_row_marker[j] = true;                    

                }
            }

            // Go through the row, which may contain 0 entries.
            // If !marker, then the k-col is 0 and can be ignored
            // Save only non-zero entries and their index
            let mut res_curr_row_final_val = vec![];
            let mut res_curr_row_final_col_idx = vec![];
            for ((k, x), marker) in res_curr_row.iter().enumerate().into_iter().zip(nz_row_marker) {
                if marker {
                    res_curr_row_final_val.push(*x);
                    res_curr_row_final_col_idx.push(k);
                }
            }
            
            // Push the final result for the current row
            res_rows.push(res_curr_row_final_val);
            res_col_idxs.push(res_curr_row_final_col_idx);
        }


        // Create the row_pos vector and flatten the other
        let mut row_pos_counter = 0;
        let mut row_pos = vec![0];

        for c in &res_col_idxs {
            row_pos_counter += c.len();
            row_pos.push(row_pos_counter);
        }

        let col_pos = res_col_idxs.concat();
        let values = res_rows.concat();

        row_pos.push(values.len());

        CSR{row_pos, col_pos, values, shape: (m,n)}

    }


}