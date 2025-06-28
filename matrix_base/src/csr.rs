use crate::{Dense, COO};

// CSR format from "Two Fast Algorithms for Sparse Matrices: Multiplication and Permuted Transposition", Rice, Gustavson
// https://dl.acm.org/doi/pdf/10.1145/355791.355796
// Notation relation with paper:
// row_pos = IA, col_pos = JA, values = A
pub struct CSR {
    pub row_pos: Vec<usize>,
    pub col_pos: Vec<usize>,
    pub values: Vec<f64>,
    pub shape: (usize, usize),
}

impl CSR {
    pub fn from_coo(coo: &COO) -> Self {
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
        let diff = shape.0 - curr_row;
        for _ in 0..diff {
            row_pos.push(values.len());
        }

        CSR {
            row_pos,
            col_pos,
            values,
            shape,
        }
    }

    pub fn print(&self) {
        println!(
            "Sparse ({},{})-matrix in CSR format with {} entries",
            self.shape.0,
            self.shape.1,
            self.values.len()
        );
        println!("Row Pos {:?}", self.row_pos);
        println!("Col Pos {:?}", self.col_pos);
        println!("Values {:?}", self.values);
    }

    pub fn get_row_nnz(&self, k: usize) -> usize {
        self.row_pos[k + 1] - self.row_pos[k]
    }

    pub fn to_dense(&self) -> Dense {
        let m = self.shape.0;
        let n = self.shape.1;
        // let mut mat = vec![vec![0.;n];m];
        let mut mat = Dense::new_zeros((m, n));

        for i in 0..m {
            for col_pos_pos in self.row_pos[i]..self.row_pos[i + 1] {
                let k = self.col_pos[col_pos_pos];
                mat.set(i, k, self.values[col_pos_pos]);
            }
        }

        mat
    }

    pub fn to_coo(&self) -> COO {
        let mut data: Vec<(usize, usize, f64)> = Vec::with_capacity(self.values.len()); // Kapazität ist NNZ

        // Gehe jede Zeile der CSR-Matrix durch
        for i in 0..self.shape.0 {
            // row_pos[i] ist der Startindex der Nicht-Null-Elemente für die aktuelle Zeile i
            // row_pos[i+1] ist der Endindex
            let start_idx = self.row_pos[i];
            let end_idx = self.row_pos[i + 1];

            // Iteriere über die Nicht-Null-Elemente dieser Zeile
            for k in start_idx..end_idx {
                let col_idx = self.col_pos[k]; // Spaltenindex des Elements
                let value = self.values[k];   // Wert des Elements
                data.push((i, col_idx, value)); // Füge als (Reihe, Spalte, Wert)-Tupel hinzu
            }
        }

        COO {
            data,
            shape: self.shape,
        }
    }
}
