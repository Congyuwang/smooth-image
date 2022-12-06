use nalgebra::DVector;
use nalgebra_sparse::coo::CooMatrix;
use nalgebra_sparse::CsrMatrix;

pub fn psnr(inferred: &DVector<f32>, original: &DVector<f32>) -> f32 {
    let dist = inferred.metric_distance(original);
    (inferred.nrows() as f32 / (dist * dist)).log10()
}

/// build the selection matrix A, and target vector b
pub fn matrix_a(mask: &[u8], img: &[u8]) -> (CsrMatrix<f32>, DVector<f32>) {
    let undamaged = mask.iter().map(|m| usize::from(*m != 0)).sum();
    let mut coo = CooMatrix::new(undamaged, mask.len());
    let mut vector_b = vec![0.0f32; undamaged];
    mask.iter()
        .zip(img.iter())
        .enumerate()
        .filter(|(_, (m, _))| **m != 0)
        .zip(vector_b.iter_mut())
        .enumerate()
        .for_each(|(select_index, ((px_index, (_, p)), b))| {
            *b = *p as f32 / 256.0;
            coo.push(select_index, px_index, 1.0f32);
        });
    (CsrMatrix::from(&coo), DVector::from_vec(vector_b))
}

/// build the difference matrix D
///
/// pixel layout:
/// `row0:[0...width], row1:[0.. width]`
pub fn matrix_d(width: usize, height: usize) -> CsrMatrix<f32> {
    let px = height * width;
    let vertical_entries = px - width;
    let horizontal_entries = px - height;
    let total_entries = (vertical_entries + horizontal_entries) * 2;

    let mut row_indices = vec![0usize; total_entries];
    let mut col_indices = vec![0usize; total_entries];
    let mut values = vec![0.0f32; total_entries];

    // vertical -1
    (0..vertical_entries)
        .zip(row_indices[..vertical_entries].iter_mut())
        .zip(col_indices[..vertical_entries].iter_mut())
        .zip(values[..vertical_entries].iter_mut())
        .for_each(|(((i, r), c), v)| {
            *r = i;
            *c = i;
            *v = -1.0;
        });
    let vert_end = vertical_entries * 2;
    // vertical 1
    (0..vertical_entries)
        .zip(row_indices[vertical_entries..vert_end].iter_mut())
        .zip(col_indices[vertical_entries..vert_end].iter_mut())
        .zip(values[vertical_entries..vert_end].iter_mut())
        .for_each(|(((i, r), c), v)| {
            *r = i;
            *c = i + width;
            *v = 1.0;
        });
    // indices generator
    let h_first_end = vert_end + horizontal_entries;
    // horizontal -1
    (0..px)
        .filter(|i| (i + 1) % width != 0)
        .zip(row_indices[vert_end..h_first_end].iter_mut())
        .zip(col_indices[vert_end..h_first_end].iter_mut())
        .zip(values[vert_end..h_first_end].iter_mut())
        .for_each(|(((i, r), c), v)| {
            *r = i + px;
            *c = i;
            *v = -1.0;
        });
    // horizontal 1
    (0..px)
        .filter(|i| (i + 1) % width != 0)
        .zip(row_indices[h_first_end..].iter_mut())
        .zip(col_indices[h_first_end..].iter_mut())
        .zip(values[h_first_end..].iter_mut())
        .for_each(|(((i, r), c), v)| {
            *r = i + px;
            *c = i + 1;
            *v = 1.0;
        });

    let coo = CooMatrix::try_from_triplets(2 * px, px, row_indices, col_indices, values).unwrap();

    CsrMatrix::from(&coo)
}

#[cfg(test)]
mod test_matrix {
    use crate::opt_utils::matrix_d;
    use nalgebra::DMatrix;

    #[test]
    fn test_matrix_d_2_5() {
        let mat_2_5 = matrix_d(2, 5);
        let computed = DMatrix::from(&mat_2_5);
        let expected = [
            [-1f32, 0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., -1., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., -1., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., -1., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., -1., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., -1., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., -1., 0., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., -1., 0., 1.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [-1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., -1., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., -1., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., -1., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., -1., 1.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        ];
        let flatten = expected.into_iter().flatten().collect::<Vec<_>>();
        let expected_mat = DMatrix::from_row_slice(20, 10, &flatten);
        assert_eq!(expected_mat, computed);
    }

    #[test]
    fn test_matrix_d_6_3() {
        let mat_6_3 = matrix_d(6, 3);
        let computed = DMatrix::from(&mat_6_3);
        let expected = [
            [
                -1f32, 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
            [
                0., -1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
            [
                0., 0., -1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
            [
                0., 0., 0., -1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
            [
                0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
            ],
            [
                0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
            ],
            [
                0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
            ],
            [
                0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
            ],
            [
                0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
            ],
            [
                0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 1., 0., 0.,
            ],
            [
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 1., 0.,
            ],
            [
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 1.,
            ],
            [
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
            [
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
            [
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
            [
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
            [
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
            [
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
            [
                -1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
            [
                0., -1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
            [
                0., 0., -1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
            [
                0., 0., 0., -1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
            [
                0., 0., 0., 0., -1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
            [
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
            [
                0., 0., 0., 0., 0., 0., -1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
            [
                0., 0., 0., 0., 0., 0., 0., -1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
            [
                0., 0., 0., 0., 0., 0., 0., 0., -1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
            [
                0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 1., 0., 0., 0., 0., 0., 0., 0.,
            ],
            [
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 1., 0., 0., 0., 0., 0., 0.,
            ],
            [
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
            [
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 1., 0., 0., 0., 0.,
            ],
            [
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 1., 0., 0., 0.,
            ],
            [
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 1., 0., 0.,
            ],
            [
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 1., 0.,
            ],
            [
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 1.,
            ],
            [
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
        ];
        let flatten = expected.into_iter().flatten().collect::<Vec<_>>();
        let expected_mat = DMatrix::from_row_slice(36, 18, &flatten);
        assert_eq!(expected_mat, computed);
    }
}
