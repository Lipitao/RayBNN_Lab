extern crate arrayfire;

use crate::graph::large_sparse_i32::integer_histogram;

pub fn input_degree(neuron_size: u64, WRowIdxCOO: &arrayfire::Array<i32>) -> arrayfire::Array<u32> {
    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let degree_dims = arrayfire::Dim4::new(&[neuron_size, 1, 1, 1]);
    let mut degree = arrayfire::constant::<u32>(0, degree_dims);

    let mut bins = arrayfire::constant::<i32>(0, temp_dims);
    let mut counts = arrayfire::constant::<u32>(0, temp_dims);

    integer_histogram(WRowIdxCOO, &mut bins, &mut counts);

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&bins, 0, None);
    arrayfire::assign_gen(&mut degree, &idxrs, &counts);

    degree
}

pub fn output_degree(neuron_size: u64, WColIdx: &arrayfire::Array<i32>) -> arrayfire::Array<u32> {
    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let degree_dims = arrayfire::Dim4::new(&[neuron_size, 1, 1, 1]);
    let mut degree = arrayfire::constant::<u32>(0, degree_dims);

    let mut bins = arrayfire::constant::<i32>(0, temp_dims);
    let mut counts = arrayfire::constant::<u32>(0, temp_dims);

    integer_histogram(WColIdx, &mut bins, &mut counts);

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&bins, 0, None);
    arrayfire::assign_gen(&mut degree, &idxrs, &counts);

    degree
}
