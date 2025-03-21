extern crate arrayfire;

const two: f64 = 2.0;

pub fn vec_norm(vector: &arrayfire::Array<f64>) -> arrayfire::Array<f64> {
    let mut sq = arrayfire::pow(vector, &two, false);
    sq = arrayfire::sum(&sq, 1);

    arrayfire::sqrt(&sq)
}

pub fn vec_dist(
    target_pos: &arrayfire::Array<f64>,
    positions: &arrayfire::Array<f64>,
    dist: &mut arrayfire::Array<f64>,
    magsq: &mut arrayfire::Array<f64>,
) {
    //let tile_dims = arrayfire::Dim4::new(&[positions.dims()[0],1,1,1]);

    //*dist = positions.clone() - arrayfire::tile(target_pos, tile_dims);

    *dist = arrayfire::sub(positions, target_pos, true);

    *magsq = arrayfire::pow(dist, &two, false);

    *magsq = arrayfire::sum(magsq, 1);
}

pub fn vec_min_dist(target_pos: &arrayfire::Array<f64>, positions: &arrayfire::Array<f64>) -> f64 {
    let pos_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let mut dist = arrayfire::constant::<f64>(0.0, pos_dims);
    let mut magsq = arrayfire::constant::<f64>(0.0, pos_dims);
    vec_dist(target_pos, positions, &mut dist, &mut magsq);
    let (m0, _) = arrayfire::min_all::<f64>(&magsq);
    m0
}

pub fn set_diag(magsq_matrix: &mut arrayfire::Array<f64>, val: f64) {
    let pos_num = magsq_matrix.dims()[0];

    let magsq_dims = magsq_matrix.dims();

    let N_dims = arrayfire::Dim4::new(&[pos_num, 1, 1, 1]);
    let repeat_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let idx = (pos_num + 1) * arrayfire::iota::<i32>(N_dims, repeat_dims);
    *magsq_matrix = arrayfire::flat(magsq_matrix);

    let large_vec = arrayfire::constant::<f64>(val, arrayfire::Dim4::new(&[pos_num, 1, 1, 1]));

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&idx, 0, None);
    arrayfire::assign_gen(magsq_matrix, &idxrs, &large_vec);

    *magsq_matrix = arrayfire::moddims(magsq_matrix, magsq_dims);
}

pub fn matrix_dist(
    pos_vec: &arrayfire::Array<f64>,
    dist_matrix: &mut arrayfire::Array<f64>,
    magsq_matrix: &mut arrayfire::Array<f64>,
) {
    //let pos_num = pos_vec.dims()[0];
    //let space_dims = pos_vec.dims()[1];

    //let mut p0 = pos_vec.clone();

    //p0 = arrayfire::tile(&p0, arrayfire::Dim4::new(&[1,1,pos_num,1]));

    let mut p1 = pos_vec.clone();
    /*
    p1 = arrayfire::transpose(&p1, false);

    p1 = arrayfire::moddims(&p1,arrayfire::Dim4::new(&[1,space_dims,pos_num,1]));

    p1 = arrayfire::tile(&p1, arrayfire::Dim4::new(&[pos_num,1,1,1]));
    */

    p1 = arrayfire::reorder_v2(&p1, 2, 1, Some(vec![0]));

    //p1 = arrayfire::tile(&p1, arrayfire::Dim4::new(&[pos_num,1,1,1]));

    //*dist_matrix = p1 - p0;
    *dist_matrix = arrayfire::sub(&p1, pos_vec, true);

    *magsq_matrix = arrayfire::pow(dist_matrix, &two, false);

    *magsq_matrix = arrayfire::sum(magsq_matrix, 1);
}

pub fn sort_neuron_pos_sphere(neuron_pos: &mut arrayfire::Array<f64>) {
    let space_dims = neuron_pos.dims()[1];

    let pos_dims = arrayfire::Dim4::new(&[1, space_dims, 1, 1]);

    let mut magsq = arrayfire::pow(neuron_pos, &two, false);
    magsq = arrayfire::sum(&magsq, 1);

    let (_, idx1) = arrayfire::sort_index(&magsq, 0, false);

    let mut idxrs1 = arrayfire::Indexer::default();
    let seq1 = arrayfire::Seq::new(0.0, (space_dims - 1) as f64, 1.0);
    idxrs1.set_index(&idx1, 0, None);
    idxrs1.set_index(&seq1, 1, Some(false));
    *neuron_pos = arrayfire::index_gen(neuron_pos, idxrs1);
}

pub fn sort_neuron_pos_line(neuron_pos: &mut arrayfire::Array<f64>, axis: u64) {
    let col = arrayfire::col(neuron_pos, axis as i64);
    let space_dims = neuron_pos.dims()[1];

    let (_, idx1) = arrayfire::sort_index(&col, 0, true);

    let mut idxrs1 = arrayfire::Indexer::default();
    let seq1 = arrayfire::Seq::new(0.0, (space_dims - 1) as f64, 1.0);
    idxrs1.set_index(&idx1, 0, None);
    idxrs1.set_index(&seq1, 1, Some(false));
    *neuron_pos = arrayfire::index_gen(neuron_pos, idxrs1);
}
