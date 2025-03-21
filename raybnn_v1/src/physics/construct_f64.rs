extern crate arrayfire;

use crate::physics::distance_f64::vec_norm;

const two: f64 = 2.0;
const one: f64 = 1.0;

const epsilon: f64 = 0.0001;

pub fn get_outside_idx_cube(pos: &arrayfire::Array<f64>, cube_size: f64) -> arrayfire::Array<u32> {
    let neg_cube_size = -cube_size;
    //cube_size < pos   or    pos < neg_cube_size
    let mut cmp1 = arrayfire::lt(&cube_size, pos, false);
    let mut cmp2 = arrayfire::lt(pos, &neg_cube_size, false);
    cmp1 = arrayfire::or(&cmp1, &cmp2, false);
    cmp1 = arrayfire::any_true(&cmp1, 1);

    arrayfire::locate(&cmp1)
}

pub fn get_inside_idx_cube(pos: &arrayfire::Array<f64>, cube_size: f64) -> arrayfire::Array<u32> {
    let neg_cube_size = -cube_size;
    //  neg_cube_size  <  pos <  cube_size
    let mut cmp1 = arrayfire::lt(pos, &cube_size, false);
    let mut cmp2 = arrayfire::lt(&neg_cube_size, pos, false);
    cmp1 = arrayfire::and(&cmp1, &cmp2, false);
    cmp1 = arrayfire::all_true(&cmp1, 1);

    arrayfire::locate(&cmp1)
}

pub fn replicate_struct(
    position: &arrayfire::Array<f64>,
    direction: &arrayfire::Array<f64>,
    num: u64,
) -> arrayfire::Array<f64> {
    let repeat_dims = arrayfire::Dim4::new(&[position.dims()[0], 1, 1, 1]);
    let tile_dims3 = arrayfire::Dim4::new(&[1, num, 1, 1]);

    let mut count = arrayfire::iota::<f64>(tile_dims3, repeat_dims);

    count = arrayfire::flat(&count);

    let tile_dims = arrayfire::Dim4::new(&[num, 1, 1, 1]);

    let mut out_position = arrayfire::tile(position, tile_dims);

    let tile_dims2 = arrayfire::Dim4::new(&[out_position.dims()[0], 1, 1, 1]);

    let mut dir_tile = arrayfire::tile(direction, tile_dims2);

    dir_tile = arrayfire::mul(&dir_tile, &count, true);

    out_position = dir_tile + out_position;

    out_position
}

pub fn NDsphere_from_NDcube(
    cube_pos: &arrayfire::Array<f64>,
    cube_radius: f64,

    sphere_radius: f64,
) -> arrayfire::Array<f64> {
    let mut sphere_pos = cube_pos.clone();

    let space_dims = cube_pos.dims()[1];

    let dir_template: Vec<f64> = vec![0.0; space_dims as usize];

    let repnum = (sphere_radius / cube_radius).ceil() as u64;

    let dir_dims = arrayfire::Dim4::new(&[1, space_dims, 1, 1]);

    for i in 0..(space_dims as usize) {
        let mut direction_cpu: Vec<f64> = dir_template.clone();
        direction_cpu[i] = two * cube_radius;

        let direction = arrayfire::Array::new(&direction_cpu, dir_dims);

        sphere_pos = replicate_struct(&sphere_pos, &direction, repnum);
    }

    let shift_cpu: Vec<f64> = vec![-((repnum as f64) - one) * cube_radius; space_dims as usize];
    let shift = arrayfire::Array::new(&shift_cpu, dir_dims);

    sphere_pos = arrayfire::add(&shift, &sphere_pos, true);

    let eps = epsilon * arrayfire::randn::<f64>(sphere_pos.dims());
    sphere_pos = sphere_pos + eps;

    let sq = vec_norm(&sphere_pos);

    let cmp1 = arrayfire::lt(&sq, &sphere_radius, false);

    let idx1 = arrayfire::locate(&cmp1);

    let mut idxrs1 = arrayfire::Indexer::default();
    let seq1 = arrayfire::Seq::new(0.0, (space_dims - 1) as f64, 1.0);
    idxrs1.set_index(&idx1, 0, None);
    idxrs1.set_index(&seq1, 1, Some(false));
    sphere_pos = arrayfire::index_gen(&sphere_pos, idxrs1);

    sphere_pos
}

pub fn plane_surface_on_NDsphere(
    plane_shape: &Vec<u64>,

    sphere_radius: f64,
) -> arrayfire::Array<f64> {
    let space_dims = (plane_shape.len() + 1) as u64;

    let mut cur_dim = 0;
    let mut cur_size = plane_shape[cur_dim];

    let plane_dims = arrayfire::Dim4::new(&[cur_size, space_dims, 1, 1]);
    let mut plane = arrayfire::constant::<f64>(0.0, plane_dims);

    let single = arrayfire::Dim4::new(&[1, 1, 1, 1]);
    let across_dims = arrayfire::Dim4::new(&[cur_size, 1, 1, 1]);

    let mut newplane = arrayfire::iota::<f64>(across_dims, single);

    newplane = newplane * ((two * sphere_radius) / (cur_size as f64));

    arrayfire::set_col(&mut plane, &newplane, 0);

    let dir_template: Vec<f64> = vec![0.0; space_dims as usize];

    let dir_dims = arrayfire::Dim4::new(&[1, space_dims, 1, 1]);

    cur_dim = cur_dim + 1;

    while cur_dim < ((space_dims - 1) as usize) {
        cur_size = plane_shape[cur_dim];

        let mut direction_cpu: Vec<f64> = dir_template.clone();
        direction_cpu[cur_dim] = ((two * sphere_radius) / (cur_size as f64));

        let direction = arrayfire::Array::new(&direction_cpu, dir_dims);

        plane = replicate_struct(&plane, &direction, cur_size);

        cur_dim = cur_dim + 1;
    }

    let shift_cpu: Vec<f64> = vec![-sphere_radius; space_dims as usize];
    let shift = arrayfire::Array::new(&shift_cpu, dir_dims);

    plane = arrayfire::add(&shift, &plane, true);

    let magsq = vec_norm(&plane);

    plane = sphere_radius * arrayfire::div(&plane, &magsq, true);

    plane
}
