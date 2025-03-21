extern crate arrayfire;
use crate::neural::network_f64::network_metadata_type;

use crate::physics::distance_f64::vec_min_dist;

use crate::graph::adjacency_f32::get_global_weight_idx;

use crate::physics::distance_f64::matrix_dist;
use crate::physics::distance_f64::set_diag;

use super::distance_f64::vec_norm;

use crate::graph::adjacency_f64::clear_input;

use crate::graph::adjacency_f64::clear_output;

use crate::graph::tree_i32::check_connected2;

use crate::graph::large_sparse_u64::COO_batch_find;

use crate::graph::large_sparse_f64::parallel_lookup;

use crate::graph::tree_u32::parallel_lookup as parallel_lookup_u32;

const COO_find_limit: u64 = 1500000000;

const high: f64 = 10000000.0;

const neuron_rad_factor: f64 = 1.1;

const two: f64 = 2.0;
const one: f64 = 1.0;
const onehalf: f64 = 0.5;
const zero: u32 = 0;
const NEURON_GEN_SIZE: u64 = 10000;
const TARGET_DENSITY: f64 = 3500.0;

/*
Creates input neurons on the surface of a sphere for 2D images of size (Nx,Ny)

Inputs
sphere_rad:   3D Sphere Radius
Nx:           Image X dimension size
Ny:           Image Y dimension size

Outputs:
The 3D position of neurons on the surface of a 3D sphere

*/

pub fn create_spaced_input_neuron_on_sphere(
    sphere_rad: f64,
    Nx: u64,
    Ny: u64,
) -> arrayfire::Array<f64> {
    let gen_dims = arrayfire::Dim4::new(&[1, Nx, 1, 1]);
    let rep_dims = arrayfire::Dim4::new(&[Ny, 1, 1, 1]);

    let mut theta = arrayfire::iota::<f64>(gen_dims, rep_dims) + one;
    theta = theta / ((Nx + 1) as f64);

    theta = two * (theta - onehalf);
    theta = arrayfire::acos(&theta);

    let gen_dims = arrayfire::Dim4::new(&[Ny, 1, 1, 1]);
    let rep_dims = arrayfire::Dim4::new(&[1, Nx, 1, 1]);

    let mut phi = arrayfire::iota::<f64>(gen_dims, rep_dims) + one;
    phi = phi / ((Ny + 1) as f64);

    phi = phi * two * std::f64::consts::PI;

    let mut x = sphere_rad * arrayfire::sin(&theta) * arrayfire::cos(&phi);
    let mut y = sphere_rad * arrayfire::sin(&theta) * arrayfire::sin(&phi);
    let mut z = sphere_rad * arrayfire::cos(&theta);

    x = arrayfire::flat(&x);
    y = arrayfire::flat(&y);
    z = arrayfire::flat(&z);

    arrayfire::join_many(1, vec![&x, &y, &z])
}

/*
Creates input neurons on the surface of a sphere for 1D data with random neuron position assignment


Inputs
sphere_rad:   3D Sphere Radius
input_size:   Number of input neurons


Outputs:
The 3D position of neurons on the surface of a 3D sphere

*/

pub fn create_spaced_input_neuron_on_sphere_1D(
    sphere_rad: f64,
    input_size: u64,
) -> arrayfire::Array<f64> {
    let sqrt_input = (input_size as f64).sqrt().ceil() as u64;

    let mut input_neurons =
        create_spaced_input_neuron_on_sphere(sphere_rad, sqrt_input, sqrt_input);

    let space_dims = input_neurons.dims()[1];

    let randarr_dims = arrayfire::Dim4::new(&[sqrt_input * sqrt_input, 1, 1, 1]);
    let randarr = arrayfire::randu::<f64>(randarr_dims);
    let (_, randidx) = arrayfire::sort_index(&randarr, 0, false);

    let mut idxrs1 = arrayfire::Indexer::default();
    let seq1 = arrayfire::Seq::new(0.0, (space_dims - 1) as f64, 1.0);
    idxrs1.set_index(&randidx, 0, None);
    idxrs1.set_index(&seq1, 1, Some(false));
    input_neurons = arrayfire::index_gen(&input_neurons, idxrs1);

    if input_neurons.dims()[0] > input_size {
        input_neurons = arrayfire::rows(&input_neurons, 0, (input_size - 1) as i64);
    }

    input_neurons
}

pub fn circle_2D(new_circle_rad: f64, add_num: u64) -> arrayfire::Array<f64> {
    let single = arrayfire::Dim4::new(&[1, 1, 1, 1]);
    let across_dims = arrayfire::Dim4::new(&[add_num, 1, 1, 1]);

    let mut t = arrayfire::iota::<f64>(across_dims, single);

    t = t * ((two * std::f64::consts::PI) / (add_num as f64));

    let x = new_circle_rad * arrayfire::cos(&t);
    let y = new_circle_rad * arrayfire::sin(&t);

    arrayfire::join(1, &x, &y)
}

pub fn golden_spiral_3D(new_sphere_rad: f64, add_num: u64) -> arrayfire::Array<f64> {
    let N_dims = arrayfire::Dim4::new(&[add_num, 1, 1, 1]);
    let repeat_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let mut idx = arrayfire::iota::<f64>(N_dims, repeat_dims);
    idx = idx + onehalf;

    let mut phi = (two * idx.clone()) / (add_num as f64);
    phi = one - phi;
    phi = arrayfire::acos::<f64>(&phi);

    let magic: f64 = 10.166407384630519631619018026484397683663678586442308240964665618;

    let theta = magic * idx;

    let costheta = arrayfire::cos(&theta);
    let sinphi = arrayfire::sin::<f64>(&phi);

    let x = new_sphere_rad * arrayfire::mul(&costheta, &sinphi, false);

    let sintheta = arrayfire::sin(&theta);

    let y = new_sphere_rad * arrayfire::mul(&sintheta, &sinphi, false);

    let z = new_sphere_rad * arrayfire::cos(&phi);

    let new_pos = arrayfire::join(1, &x, &y);

    arrayfire::join(1, &new_pos, &z)
}

pub fn cube_struct(
    netdata: &network_metadata_type,
    glia_pos: &mut arrayfire::Array<f64>,
    neuron_pos: &mut arrayfire::Array<f64>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();

    let del_unused_neuron: bool = netdata.del_unused_neuron.clone();

    let nratio: f64 = netdata.nratio.clone();
    let neuron_std: f64 = netdata.neuron_std.clone();
    let sphere_rad: f64 = netdata.sphere_rad.clone();
    let neuron_rad: f64 = netdata.neuron_rad.clone();
    let con_rad: f64 = netdata.con_rad.clone();
    let center_const: f64 = netdata.center_const.clone();
    let spring_const: f64 = netdata.spring_const.clone();
    let repel_const: f64 = netdata.repel_const.clone();

    let pos_dims = arrayfire::Dim4::new(&[1, space_dims, 1, 1]);

    let mut new_pos = arrayfire::randu::<f64>(pos_dims);
    new_pos = ((sphere_rad - neuron_rad) * two) * (new_pos - onehalf);

    let mut total_neurons = new_pos.clone();

    let neuron_sq: f64 = 4.0 * neuron_rad * neuron_rad * neuron_rad_factor;

    for i in 0u64..active_size {
        let mut mdist: f64 = 0.0;
        while (mdist < neuron_sq) {
            new_pos =
                ((sphere_rad - neuron_rad) * two) * (arrayfire::randu::<f64>(pos_dims) - onehalf);

            mdist = vec_min_dist(&new_pos, &total_neurons);
        }
        total_neurons = arrayfire::join::<f64>(0, &total_neurons, &new_pos);
    }
    let rand_dims = arrayfire::Dim4::new(&[active_size, 1, 1, 1]);

    let randarr = arrayfire::randu::<f64>(rand_dims);
    let c1 = arrayfire::le(&nratio, &randarr, false);
    let c2 = arrayfire::gt(&nratio, &randarr, false);
    let idx1 = arrayfire::locate(&c1);
    let idx2 = arrayfire::locate(&c2);

    let mut idxrs1 = arrayfire::Indexer::default();
    let seq1 = arrayfire::Seq::new(0.0, (space_dims - 1) as f64, 1.0);
    idxrs1.set_index(&idx1, 0, None);
    idxrs1.set_index(&seq1, 1, Some(false));
    *glia_pos = arrayfire::index_gen(&total_neurons, idxrs1);

    let mut idxrs2 = arrayfire::Indexer::default();
    let seq2 = arrayfire::Seq::new(0.0, (space_dims - 1) as f64, 1.0);
    idxrs2.set_index(&idx2, 0, None);
    idxrs2.set_index(&seq2, 1, Some(false));
    *neuron_pos = arrayfire::index_gen(&total_neurons, idxrs2);
}

pub fn get_inside_idx_cubeV2(
    pos: &arrayfire::Array<f64>,
    cube_size: f64,
    pivot: &Vec<f64>,
) -> arrayfire::Array<u32> {
    let pivot_pos = pivot.clone();
    let space_dims = pivot_pos.len();

    let mut negative_range = pivot_pos[0].clone();
    let mut positive_range = negative_range + cube_size;

    let mut axis = arrayfire::col(pos, 0);

    let mut cmp1 = arrayfire::lt(&axis, &positive_range, false);
    let mut cmp2 = arrayfire::lt(&negative_range, &axis, false);
    cmp1 = arrayfire::and(&cmp1, &cmp2, false);

    for idx in 1..space_dims {
        negative_range = pivot_pos[idx].clone();
        positive_range = negative_range + cube_size;

        axis = arrayfire::col(pos, idx as i64);

        cmp2 = arrayfire::lt(&axis, &positive_range, false);
        cmp1 = arrayfire::and(&cmp1, &cmp2, false);
        cmp2 = arrayfire::lt(&negative_range, &axis, false);
        cmp1 = arrayfire::and(&cmp1, &cmp2, false);
    }

    arrayfire::locate(&cmp1)
}

pub fn select_non_overlap(pos: &arrayfire::Array<f64>, neuron_rad: f64) -> arrayfire::Array<u32> {
    let mut p1 = pos.clone();

    p1 = arrayfire::reorder_v2(&p1, 2, 1, Some(vec![0]));

    let mut magsq = arrayfire::sub(&p1, pos, true);
    drop(p1);
    magsq = arrayfire::pow(&magsq, &two, false);

    magsq = arrayfire::sum(&magsq, 1);

    set_diag(&mut magsq, high);

    let neuron_sq: f64 = 4.0 * neuron_rad * neuron_rad * neuron_rad_factor;

    //Select close objects
    let mut cmp = arrayfire::lt(&magsq, &neuron_sq, false);
    drop(magsq);

    cmp = arrayfire::any_true(&cmp, 2);
    //Lookup  1 >= dir_line  >= 0
    arrayfire::locate(&cmp)
}

pub fn select_non_overlap2(
    pos: &arrayfire::Array<f64>,
    neuron_rad: f64,
) -> (arrayfire::Array<u32>, arrayfire::Array<u32>) {
    let mut p1 = pos.clone();

    p1 = arrayfire::reorder_v2(&p1, 2, 1, Some(vec![0]));

    let mut magsq = arrayfire::sub(&p1, pos, true);
    drop(p1);
    magsq = arrayfire::pow(&magsq, &two, false);

    magsq = arrayfire::sum(&magsq, 1);

    set_diag(&mut magsq, high);

    let neuron_sq: f64 = 4.0 * neuron_rad * neuron_rad * neuron_rad_factor;

    //Select close objects
    let mut cmp = arrayfire::lt(&magsq, &neuron_sq, false);
    drop(magsq);

    let counter_temp = cmp.cast::<u8>();

    let counter = arrayfire::sum(&counter_temp, 2);
    drop(counter_temp);

    cmp = arrayfire::ge(&zero, &counter, false);

    let cmp2 = arrayfire::eq(&cmp, &zero, false);

    //Lookup  1 >= dir_line  >= 0
    (arrayfire::locate(&cmp), arrayfire::locate(&cmp2))
}

/*
Generates a sphere and detects cell collisions in minibatch. Where groups/minibatches of cells are checked

Inputs
netdata:  The sphere radius, neuron radius, mumber of neurons and glial cells to be created

Outputs:
glia_pos:    The 3D position of glial cells in the shape of a 3D sphere
neuron_pos:  The 3D position of neurons in the shape of a 3D sphere


*/

pub fn sphere_cell_collision_minibatch(
    netdata: &network_metadata_type,
    glia_pos: &mut arrayfire::Array<f64>,
    neuron_pos: &mut arrayfire::Array<f64>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();

    let del_unused_neuron: bool = netdata.del_unused_neuron.clone();

    let nratio: f64 = netdata.nratio.clone();
    let neuron_std: f64 = netdata.neuron_std.clone();
    let sphere_rad: f64 = netdata.sphere_rad.clone();
    let neuron_rad: f64 = netdata.neuron_rad.clone();
    let con_rad: f64 = netdata.con_rad.clone();
    let center_const: f64 = netdata.center_const.clone();
    let spring_const: f64 = netdata.spring_const.clone();
    let repel_const: f64 = netdata.repel_const.clone();

    let generate_dims = arrayfire::Dim4::new(&[2 * active_size, 1, 1, 1]);
    let single_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);
    let pos_dims = arrayfire::Dim4::new(&[1, space_dims, 1, 1]);

    let mut r = arrayfire::randu::<f64>(generate_dims);
    r = (sphere_rad - neuron_rad) * arrayfire::cbrt(&r);
    let mut theta = two * (arrayfire::randu::<f64>(generate_dims) - onehalf);
    theta = arrayfire::acos(&theta);
    let mut phi = two * std::f64::consts::PI * arrayfire::randu::<f64>(generate_dims);

    let x = r.clone() * arrayfire::sin(&theta) * arrayfire::cos(&phi);
    let y = r.clone() * arrayfire::sin(&theta) * arrayfire::sin(&phi);
    let z = r.clone() * arrayfire::cos(&theta);

    drop(r);
    drop(theta);
    drop(phi);

    let mut total_obj2 = arrayfire::join_many(1, vec![&x, &y, &z]);
    drop(x);
    drop(y);
    drop(z);

    let mut pivot_rad = ((4.0 / 3.0)
        * std::f64::consts::PI
        * TARGET_DENSITY
        * sphere_rad
        * sphere_rad
        * sphere_rad);
    pivot_rad = (pivot_rad / ((2 * active_size) as f64)).cbrt();

    let pivot_rad2 = pivot_rad + (2.05f64 * neuron_rad * neuron_rad_factor);

    let mut loop_end_flag = false;
    let mut pivot_pos = vec![-sphere_rad; space_dims as usize];

    let single_dims = arrayfire::Dim4::new(&[1, space_dims, 1, 1]);

    let select_idx_dims = arrayfire::Dim4::new(&[total_obj2.dims()[0], 1, 1, 1]);
    let mut select_idx = arrayfire::constant::<bool>(true, select_idx_dims);

    loop {
        let idx = get_inside_idx_cubeV2(&total_obj2, pivot_rad2, &pivot_pos);

        if idx.dims()[0] > 1 {
            let tmp_obj = arrayfire::lookup(&total_obj2, &idx, 0);

            let mut neg_idx = select_non_overlap(&tmp_obj, neuron_rad);

            if neg_idx.dims()[0] > 0 {
                neg_idx = arrayfire::lookup(&idx, &neg_idx, 0);

                let insert = arrayfire::constant::<bool>(false, neg_idx.dims());

                let mut idxrs = arrayfire::Indexer::default();
                idxrs.set_index(&neg_idx, 0, None);
                arrayfire::assign_gen(&mut select_idx, &idxrs, &insert);
            }
        }
        drop(idx);

        pivot_pos[0] = pivot_pos[0] + pivot_rad;

        for idx in 0..space_dims {
            if pivot_pos[idx as usize] > sphere_rad {
                if idx == (space_dims - 1) {
                    loop_end_flag = true;
                    break;
                }

                pivot_pos[idx as usize] = -sphere_rad;
                pivot_pos[(idx + 1) as usize] = pivot_pos[(idx + 1) as usize] + pivot_rad;
            }
        }

        if loop_end_flag {
            break;
        }
    }

    let idx2 = arrayfire::locate(&select_idx);
    drop(select_idx);

    total_obj2 = arrayfire::lookup(&total_obj2, &idx2, 0);

    let total_obj_size = total_obj2.dims()[0];

    let split_idx = ((total_obj_size as f64) * nratio) as u64;

    *neuron_pos = arrayfire::rows(&total_obj2, 0, (split_idx - 1) as i64);

    *glia_pos = arrayfire::rows(&total_obj2, split_idx as i64, (total_obj_size - 1) as i64);
}

pub fn spherical_existingV3(
    netdata: &network_metadata_type,
    glia_pos: &mut arrayfire::Array<f64>,
    neuron_pos: &mut arrayfire::Array<f64>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();

    let del_unused_neuron: bool = netdata.del_unused_neuron.clone();

    let nratio: f64 = netdata.nratio.clone();
    let neuron_std: f64 = netdata.neuron_std.clone();
    let sphere_rad: f64 = netdata.sphere_rad.clone();
    let neuron_rad: f64 = netdata.neuron_rad.clone();
    let con_rad: f64 = netdata.con_rad.clone();
    let center_const: f64 = netdata.center_const.clone();
    let spring_const: f64 = netdata.spring_const.clone();
    let repel_const: f64 = netdata.repel_const.clone();

    let generate_dims = arrayfire::Dim4::new(&[2 * active_size, 1, 1, 1]);
    let single_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);
    let pos_dims = arrayfire::Dim4::new(&[1, space_dims, 1, 1]);

    let mut r = arrayfire::randu::<f64>(generate_dims);
    r = (sphere_rad - neuron_rad) * arrayfire::cbrt(&r);
    let mut theta = two * (arrayfire::randu::<f64>(generate_dims) - onehalf);
    theta = arrayfire::acos(&theta);
    let mut phi = two * std::f64::consts::PI * arrayfire::randu::<f64>(generate_dims);

    let x = r.clone() * arrayfire::sin(&theta) * arrayfire::cos(&phi);
    let y = r.clone() * arrayfire::sin(&theta) * arrayfire::sin(&phi);
    let z = r.clone() * arrayfire::cos(&theta);

    drop(r);
    drop(theta);
    drop(phi);

    let mut total_obj2 = arrayfire::join_many(1, vec![&x, &y, &z]);
    drop(x);
    drop(y);
    drop(z);

    let mut pivot_rad = ((4.0 / 3.0)
        * std::f64::consts::PI
        * TARGET_DENSITY
        * sphere_rad
        * sphere_rad
        * sphere_rad);
    pivot_rad = (pivot_rad / ((2 * active_size) as f64)).cbrt();

    let pivot_rad2 = pivot_rad + (2.05f64 * neuron_rad * neuron_rad_factor);

    let mut loop_end_flag = false;
    let mut pivot_pos = vec![-sphere_rad; space_dims as usize];

    let single_dims = arrayfire::Dim4::new(&[1, space_dims, 1, 1]);

    let select_idx_dims = arrayfire::Dim4::new(&[total_obj2.dims()[0], 1, 1, 1]);
    let mut select_idx = arrayfire::constant::<bool>(true, select_idx_dims);

    let old_obj = arrayfire::join(0, &glia_pos, &neuron_pos);

    loop {
        let idx = get_inside_idx_cubeV2(&total_obj2, pivot_rad2, &pivot_pos);

        let idx_old = get_inside_idx_cubeV2(&old_obj, pivot_rad2, &pivot_pos);

        if idx.dims()[0] > 0 {
            let tmp_obj = arrayfire::lookup(&total_obj2, &idx, 0);

            let mut neg_idx = select_non_overlap(&tmp_obj, neuron_rad);

            if neg_idx.dims()[0] > 0 {
                neg_idx = arrayfire::lookup(&idx, &neg_idx, 0);

                let insert = arrayfire::constant::<bool>(false, neg_idx.dims());

                let mut idxrs = arrayfire::Indexer::default();
                idxrs.set_index(&neg_idx, 0, None);
                arrayfire::assign_gen(&mut select_idx, &idxrs, &insert);
            }
            drop(neg_idx);

            if idx_old.dims()[0] > 0 {
                let old_tmp_obj = arrayfire::lookup(&old_obj, &idx_old, 0);

                let mut p1 = old_tmp_obj.clone();

                p1 = arrayfire::reorder_v2(&p1, 2, 1, Some(vec![0]));

                let mut magsq = arrayfire::sub(&p1, &tmp_obj, true);
                drop(p1);
                magsq = arrayfire::pow(&magsq, &two, false);

                magsq = arrayfire::sum(&magsq, 1);

                let neuron_sq: f64 = 4.0 * neuron_rad * neuron_rad * neuron_rad_factor;

                //Select close objects
                let mut cmp = arrayfire::lt(&magsq, &neuron_sq, false);
                drop(magsq);

                cmp = arrayfire::any_true(&cmp, 2);
                //Lookup  1 >= dir_line  >= 0
                let mut neg_idx = arrayfire::locate(&cmp);

                neg_idx = arrayfire::lookup(&idx, &neg_idx, 0);

                let insert = arrayfire::constant::<bool>(false, neg_idx.dims());

                let mut idxrs = arrayfire::Indexer::default();
                idxrs.set_index(&neg_idx, 0, None);
                arrayfire::assign_gen(&mut select_idx, &idxrs, &insert);
            }
        }
        drop(idx);

        pivot_pos[0] = pivot_pos[0] + pivot_rad;

        for idx in 0..space_dims {
            if pivot_pos[idx as usize] > sphere_rad {
                if idx == (space_dims - 1) {
                    loop_end_flag = true;
                    break;
                }

                pivot_pos[idx as usize] = -sphere_rad;
                pivot_pos[(idx + 1) as usize] = pivot_pos[(idx + 1) as usize] + pivot_rad;
            }
        }

        if loop_end_flag {
            break;
        }
    }

    let idx2 = arrayfire::locate(&select_idx);
    drop(select_idx);
    drop(old_obj);

    total_obj2 = arrayfire::lookup(&total_obj2, &idx2, 0);

    let total_obj_size = total_obj2.dims()[0];

    let split_idx = ((total_obj_size as f64) * nratio) as u64;

    let new_neuron_pos = arrayfire::rows(&total_obj2, 0, (split_idx - 1) as i64);

    let new_glia_pos = arrayfire::rows(&total_obj2, split_idx as i64, (total_obj_size - 1) as i64);

    *neuron_pos = arrayfire::join(0, neuron_pos, &new_neuron_pos);
    *glia_pos = arrayfire::join(0, glia_pos, &new_glia_pos);
}

pub fn assign_neuron_idx(
    netdata: &network_metadata_type,
    neuron_pos: &arrayfire::Array<f64>,
    neuron_idx: &mut arrayfire::Array<i32>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();

    let time_step: f64 = netdata.time_step.clone();
    let nratio: f64 = netdata.nratio.clone();
    let neuron_std: f64 = netdata.neuron_std.clone();
    let sphere_rad: f64 = netdata.sphere_rad.clone();
    let neuron_rad: f64 = netdata.neuron_rad.clone();
    let con_rad: f64 = netdata.con_rad.clone();
    let center_const: f64 = netdata.center_const.clone();
    let spring_const: f64 = netdata.spring_const.clone();
    let repel_const: f64 = netdata.repel_const.clone();

    let N_dims = arrayfire::Dim4::new(&[neuron_pos.dims()[0] - output_size, 1, 1, 1]);
    let repeat_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);
    let L_dims = arrayfire::Dim4::new(&[output_size, 1, 1, 1]);

    *neuron_idx = arrayfire::iota::<i32>(N_dims, repeat_dims);
    let last_neurons =
        arrayfire::iota::<i32>(L_dims, repeat_dims) + ((neuron_size - output_size) as i32);

    *neuron_idx = arrayfire::join(0, neuron_idx, &last_neurons);
}

pub fn assign_neuron_idx_with_buffer(
    max_input_size: u64,
    max_output_size: u64,
    netdata: &network_metadata_type,
    neuron_pos: &arrayfire::Array<f64>,
    neuron_idx: &mut arrayfire::Array<i32>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();

    let repeat_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let input_dims = arrayfire::Dim4::new(&[input_size, 1, 1, 1]);

    let input_idx = arrayfire::iota::<i32>(input_dims, repeat_dims);

    let hidden_dims =
        arrayfire::Dim4::new(&[neuron_pos.dims()[0] - output_size - input_size, 1, 1, 1]);

    let hidden_idx = (max_input_size as i32) + arrayfire::iota::<i32>(hidden_dims, repeat_dims);

    let output_dims = arrayfire::Dim4::new(&[output_size, 1, 1, 1]);

    let output_idx =
        arrayfire::iota::<i32>(output_dims, repeat_dims) + ((neuron_size - max_output_size) as i32);

    *neuron_idx = arrayfire::join(0, &input_idx, &hidden_idx);

    *neuron_idx = arrayfire::join(0, neuron_idx, &output_idx);
}

pub fn input_and_output_layers_old(
    netdata: &network_metadata_type,
    neuron_pos: &arrayfire::Array<f64>,
    neuron_idx: &arrayfire::Array<i32>,

    input_connections: u64,
    output_connections: u64,

    WValues: &mut arrayfire::Array<f64>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();

    let time_step: f64 = netdata.time_step.clone();
    let nratio: f64 = netdata.nratio.clone();
    let neuron_std: f64 = netdata.neuron_std.clone();
    let sphere_rad: f64 = netdata.sphere_rad.clone();
    let neuron_rad: f64 = netdata.neuron_rad.clone();
    let con_rad: f64 = netdata.con_rad.clone();
    let init_prob: f64 = netdata.init_prob.clone();
    let center_const: f64 = netdata.center_const.clone();
    let spring_const: f64 = netdata.spring_const.clone();
    let repel_const: f64 = netdata.repel_const.clone();

    let neuron_dims = neuron_pos.dims();
    let neuron_num = neuron_dims[0];

    let single_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let mut dist = arrayfire::constant::<f64>(0.0, single_dims);
    let mut magsq = arrayfire::constant::<f64>(0.0, single_dims);

    matrix_dist(&neuron_pos, &mut dist, &mut magsq);

    set_diag(&mut magsq, high);

    //Remove self input and output
    let seqs = &[
        arrayfire::Seq::new(
            (input_size as f64),
            (neuron_num - output_size - 1) as f64,
            1.0,
        ),
        arrayfire::Seq::default(),
        arrayfire::Seq::default(),
    ];
    magsq = arrayfire::index(&magsq, seqs);

    //Input
    let seqs = &[
        arrayfire::Seq::default(),
        arrayfire::Seq::default(),
        arrayfire::Seq::new(0.0, (input_size as f64) - 1.0, 1.0),
    ];
    let input_magsq = arrayfire::index(&magsq, seqs);

    let (_, mut inidx) = arrayfire::sort_index(&input_magsq, 0, true);

    let seqs = &[
        arrayfire::Seq::new(0u32, (input_connections as u32) - 1, 1),
        arrayfire::Seq::default(),
        arrayfire::Seq::default(),
    ];
    inidx = arrayfire::index(&inidx, seqs);

    //inidx = inidx + (input_size as u32);

    let input_offset_arr = arrayfire::row(&neuron_idx, input_size as i64);

    let mut input_offset_cpu = vec![i32::default(); input_offset_arr.elements()];
    input_offset_arr.host(&mut input_offset_cpu);

    inidx = inidx + (input_offset_cpu[0] as u32);

    inidx = arrayfire::flat(&inidx);

    let mut first_idx = arrayfire::rows(&neuron_idx, 0, (input_size - 1) as i64);

    first_idx = arrayfire::transpose(&first_idx, false);

    let tile_dims = arrayfire::Dim4::new(&[input_connections, 1, 1, 1]);

    first_idx = arrayfire::tile(&first_idx, tile_dims);

    first_idx = arrayfire::flat(&first_idx);

    *WColIdx = first_idx;

    *WRowIdxCOO = inidx.cast::<i32>();

    //Output
    let seqs = &[
        arrayfire::Seq::default(),
        arrayfire::Seq::default(),
        arrayfire::Seq::new(
            (neuron_num - output_size) as f64,
            (neuron_num as f64) - 1.0,
            1.0,
        ),
    ];
    let output_magsq = arrayfire::index(&magsq, seqs);

    let (_, mut outidx) = arrayfire::sort_index(&output_magsq, 0, true);

    let seqs = &[
        arrayfire::Seq::new(0u32, (output_connections as u32) - 1, 1),
        arrayfire::Seq::default(),
        arrayfire::Seq::default(),
    ];
    outidx = arrayfire::index(&outidx, seqs);

    //outidx = outidx + (input_size as u32);

    outidx = outidx + (input_offset_cpu[0] as u32);

    outidx = arrayfire::flat(&outidx);

    let outidx2 = outidx.cast::<i32>();

    let mut last_idx = arrayfire::rows(
        &neuron_idx,
        (neuron_idx.dims()[0] - output_size) as i64,
        (neuron_idx.dims()[0] - 1) as i64,
    );

    last_idx = arrayfire::transpose(&last_idx, false);

    let tile_dims = arrayfire::Dim4::new(&[output_connections, 1, 1, 1]);

    last_idx = arrayfire::tile(&last_idx, tile_dims);

    last_idx = arrayfire::flat(&last_idx);

    *WColIdx = arrayfire::join(0, WColIdx, &outidx2);

    *WRowIdxCOO = arrayfire::join(0, WRowIdxCOO, &last_idx);

    let mut global_idx = get_global_weight_idx(neuron_size, WRowIdxCOO, WColIdx);

    global_idx = arrayfire::set_unique(&global_idx, false);

    *WRowIdxCOO = arrayfire::div(&global_idx, &neuron_size, false).cast::<i32>();

    *WColIdx = arrayfire::modulo(&global_idx, &neuron_size, false).cast::<i32>();

    *WValues = neuron_std * arrayfire::randn::<f64>(WRowIdxCOO.dims());
}

pub fn matrix_dist_sort(
    num: u64,

    position0: &arrayfire::Array<f64>,
    position1: &arrayfire::Array<f64>,

    magsq_matrix: &mut arrayfire::Array<f64>,
    idx_matrix: &mut arrayfire::Array<u32>,
) {
    *magsq_matrix = position1.clone();

    *magsq_matrix = arrayfire::reorder_v2(magsq_matrix, 2, 1, Some(vec![0]));

    *magsq_matrix = arrayfire::sub(position0, magsq_matrix, true);

    *magsq_matrix = arrayfire::pow(magsq_matrix, &two, false);

    *magsq_matrix = arrayfire::sum(magsq_matrix, 1);

    //arrayfire::print_gen("magsq_matrix".to_string(), &magsq_matrix, Some(6));

    let (magsq, idx) = arrayfire::sort_index(magsq_matrix, 2, true);

    //arrayfire::print_gen("magsq".to_string(), &magsq, Some(6));

    *magsq_matrix = arrayfire::slices(&magsq, 0, (num - 1) as i64);

    drop(magsq);

    *idx_matrix = arrayfire::slices(&idx, 0, (num - 1) as i64);
}

pub fn input_layers2(
    netdata: &network_metadata_type,
    neuron_pos: &arrayfire::Array<f64>,
    neuron_idx: &arrayfire::Array<i32>,

    input_connections: u64,
    batch_size: u64,

    WValues: &mut arrayfire::Array<f64>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();

    let time_step: f64 = netdata.time_step.clone();
    let nratio: f64 = netdata.nratio.clone();
    let neuron_std: f64 = netdata.neuron_std.clone();
    let sphere_rad: f64 = netdata.sphere_rad.clone();
    let neuron_rad: f64 = netdata.neuron_rad.clone();
    let con_rad: f64 = netdata.con_rad.clone();
    let init_prob: f64 = netdata.init_prob.clone();
    let center_const: f64 = netdata.center_const.clone();
    let spring_const: f64 = netdata.spring_const.clone();
    let repel_const: f64 = netdata.repel_const.clone();

    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let neuron_dims = neuron_pos.dims();
    let neuron_num = neuron_dims[0];

    let mut input_neurons = arrayfire::rows(neuron_pos, 0, (input_size - 1) as i64);

    let mut hidden_neurons = arrayfire::rows(
        neuron_pos,
        input_size as i64,
        (neuron_num - output_size - 1) as i64,
    );

    let mut i: u64 = 0;
    let mut startseq: u64 = 0;
    let mut endseq: u64 = 0;
    let total_size = hidden_neurons.dims()[0];

    startseq = i;
    endseq = i + batch_size - 1;
    if (endseq >= (total_size - 1)) {
        endseq = total_size - 1;
    }

    let mut targetarr = arrayfire::rows(&hidden_neurons, startseq as i64, endseq as i64);

    let mut tempnum = input_connections.clone();

    if tempnum > targetarr.dims()[0] {
        tempnum = targetarr.dims()[0];
    }

    let mut magsq_matrix = arrayfire::constant::<f64>(0.0, temp_dims);

    let mut idx_matrix = arrayfire::constant::<u32>(0, temp_dims);

    matrix_dist_sort(
        tempnum,
        &input_neurons,
        &targetarr,
        &mut magsq_matrix,
        &mut idx_matrix,
    );

    i = i + batch_size;

    while i < total_size {
        startseq = i;
        endseq = i + batch_size - 1;
        if (endseq >= (total_size - 1)) {
            endseq = total_size - 1;
        }

        targetarr = arrayfire::rows(&hidden_neurons, startseq as i64, endseq as i64);

        tempnum = input_connections.clone();

        if tempnum > targetarr.dims()[0] {
            tempnum = targetarr.dims()[0];
        }

        let mut magsq_matrix2 = arrayfire::constant::<f64>(0.0, temp_dims);

        let mut idx_matrix2 = arrayfire::constant::<u32>(0, temp_dims);

        matrix_dist_sort(
            tempnum,
            &input_neurons,
            &targetarr,
            &mut magsq_matrix2,
            &mut idx_matrix2,
        );

        idx_matrix2 = idx_matrix2 + (startseq as u32);

        idx_matrix = arrayfire::join(2, &idx_matrix, &idx_matrix2);
        drop(idx_matrix2);

        magsq_matrix = arrayfire::join(2, &magsq_matrix, &magsq_matrix2);
        drop(magsq_matrix2);

        let (_, mut idx) = arrayfire::sort_index(&magsq_matrix, 2, true);

        if idx.dims()[2] > input_connections {
            idx = arrayfire::slices(&idx, 0, (input_connections - 1) as i64);
        }

        magsq_matrix = parallel_lookup(0, 2, &idx, &magsq_matrix);

        idx_matrix = parallel_lookup_u32(0, 2, &idx, &idx_matrix);

        i = i + batch_size;
    }

    drop(magsq_matrix);
    drop(hidden_neurons);
    drop(input_neurons);

    let mut newWRowIdxCOO = idx_matrix.clone().cast::<i32>();

    drop(idx_matrix);

    let mut hidden_idx = arrayfire::rows(
        neuron_idx,
        input_size as i64,
        (neuron_num - output_size - 1) as i64,
    );

    /*
    let mut hidden_idx_cpu = vec!(i32::default();hidden_idx.elements());

    hidden_idx.host(&mut  hidden_idx_cpu);

    newWRowIdxCOO = newWRowIdxCOO + hidden_idx_cpu[0] ;
    */

    newWRowIdxCOO = arrayfire::flat(&newWRowIdxCOO);

    newWRowIdxCOO = arrayfire::lookup(&hidden_idx, &newWRowIdxCOO, 0);
    drop(hidden_idx);

    let repeat_dims = arrayfire::Dim4::new(&[1, input_connections, 1, 1]);
    let mut tile_dims = arrayfire::Dim4::new(&[input_size, 1, 1, 1]);

    let mut newWColIdx = arrayfire::iota::<i32>(tile_dims, repeat_dims);

    newWColIdx = arrayfire::flat(&newWColIdx);

    *WColIdx = newWColIdx.clone();

    *WRowIdxCOO = newWRowIdxCOO.clone();

    /*
    *WColIdx = arrayfire::join(0, WColIdx , &newWColIdx );

    *WRowIdxCOO = arrayfire::join(0, WRowIdxCOO , &newWRowIdxCOO );
    */

    drop(newWColIdx);
    drop(newWRowIdxCOO);

    let mut global_idx = get_global_weight_idx(neuron_size, WRowIdxCOO, WColIdx);

    global_idx = arrayfire::set_unique(&global_idx, false);

    *WRowIdxCOO = arrayfire::div(&global_idx, &neuron_size, false).cast::<i32>();

    *WColIdx = arrayfire::modulo(&global_idx, &neuron_size, false).cast::<i32>();

    *WValues = neuron_std * arrayfire::randn::<f64>(WRowIdxCOO.dims());
}

pub fn output_layers2(
    netdata: &network_metadata_type,
    neuron_pos: &arrayfire::Array<f64>,
    neuron_idx: &arrayfire::Array<i32>,

    output_connections: u64,
    batch_size: u64,

    WValues: &mut arrayfire::Array<f64>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();

    let time_step: f64 = netdata.time_step.clone();
    let nratio: f64 = netdata.nratio.clone();
    let neuron_std: f64 = netdata.neuron_std.clone();
    let sphere_rad: f64 = netdata.sphere_rad.clone();
    let neuron_rad: f64 = netdata.neuron_rad.clone();
    let con_rad: f64 = netdata.con_rad.clone();
    let init_prob: f64 = netdata.init_prob.clone();
    let center_const: f64 = netdata.center_const.clone();
    let spring_const: f64 = netdata.spring_const.clone();
    let repel_const: f64 = netdata.repel_const.clone();

    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let neuron_dims = neuron_pos.dims();
    let neuron_num = neuron_dims[0];

    let mut hidden_neurons = arrayfire::rows(
        neuron_pos,
        input_size as i64,
        (neuron_num - output_size - 1) as i64,
    );

    let mut output_neurons = arrayfire::rows(
        neuron_pos,
        (neuron_num - output_size) as i64,
        (neuron_num - 1) as i64,
    );

    let mut i: u64 = 0;
    let mut startseq: u64 = 0;
    let mut endseq: u64 = 0;
    let total_size = hidden_neurons.dims()[0];

    startseq = i;
    endseq = i + batch_size - 1;
    if (endseq >= (total_size - 1)) {
        endseq = total_size - 1;
    }

    let mut targetarr = arrayfire::rows(&hidden_neurons, startseq as i64, endseq as i64);

    let mut tempnum = output_connections.clone();

    if tempnum > targetarr.dims()[0] {
        tempnum = targetarr.dims()[0];
    }

    let mut magsq_matrix = arrayfire::constant::<f64>(0.0, temp_dims);

    let mut idx_matrix = arrayfire::constant::<u32>(0, temp_dims);

    matrix_dist_sort(
        tempnum,
        &output_neurons,
        &targetarr,
        &mut magsq_matrix,
        &mut idx_matrix,
    );

    i = i + batch_size;

    while i < total_size {
        startseq = i;
        endseq = i + batch_size - 1;
        if (endseq >= (total_size - 1)) {
            endseq = total_size - 1;
        }

        targetarr = arrayfire::rows(&hidden_neurons, startseq as i64, endseq as i64);

        tempnum = output_connections.clone();

        if tempnum > targetarr.dims()[0] {
            tempnum = targetarr.dims()[0];
        }

        let mut magsq_matrix2 = arrayfire::constant::<f64>(0.0, temp_dims);

        let mut idx_matrix2 = arrayfire::constant::<u32>(0, temp_dims);

        matrix_dist_sort(
            tempnum,
            &output_neurons,
            &targetarr,
            &mut magsq_matrix2,
            &mut idx_matrix2,
        );

        idx_matrix2 = idx_matrix2 + (startseq as u32);

        idx_matrix = arrayfire::join(2, &idx_matrix, &idx_matrix2);
        drop(idx_matrix2);

        magsq_matrix = arrayfire::join(2, &magsq_matrix, &magsq_matrix2);
        drop(magsq_matrix2);

        let (_, mut idx) = arrayfire::sort_index(&magsq_matrix, 2, true);

        if idx.dims()[2] > output_connections {
            idx = arrayfire::slices(&idx, 0, (output_connections - 1) as i64);
        }

        magsq_matrix = parallel_lookup(0, 2, &idx, &magsq_matrix);

        idx_matrix = parallel_lookup_u32(0, 2, &idx, &idx_matrix);

        i = i + batch_size;
    }

    drop(magsq_matrix);
    drop(hidden_neurons);
    drop(output_neurons);

    let mut newWRowIdxCOO = idx_matrix.clone().cast::<i32>();

    drop(idx_matrix);

    let mut hidden_idx = arrayfire::rows(
        neuron_idx,
        input_size as i64,
        (neuron_num - output_size - 1) as i64,
    );

    /*
    let mut hidden_idx_cpu = vec!(i32::default();hidden_idx.elements());

    hidden_idx.host(&mut  hidden_idx_cpu);

    newWRowIdxCOO = newWRowIdxCOO + hidden_idx_cpu[0] ;
    */

    newWRowIdxCOO = arrayfire::flat(&newWRowIdxCOO);

    newWRowIdxCOO = arrayfire::lookup(&hidden_idx, &newWRowIdxCOO, 0);
    drop(hidden_idx);

    //drop(hidden_idx_cpu);
    //drop(hidden_idx);

    let mut output_idx = arrayfire::rows(
        neuron_idx,
        (neuron_num - output_size) as i64,
        (neuron_num - 1) as i64,
    );

    /*
    let mut output_idx_cpu = vec!(i32::default();output_idx.elements());

    output_idx.host(&mut  output_idx_cpu);
    */

    let repeat_dims = arrayfire::Dim4::new(&[1, output_connections, 1, 1]);
    let mut tile_dims = arrayfire::Dim4::new(&[output_size, 1, 1, 1]);

    let mut newWColIdx = arrayfire::iota::<i32>(tile_dims, repeat_dims);

    //newWColIdx = arrayfire::flat(&newWColIdx)  +  output_idx_cpu[0];

    newWColIdx = arrayfire::flat(&newWColIdx);

    newWColIdx = arrayfire::lookup(&output_idx, &newWColIdx, 0);

    *WColIdx = arrayfire::join(0, WColIdx, &newWRowIdxCOO);

    *WRowIdxCOO = arrayfire::join(0, WRowIdxCOO, &newWColIdx);

    drop(newWColIdx);
    drop(newWRowIdxCOO);

    let mut global_idx = get_global_weight_idx(neuron_size, WRowIdxCOO, WColIdx);

    global_idx = arrayfire::set_unique(&global_idx, false);

    *WRowIdxCOO = arrayfire::div(&global_idx, &neuron_size, false).cast::<i32>();

    *WColIdx = arrayfire::modulo(&global_idx, &neuron_size, false).cast::<i32>();

    *WValues = neuron_std * arrayfire::randn::<f64>(WRowIdxCOO.dims());
}

pub fn input_and_output_layers(
    netdata: &network_metadata_type,
    neuron_pos: &arrayfire::Array<f64>,
    neuron_idx: &arrayfire::Array<i32>,

    input_connections: u64,
    output_connections: u64,

    WValues: &mut arrayfire::Array<f64>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
) {
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();

    let COO_batch_size = 1 + ((COO_find_limit / neuron_pos.dims()[0]) as u64);

    arrayfire::device_gc();

    input_layers2(
        netdata,
        neuron_pos,
        neuron_idx,
        input_connections,
        COO_batch_size,
        WValues,
        WRowIdxCOO,
        WColIdx,
    );

    arrayfire::device_gc();

    output_layers2(
        netdata,
        neuron_pos,
        neuron_idx,
        output_connections,
        COO_batch_size,
        WValues,
        WRowIdxCOO,
        WColIdx,
    );

    arrayfire::device_gc();
}

pub fn hidden_layers_old(
    netdata: &network_metadata_type,
    neuron_pos: &arrayfire::Array<f64>,
    neuron_idx: &arrayfire::Array<i32>,

    WValues: &mut arrayfire::Array<f64>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();

    let time_step: f64 = netdata.time_step.clone();
    let nratio: f64 = netdata.nratio.clone();
    let neuron_std: f64 = netdata.neuron_std.clone();
    let sphere_rad: f64 = netdata.sphere_rad.clone();
    let neuron_rad: f64 = netdata.neuron_rad.clone();
    let con_rad: f64 = netdata.con_rad.clone();
    let init_prob: f64 = netdata.init_prob.clone();
    let center_const: f64 = netdata.center_const.clone();
    let spring_const: f64 = netdata.spring_const.clone();
    let repel_const: f64 = netdata.repel_const.clone();

    let neuron_dims = neuron_pos.dims();
    let neuron_num = neuron_dims[0];

    let single_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let neuron_sq: f64 = 4.0 * neuron_rad * neuron_rad;
    let con_sq: f64 = 4.0 * con_rad * con_rad;

    let alpha: f64 = ((0.01 as f64).ln()) / (con_sq - neuron_sq);

    let mut dist = arrayfire::constant::<f64>(0.0, single_dims);
    let mut magsq = arrayfire::constant::<f64>(0.0, single_dims);

    matrix_dist(&neuron_pos, &mut dist, &mut magsq);

    set_diag(&mut magsq, high);

    //Get neurons in con_sq
    let mut cmp1 = arrayfire::lt(&magsq, &con_sq, false);
    cmp1 = arrayfire::flat(&cmp1);
    let sel1 = arrayfire::locate(&cmp1);

    //No selected values
    if sel1.dims()[0] == 0 {
        return;
    }

    magsq = arrayfire::flat(&magsq);
    let mut idxrs1 = arrayfire::Indexer::default();
    idxrs1.set_index(&sel1, 0, None);
    let mut mg = arrayfire::index_gen(&magsq, idxrs1);

    mg = (mg - neuron_sq) * alpha;
    mg = init_prob * arrayfire::exp(&mg);

    let randarr = arrayfire::randu::<f64>(mg.dims());

    let cmp2 = arrayfire::lt(&randarr, &mg, false);
    let sel2 = arrayfire::locate(&cmp2);

    //No selected values
    if sel2.dims()[0] == 0 {
        return;
    }

    let mut idxrs2 = arrayfire::Indexer::default();
    idxrs2.set_index(&sel2, 0, None);
    let mut selidx = arrayfire::index_gen(&sel1, idxrs2);

    //No selected values
    if selidx.dims()[0] == 0 {
        return;
    }

    selidx = arrayfire::set_unique(&selidx, false);

    let col = arrayfire::modulo(&selidx, &neuron_num, false);

    let mut idxrs3 = arrayfire::Indexer::default();
    idxrs3.set_index(&col, 0, None);
    let newWColIdx = arrayfire::index_gen(neuron_idx, idxrs3);

    let row = arrayfire::div(&selidx, &neuron_num, false);

    let mut idxrs4 = arrayfire::Indexer::default();
    idxrs4.set_index(&row, 0, None);
    let newWRowIdxCOO = arrayfire::index_gen(neuron_idx, idxrs4);

    *WColIdx = arrayfire::join(0, WColIdx, &newWColIdx);

    *WRowIdxCOO = arrayfire::join(0, WRowIdxCOO, &newWRowIdxCOO);

    let mut global_idx = get_global_weight_idx(neuron_size, WRowIdxCOO, WColIdx);

    global_idx = arrayfire::set_unique(&global_idx, false);

    *WRowIdxCOO = arrayfire::div(&global_idx, &neuron_size, false).cast::<i32>();

    *WColIdx = arrayfire::modulo(&global_idx, &neuron_size, false).cast::<i32>();

    *WValues = neuron_std * arrayfire::randn::<f64>(WRowIdxCOO.dims());
}

pub fn matrix_dist_connect(
    netdata: &network_metadata_type,

    position0: &arrayfire::Array<f64>,
    position1: &arrayfire::Array<f64>,

    col_idx: &mut arrayfire::Array<u32>,
    row_idx: &mut arrayfire::Array<u32>,
) -> bool {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();

    let time_step: f64 = netdata.time_step.clone();
    let nratio: f64 = netdata.nratio.clone();
    let neuron_std: f64 = netdata.neuron_std.clone();
    let sphere_rad: f64 = netdata.sphere_rad.clone();
    let neuron_rad: f64 = netdata.neuron_rad.clone();
    let con_rad: f64 = netdata.con_rad.clone();
    let init_prob: f64 = netdata.init_prob.clone();
    let center_const: f64 = netdata.center_const.clone();
    let spring_const: f64 = netdata.spring_const.clone();
    let repel_const: f64 = netdata.repel_const.clone();

    //let neuron_dims = position0.dims();
    //let neuron_num = neuron_dims[0];

    let position0_size = position0.dims()[0] as u32;
    //let position1_size = position1.dims()[0];

    //let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);

    let neuron_sq: f64 = 4.0 * neuron_rad * neuron_rad;
    let con_sq: f64 = 4.0 * con_rad * con_rad;

    let alpha: f64 = ((0.01 as f64).ln()) / (con_sq - neuron_sq);

    let mut magsq_matrix = position1.clone();

    magsq_matrix = arrayfire::reorder_v2(&magsq_matrix, 2, 1, Some(vec![0]));

    magsq_matrix = arrayfire::sub(position0, &magsq_matrix, true);

    magsq_matrix = arrayfire::pow(&magsq_matrix, &two, false);

    magsq_matrix = arrayfire::sum(&magsq_matrix, 1);

    //Get neurons in con_sq
    let mut cmp1 = arrayfire::lt(&magsq_matrix, &con_sq, false);
    cmp1 = arrayfire::flat(&cmp1);
    let sel1 = arrayfire::locate(&cmp1);

    //No selected values
    if sel1.dims()[0] == 0 {
        return false;
    }

    magsq_matrix = arrayfire::flat(&magsq_matrix);
    let mut idxrs1 = arrayfire::Indexer::default();
    idxrs1.set_index(&sel1, 0, None);
    let mut mg = arrayfire::index_gen(&magsq_matrix, idxrs1);

    drop(magsq_matrix);

    mg = (mg - neuron_sq) * alpha;
    mg = init_prob * arrayfire::exp(&mg);

    let randarr = arrayfire::randu::<f64>(mg.dims());

    let cmp2 = arrayfire::lt(&randarr, &mg, false);
    let sel2 = arrayfire::locate(&cmp2);

    //No selected values
    if sel2.dims()[0] == 0 {
        return false;
    }

    let mut idxrs2 = arrayfire::Indexer::default();
    idxrs2.set_index(&sel2, 0, None);
    let mut selidx = arrayfire::index_gen(&sel1, idxrs2);

    //No selected values
    if selidx.dims()[0] == 0 {
        return false;
    }

    selidx = arrayfire::set_unique(&selidx, false);

    *col_idx = arrayfire::modulo(&selidx, &position0_size, false);

    *row_idx = arrayfire::div(&selidx, &position0_size, false);

    true
}

pub fn hidden_layers(
    netdata: &network_metadata_type,
    neuron_pos: &arrayfire::Array<f64>,
    neuron_idx: &arrayfire::Array<i32>,

    WValues: &mut arrayfire::Array<f64>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();

    let time_step: f64 = netdata.time_step.clone();
    let nratio: f64 = netdata.nratio.clone();
    let neuron_std: f64 = netdata.neuron_std.clone();
    let sphere_rad: f64 = netdata.sphere_rad.clone();
    let neuron_rad: f64 = netdata.neuron_rad.clone();
    let con_rad: f64 = netdata.con_rad.clone();
    let init_prob: f64 = netdata.init_prob.clone();
    let center_const: f64 = netdata.center_const.clone();
    let spring_const: f64 = netdata.spring_const.clone();
    let repel_const: f64 = netdata.repel_const.clone();

    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let neuron_dims = neuron_pos.dims();
    let neuron_num = neuron_dims[0];

    //let mut hidden_neurons = arrayfire::rows(neuron_pos, input_size as i64, (neuron_num-output_size-1)  as i64);

    let batch_size = 1 + ((COO_find_limit / (10 * neuron_num)) as u64);

    let mut i: u64 = 0;
    let mut startseq: u64 = 0;
    let mut endseq: u64 = 0;
    let total_size = neuron_pos.dims()[0];

    startseq = i;
    endseq = i + batch_size - 1;
    if (endseq >= (total_size - 1)) {
        endseq = total_size - 1;
    }

    let mut targetarr = arrayfire::rows(&neuron_pos, startseq as i64, endseq as i64);

    let mut col_idx = arrayfire::constant::<u32>(0, temp_dims);

    let mut row_idx = arrayfire::constant::<u32>(0, temp_dims);

    let mut result =
        matrix_dist_connect(netdata, &neuron_pos, &targetarr, &mut col_idx, &mut row_idx);

    if result == true {
    } else {
        col_idx = arrayfire::constant::<u32>(0, temp_dims);

        row_idx = arrayfire::constant::<u32>(0, temp_dims);
    }

    i = i + batch_size;

    while i < total_size {
        startseq = i;
        endseq = i + batch_size - 1;
        if (endseq >= (total_size - 1)) {
            endseq = total_size - 1;
        }

        targetarr = arrayfire::rows(&neuron_pos, startseq as i64, endseq as i64);

        let mut col_idx2 = arrayfire::constant::<u32>(0, temp_dims);

        let mut row_idx2 = arrayfire::constant::<u32>(0, temp_dims);

        result = matrix_dist_connect(
            netdata,
            &neuron_pos,
            &targetarr,
            &mut col_idx2,
            &mut row_idx2,
        );

        if result == true {
            row_idx2 = row_idx2 + (startseq as u32);

            col_idx = arrayfire::join(0, &col_idx, &col_idx2);

            row_idx = arrayfire::join(0, &row_idx, &row_idx2);
        } else {
        }

        drop(col_idx2);

        drop(row_idx2);

        i = i + batch_size;
    }

    //drop(magsq_matrix);
    //drop(hidden_neurons);
    //drop(output_neurons);

    let mut newWRowIdxCOO = row_idx.clone().cast::<i32>();
    let mut newWColIdx = col_idx.clone().cast::<i32>();

    drop(row_idx);
    drop(col_idx);

    //let mut hidden_idx = arrayfire::rows(neuron_idx, input_size as i64, (neuron_num-output_size-1)  as i64);

    /*
    let mut hidden_idx_cpu = vec!(i32::default();hidden_idx.elements());

    hidden_idx.host(&mut  hidden_idx_cpu);

    newWRowIdxCOO = newWRowIdxCOO + hidden_idx_cpu[0] ;

    newWColIdx = newWColIdx + hidden_idx_cpu[0] ;
    */

    newWRowIdxCOO = arrayfire::flat(&newWRowIdxCOO);

    newWColIdx = arrayfire::flat(&newWColIdx);

    newWRowIdxCOO = arrayfire::lookup(&neuron_idx, &newWRowIdxCOO, 0);

    newWColIdx = arrayfire::lookup(&neuron_idx, &newWColIdx, 0);

    //drop(hidden_idx_cpu);
    //drop(hidden_idx);

    *WColIdx = arrayfire::join(0, WColIdx, &newWRowIdxCOO);

    *WRowIdxCOO = arrayfire::join(0, WRowIdxCOO, &newWColIdx);

    drop(newWColIdx);
    drop(newWRowIdxCOO);

    let mut global_idx = get_global_weight_idx(neuron_size, WRowIdxCOO, WColIdx);

    global_idx = arrayfire::set_unique(&global_idx, false);

    *WRowIdxCOO = arrayfire::div(&global_idx, &neuron_size, false).cast::<i32>();

    *WColIdx = arrayfire::modulo(&global_idx, &neuron_size, false).cast::<i32>();

    *WValues = neuron_std * arrayfire::randn::<f64>(WRowIdxCOO.dims());
}

pub fn fully_connected_hidden_layers(
    neuron_pos: &arrayfire::Array<f64>,
    neuron_idx: &arrayfire::Array<i32>,

    netdata: &mut network_metadata_type,
    WValues: &mut arrayfire::Array<f64>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
) {
    (*netdata).init_prob = 0.01;

    let neuron_size: u64 = (*netdata).neuron_size.clone();
    let input_size: u64 = (*netdata).input_size.clone();
    let output_size: u64 = (*netdata).output_size.clone();
    let proc_num: u64 = (*netdata).proc_num.clone();
    let active_size: u64 = (*netdata).active_size.clone();
    let space_dims: u64 = (*netdata).space_dims.clone();
    let step_num: u64 = (*netdata).step_num.clone();

    let time_step: f64 = (*netdata).time_step.clone();
    let nratio: f64 = (*netdata).nratio.clone();
    let neuron_std: f64 = (*netdata).neuron_std.clone();
    let sphere_rad: f64 = (*netdata).sphere_rad.clone();
    let neuron_rad: f64 = (*netdata).neuron_rad.clone();
    let con_rad: f64 = (*netdata).con_rad.clone();
    let init_prob: f64 = (*netdata).init_prob.clone();
    let center_const: f64 = (*netdata).center_const.clone();
    let spring_const: f64 = (*netdata).spring_const.clone();
    let repel_const: f64 = (*netdata).repel_const.clone();

    let in_idx = arrayfire::rows(neuron_idx, 0, (input_size - 1) as i64);

    let out_idx = arrayfire::rows(
        neuron_idx,
        (neuron_idx.dims()[0] - output_size) as i64,
        (neuron_idx.dims()[0] - 1) as i64,
    );

    let mut tempWValues = WValues.clone();
    let mut tempWRowIdxCOO = WRowIdxCOO.clone();
    let mut tempWColIdx = WColIdx.clone();

    while 1 == 1 {
        tempWValues = WValues.clone();
        tempWRowIdxCOO = WRowIdxCOO.clone();
        tempWColIdx = WColIdx.clone();

        hidden_layers(
            netdata,
            neuron_pos,
            neuron_idx,
            &mut tempWValues,
            &mut tempWRowIdxCOO,
            &mut tempWColIdx,
        );

        clear_input(
            &mut tempWValues,
            &mut tempWRowIdxCOO,
            &mut tempWColIdx,
            input_size,
        );

        clear_output(
            &mut tempWValues,
            &mut tempWRowIdxCOO,
            &mut tempWColIdx,
            neuron_size - output_size,
        );

        let connected = check_connected2(
            &in_idx,
            &out_idx,
            &tempWRowIdxCOO,
            &tempWColIdx,
            neuron_size,
            proc_num,
        );

        if connected {
            break;
        }

        (*netdata).init_prob = (*netdata).init_prob.clone() * 1.05;
    }

    *WValues = tempWValues.clone();
    *WRowIdxCOO = tempWRowIdxCOO.clone();
    *WColIdx = tempWColIdx.clone();
}

pub fn self_loops(
    netdata: &network_metadata_type,
    //neuron_pos: &arrayfire::Array<f64>,
    neuron_idx: &arrayfire::Array<i32>,

    WValues: &mut arrayfire::Array<f64>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();

    let time_step: f64 = netdata.time_step.clone();
    let nratio: f64 = netdata.nratio.clone();
    let neuron_std: f64 = netdata.neuron_std.clone();
    let sphere_rad: f64 = netdata.sphere_rad.clone();
    let neuron_rad: f64 = netdata.neuron_rad.clone();
    let con_rad: f64 = netdata.con_rad.clone();
    let init_prob: f64 = netdata.init_prob.clone();
    let center_const: f64 = netdata.center_const.clone();
    let spring_const: f64 = netdata.spring_const.clone();
    let repel_const: f64 = netdata.repel_const.clone();

    let mut active_size = neuron_idx.dims()[0];

    let hidden_idx = arrayfire::rows(
        &neuron_idx,
        input_size as i64,
        (active_size - output_size - 1) as i64,
    );

    *WColIdx = arrayfire::join(0, WColIdx, &hidden_idx);

    *WRowIdxCOO = arrayfire::join(0, WRowIdxCOO, &hidden_idx);

    let mut global_idx = get_global_weight_idx(neuron_size, WRowIdxCOO, WColIdx);

    global_idx = arrayfire::set_unique(&global_idx, false);

    *WRowIdxCOO = arrayfire::div(&global_idx, &neuron_size, false).cast::<i32>();

    *WColIdx = arrayfire::modulo(&global_idx, &neuron_size, false).cast::<i32>();

    *WValues = 0.00000001f64 * arrayfire::randn::<f64>(WRowIdxCOO.dims());
}

pub fn assign_self_loop_value(
    netdata: &network_metadata_type,
    //neuron_pos: &arrayfire::Array<f64>,
    neuron_idx: &arrayfire::Array<i32>,

    WValues: &mut arrayfire::Array<f64>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();

    let time_step: f64 = netdata.time_step.clone();
    let nratio: f64 = netdata.nratio.clone();
    let neuron_std: f64 = netdata.neuron_std.clone();
    let sphere_rad: f64 = netdata.sphere_rad.clone();
    let neuron_rad: f64 = netdata.neuron_rad.clone();
    let con_rad: f64 = netdata.con_rad.clone();
    let init_prob: f64 = netdata.init_prob.clone();
    let center_const: f64 = netdata.center_const.clone();
    let spring_const: f64 = netdata.spring_const.clone();
    let repel_const: f64 = netdata.repel_const.clone();

    let mut active_size = neuron_idx.dims()[0];

    let hidden_idx = arrayfire::rows(
        &neuron_idx,
        input_size as i64,
        (active_size - output_size - 1) as i64,
    );

    let tempWColIdx = hidden_idx.clone();

    let tempWRowIdxCOO = hidden_idx.clone();

    let global_idx = get_global_weight_idx(neuron_size, &tempWRowIdxCOO, &tempWColIdx);

    let global_idx2 = get_global_weight_idx(neuron_size, &WRowIdxCOO, &WColIdx);

    let COO_batch_size = 1 + ((COO_find_limit / global_idx2.dims()[0]) as u64);

    let valsel = COO_batch_find(&global_idx2, &global_idx, COO_batch_size);

    if valsel.dims()[0] == 0 {
        return;
    }

    let newWValues = 0.00000001f64 * arrayfire::randn::<f64>(valsel.dims());

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&valsel, 0, None);
    arrayfire::assign_gen(WValues, &idxrs, &newWValues);
}

/*
Generates a sphere and detects cell collisions in serial. Where each cell is checked one by one

Inputs
netdata:  The sphere radius, neuron radius, mumber of neurons and glial cells to be created

Outputs:
glia_pos:    The 3D position of glial cells in the shape of a 3D sphere
neuron_pos:  The 3D position of neurons in the shape of a 3D sphere


*/

pub fn sphere_cell_collision_serial(
    netdata: &network_metadata_type,
    glia_pos: &mut arrayfire::Array<f64>,
    neuron_pos: &mut arrayfire::Array<f64>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();

    let del_unused_neuron: bool = netdata.del_unused_neuron.clone();

    let nratio: f64 = netdata.nratio.clone();
    let neuron_std: f64 = netdata.neuron_std.clone();
    let sphere_rad: f64 = netdata.sphere_rad.clone();
    let neuron_rad: f64 = netdata.neuron_rad.clone();
    let con_rad: f64 = netdata.con_rad.clone();
    let center_const: f64 = netdata.center_const.clone();
    let spring_const: f64 = netdata.spring_const.clone();
    let repel_const: f64 = netdata.repel_const.clone();

    let generate_dims = arrayfire::Dim4::new(&[2 * active_size, 1, 1, 1]);
    let single_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);
    let pos_dims = arrayfire::Dim4::new(&[1, space_dims, 1, 1]);

    let mut r = arrayfire::randu::<f64>(generate_dims);
    r = (sphere_rad - neuron_rad) * arrayfire::cbrt(&r);
    let mut theta = two * (arrayfire::randu::<f64>(generate_dims) - onehalf);
    theta = arrayfire::acos(&theta);
    let mut phi = two * std::f64::consts::PI * arrayfire::randu::<f64>(generate_dims);

    let x = r.clone() * arrayfire::sin(&theta) * arrayfire::cos(&phi);
    let y = r.clone() * arrayfire::sin(&theta) * arrayfire::sin(&phi);
    let z = r.clone() * arrayfire::cos(&theta);

    drop(r);
    drop(theta);
    drop(phi);

    let mut total_obj2 = arrayfire::join_many(1, vec![&x, &y, &z]);
    drop(x);
    drop(y);
    drop(z);

    let mut tempz = arrayfire::constant::<f64>(1000000.0, arrayfire::Dim4::new(&[1, 3, 1, 1]));

    let single_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let neuron_sq = 4.0 * neuron_rad * neuron_rad;

    for i in 0u64..total_obj2.dims()[0] {
        let select_pos = arrayfire::row(&total_obj2, i as i64);

        let mut dist = arrayfire::sub(&select_pos, &total_obj2, true);
        let mut magsq = arrayfire::pow(&dist, &two, false);
        let mut magsq = arrayfire::sum(&magsq, 1);

        let insert = arrayfire::constant::<f64>(1000000.0, single_dims);

        arrayfire::set_row(&mut magsq, &insert, i as i64);

        let (m0, _) = arrayfire::min_all::<f64>(&magsq);

        //println!("{} dist {}",i, m0);
        //assert!(m0 > neuron_sq);

        if m0 > neuron_sq {
            tempz = arrayfire::join(0, &tempz, &select_pos);
        }
    }

    let total_obj_size = tempz.dims()[0];

    let split_idx = ((total_obj_size as f64) * nratio) as u64;

    *neuron_pos = arrayfire::rows(&tempz, 1, (split_idx - 1) as i64);

    *glia_pos = arrayfire::rows(&tempz, split_idx as i64, (total_obj_size - 1) as i64);
}

/*
Generates a sphere and detects cell collisions in batch. Where all cells are check at the same time

Inputs
netdata:  The sphere radius, neuron radius, mumber of neurons and glial cells to be created

Outputs:
glia_pos:    The 3D position of glial cells in the shape of a 3D sphere
neuron_pos:  The 3D position of neurons in the shape of a 3D sphere

*/

pub fn sphere_cell_collision_batch(
    netdata: &network_metadata_type,
    glia_pos: &mut arrayfire::Array<f64>,
    neuron_pos: &mut arrayfire::Array<f64>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();

    let del_unused_neuron: bool = netdata.del_unused_neuron.clone();

    let nratio: f64 = netdata.nratio.clone();
    let neuron_std: f64 = netdata.neuron_std.clone();
    let sphere_rad: f64 = netdata.sphere_rad.clone();
    let neuron_rad: f64 = netdata.neuron_rad.clone();
    let con_rad: f64 = netdata.con_rad.clone();
    let center_const: f64 = netdata.center_const.clone();
    let spring_const: f64 = netdata.spring_const.clone();
    let repel_const: f64 = netdata.repel_const.clone();

    let generate_dims = arrayfire::Dim4::new(&[2 * active_size, 1, 1, 1]);
    let single_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);
    let pos_dims = arrayfire::Dim4::new(&[1, space_dims, 1, 1]);

    let mut r = arrayfire::randu::<f64>(generate_dims);
    r = (sphere_rad - neuron_rad) * arrayfire::cbrt(&r);
    let mut theta = two * (arrayfire::randu::<f64>(generate_dims) - onehalf);
    theta = arrayfire::acos(&theta);
    let mut phi = two * std::f64::consts::PI * arrayfire::randu::<f64>(generate_dims);

    let x = r.clone() * arrayfire::sin(&theta) * arrayfire::cos(&phi);
    let y = r.clone() * arrayfire::sin(&theta) * arrayfire::sin(&phi);
    let z = r.clone() * arrayfire::cos(&theta);

    drop(r);
    drop(theta);
    drop(phi);

    let mut total_obj2 = arrayfire::join_many(1, vec![&x, &y, &z]);
    drop(x);
    drop(y);
    drop(z);

    let mut pivot_rad = ((4.0 / 3.0)
        * std::f64::consts::PI
        * TARGET_DENSITY
        * sphere_rad
        * sphere_rad
        * sphere_rad);
    pivot_rad = (pivot_rad / ((2 * active_size) as f64)).cbrt();

    let pivot_rad2 = pivot_rad + 1000.0 * sphere_rad;

    let mut loop_end_flag = false;
    let mut pivot_pos = vec![-sphere_rad; space_dims as usize];

    let single_dims = arrayfire::Dim4::new(&[1, space_dims, 1, 1]);

    let select_idx_dims = arrayfire::Dim4::new(&[total_obj2.dims()[0], 1, 1, 1]);
    let mut select_idx = arrayfire::constant::<bool>(true, select_idx_dims);

    loop {
        let idx = get_inside_idx_cubeV2(&total_obj2, pivot_rad2, &pivot_pos);

        if idx.dims()[0] > 1 {
            let tmp_obj = arrayfire::lookup(&total_obj2, &idx, 0);

            let mut neg_idx = select_non_overlap(&tmp_obj, neuron_rad);

            if neg_idx.dims()[0] > 0 {
                neg_idx = arrayfire::lookup(&idx, &neg_idx, 0);

                let insert = arrayfire::constant::<bool>(false, neg_idx.dims());

                let mut idxrs = arrayfire::Indexer::default();
                idxrs.set_index(&neg_idx, 0, None);
                arrayfire::assign_gen(&mut select_idx, &idxrs, &insert);
            }
        }
        drop(idx);

        pivot_pos[0] = pivot_pos[0] + pivot_rad;

        for idx in 0..space_dims {
            if pivot_pos[idx as usize] > sphere_rad {
                if idx == (space_dims - 1) {
                    loop_end_flag = true;
                    break;
                }

                pivot_pos[idx as usize] = -sphere_rad;
                pivot_pos[(idx + 1) as usize] = pivot_pos[(idx + 1) as usize] + pivot_rad;
            }
        }

        if loop_end_flag {
            break;
        }
    }

    let idx2 = arrayfire::locate(&select_idx);
    drop(select_idx);

    total_obj2 = arrayfire::lookup(&total_obj2, &idx2, 0);

    let total_obj_size = total_obj2.dims()[0];

    let split_idx = ((total_obj_size as f64) * nratio) as u64;

    *neuron_pos = arrayfire::rows(&total_obj2, 0, (split_idx - 1) as i64);

    *glia_pos = arrayfire::rows(&total_obj2, split_idx as i64, (total_obj_size - 1) as i64);
}
