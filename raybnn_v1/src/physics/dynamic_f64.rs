extern crate arrayfire;
use crate::neural::network_f64::network_metadata_type;

use crate::physics::distance_f64::matrix_dist;
use crate::physics::distance_f64::set_diag;

use super::distance_f64::vec_norm;

const high: f64 = 10000000.0;

const neuron_rad_factor: f64 = 1.1;
const push: f64 = 100.0;

pub fn run(
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
    let step_num: u64 = netdata.step_num.clone();

    let del_unused_neuron: bool = netdata.del_unused_neuron.clone();

    let time_step: f64 = netdata.time_step.clone();
    let nratio: f64 = netdata.nratio.clone();
    let neuron_std: f64 = netdata.neuron_std.clone();
    let sphere_rad: f64 = netdata.sphere_rad.clone();
    let neuron_rad: f64 = netdata.neuron_rad.clone();
    let con_rad: f64 = netdata.con_rad.clone();
    let center_const: f64 = netdata.center_const.clone();
    let spring_const: f64 = netdata.spring_const.clone();
    let repel_const: f64 = netdata.repel_const.clone();

    let glia_dims = glia_pos.dims();
    let neuron_dims = neuron_pos.dims();
    let pos_dims = arrayfire::Dim4::new(&[1, space_dims, 1, 1]);

    //let origin = arrayfire::constant::<f64>(0.0,pos_dims);

    let single_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);
    //let singlevec = arrayfire::constant::<f64>(high,single_dims);

    let neuron_sq: f64 = 4.0 * neuron_rad * neuron_rad * neuron_rad_factor;
    let con_sq: f64 = 4.0 * con_rad * con_rad;

    let neuron_num = neuron_dims[0];
    let glia_num = glia_dims[0];
    let total_num = neuron_num + glia_num;

    let mut total_obj = arrayfire::join(0, &neuron_pos, &glia_pos);

    let vec_tile_dims = arrayfire::Dim4::new(&[1, 1, total_num, 1]);

    let mut dist = arrayfire::constant::<f64>(0.0, single_dims);
    let mut magsq = arrayfire::constant::<f64>(0.0, single_dims);

    let mut dist2 = arrayfire::constant::<f64>(0.0, single_dims);
    let mut magsq2 = arrayfire::constant::<f64>(0.0, single_dims);

    let mut newvel = arrayfire::constant::<f64>(0.0, single_dims);
    let mut newvel2 = arrayfire::constant::<f64>(0.0, single_dims);

    let mut cmp1 = arrayfire::constant::<bool>(false, single_dims);
    let mut cmp2 = arrayfire::constant::<bool>(false, single_dims);
    let mut and1 = arrayfire::constant::<bool>(false, single_dims);
    let mut and2 = arrayfire::constant::<bool>(false, single_dims);

    let mut cmp3 = arrayfire::constant::<bool>(false, single_dims);
    let mut cmp4 = arrayfire::constant::<bool>(false, single_dims);

    let mut m0 = arrayfire::constant::<f64>(
        -1.0 * repel_const,
        arrayfire::Dim4::new(&[neuron_num, 1, neuron_num, 1]),
    );

    let m1 = arrayfire::constant::<f64>(
        spring_const,
        arrayfire::Dim4::new(&[glia_num, 1, neuron_num, 1]),
    );

    m0 = arrayfire::join(0, &m0, &m1);

    let mut m2 = arrayfire::constant::<f64>(
        spring_const,
        arrayfire::Dim4::new(&[neuron_num, 1, glia_num, 1]),
    );

    let m3 = arrayfire::constant::<f64>(
        -1.0 * repel_const,
        arrayfire::Dim4::new(&[glia_num, 1, glia_num, 1]),
    );

    m2 = arrayfire::join(0, &m2, &m3);

    m0 = arrayfire::join(2, &m0, &m2);

    let mut total_obj_norm = vec_norm(&total_obj);

    for step in 0u64..step_num {
        //Get distance matrix
        matrix_dist(&total_obj, &mut dist, &mut magsq);

        //Clone object
        magsq2 = magsq.clone();
        dist2 = dist.clone();

        set_diag(&mut magsq2, high);

        //Select close objects
        cmp3 = arrayfire::lt(&magsq2, &neuron_sq, false);

        cmp4 = arrayfire::tile(&cmp3, pos_dims);

        //let cmp1 = (magsq < con_sq);
        cmp1 = arrayfire::lt(&magsq, &con_sq, false);
        //let cmp2 = (neuron_sq < magsq );
        cmp2 = arrayfire::lt(&neuron_sq, &magsq, false);
        //let select = af::where( (cmp1*cmp2) == 1);
        and1 = arrayfire::and(&cmp1, &cmp2, false);

        and2 = arrayfire::tile(&and1, pos_dims);

        //Zero distance
        //arrayfire::replace_scalar(&mut dist, &and2 , zero);
        arrayfire::replace_scalar(&mut dist, &and2, 0.0);

        //arrayfire::replace_scalar(&mut magsq, &and1 , high);
        arrayfire::replace_scalar(&mut magsq, &and1, 100000000.0);

        //Compute attraction and repulsion
        magsq = m0.clone() / magsq;

        //magsq = arrayfire::tile(&magsq, pos_dims);

        dist = arrayfire::mul(&magsq, &dist, true);

        newvel = arrayfire::sum(&dist, 2);

        //Center velocity
        //newvel = newvel - ((center_const/(arrayfire::norm(&total_obj,arrayfire::NormType::VECTOR_2,0.0,0.0 ) as f64))*total_obj.clone());
        total_obj_norm = vec_norm(&total_obj);
        newvel = newvel - (center_const * arrayfire::div(&total_obj, &total_obj_norm, true));

        //Delete velocity going into other objects

        //Tile velocity
        newvel2 = arrayfire::tile(&newvel, vec_tile_dims);

        //Select dist2
        //arrayfire::replace_scalar(&mut dist2, &cmp4 , zero);
        arrayfire::replace_scalar(&mut dist2, &cmp4, 0.0);

        //Dot product
        newvel2 = arrayfire::mul(&newvel2, &dist2, false);
        newvel2 = arrayfire::sum(&newvel2, 1);
        newvel2 = arrayfire::abs(&newvel2);

        newvel2 = newvel2 / magsq2;
        //newvel2 = arrayfire::tile(&newvel2, pos_dims);

        //Projection
        dist2 = arrayfire::mul(&newvel2, &dist2, true);
        dist2 = arrayfire::sum(&dist2, 2);

        newvel = newvel - push * dist2;

        total_obj = total_obj + (newvel * time_step);
    }

    *neuron_pos = arrayfire::rows(&total_obj, 0, (neuron_num - 1) as i64);

    *glia_pos = arrayfire::rows(
        &total_obj,
        neuron_num as i64,
        (total_obj.dims()[0] - 1) as i64,
    );
}
