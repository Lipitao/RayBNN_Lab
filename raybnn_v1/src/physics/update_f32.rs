extern crate arrayfire;
use crate::neural::network_f32::network_metadata_type;

use nohash_hasher;
use rayon::prelude::*;
use std::collections::HashMap;

use crate::export::rand_u64::random_uniform_range;

use crate::physics::raytrace_f32::raytrace_option_type;

use crate::graph::adjacency_f32::select_forward_sphere;

use crate::physics::initial_f32::hidden_layers;
use crate::physics::initial_f32::input_and_output_layers;

use crate::physics::construct_f32::NDsphere_from_NDcube;
use crate::physics::distance_f32::sort_neuron_pos_sphere;
use crate::physics::initial_f32::assign_neuron_idx;

use crate::graph::adjacency_f32::clear_input_to_hidden;

use crate::neural::network_f32::clone_netdata;
use serde::{Deserialize, Serialize};

use crate::graph::adjacency_f32::delete_unused_neurons;

use crate::physics::initial_f32::fully_connected_hidden_layers;

use crate::graph::large_sparse_i32::COO_to_CSR;

use crate::physics::initial_f32::spherical_existingV3;

use crate::physics::initial_f32::self_loops;

use crate::physics::initial_f32::assign_self_loop_value;

use crate::physics::distance_f32::vec_min_dist;

use crate::physics::distance_f32::matrix_dist;
use crate::physics::distance_f32::set_diag;

use crate::physics::dynamic_f32::run;

use crate::graph::adjacency_f32::clear_input;
use crate::graph::adjacency_f32::clear_output;

use crate::graph::adjacency_f32::get_global_weight_idx;
use crate::graph::large_sparse_u64::COO_batch_find;

use crate::graph::large_sparse_i32::CSR_to_COO;

use super::distance_f32::vec_norm;

//use crate::graph::adjacency_f32::delete_smallest_weights;
//use crate::graph::adjacency_f32::delete_neurons_at_idx;
//use crate::graph::adjacency_f32::delete_smallest_neurons;
use crate::graph::adjacency_f32::delete_smallest_neurons_with_prob;
use crate::graph::adjacency_f32::delete_weights_with_prob;

use crate::physics::raytrace_f32::RT3_distance_limited_directly_connected;

use crate::neural::network_f32::neural_network_type;

use crate::interface::automatic_f32::arch_search_type;

use crate::neural::network_f32::xavier_init;

use crate::physics::initial_f32::assign_neuron_idx_with_buffer;

const INOUT_FACTOR: f32 = 0.15;

const input_factor: f32 = 0.05;
const output_factor: f32 = 0.09;

const sphere_rad_factor: f32 = 1.1;
const con_rad_factor: f32 = 1.4;
const neuron_rad_factor: f32 = 1.14;
const high: f32 = 10000000.0;
const zero: bool = false;

const COO_find_limit: u64 = 1500000000;

pub fn create_objects(
    netdata: &network_metadata_type,
    glia_pos: &arrayfire::Array<f32>,
    neuron_pos: &arrayfire::Array<f32>,
    new_object_num: u64,

    new_glia_pos: &mut arrayfire::Array<f32>,
    new_neuron_pos: &mut arrayfire::Array<f32>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();

    let del_unused_neuron: bool = netdata.del_unused_neuron.clone();

    let time_step: f32 = netdata.time_step.clone();
    let nratio: f32 = netdata.nratio.clone();
    let neuron_std: f32 = netdata.neuron_std.clone();
    let sphere_rad: f32 = netdata.sphere_rad.clone();
    let neuron_rad: f32 = netdata.neuron_rad.clone();
    let con_rad: f32 = netdata.con_rad.clone();
    let center_const: f32 = netdata.center_const.clone();
    let spring_const: f32 = netdata.spring_const.clone();
    let repel_const: f32 = netdata.repel_const.clone();

    let glia_dims = glia_pos.dims();
    let neuron_dims = neuron_pos.dims();
    let pos_dims = arrayfire::Dim4::new(&[1, space_dims, 1, 1]);

    let single_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);
    let singlevec = arrayfire::constant::<f32>(1000000.0, single_dims);

    let neuron_sq: f32 = 4.0 * neuron_rad * neuron_rad * neuron_rad_factor;
    let con_sq: f32 = 4.0 * con_rad * con_rad;

    let neuron_num = neuron_dims[0];
    let glia_num = glia_dims[0];

    let mut total_num: i64 = (neuron_num + glia_num) as i64;
    let mut total_neurons = arrayfire::join::<f32>(0, &glia_pos, &neuron_pos);

    let mut new_pos = arrayfire::randn::<f32>(pos_dims);

    let gen_rad = con_rad / 2.0;

    let mut mdist: f32 = 0.0;
    while (1 == 1) {
        let rand_row_num = random_uniform_range(total_num as u64);

        let sel_pos = arrayfire::row(&total_neurons, rand_row_num as i64);
        new_pos = sel_pos + (gen_rad) * arrayfire::randn::<f32>(pos_dims);

        mdist = vec_min_dist(&new_pos, &total_neurons);

        if (neuron_sq < mdist) && (mdist < con_sq) {
            break;
        }
    }

    total_neurons = arrayfire::join::<f32>(0, &total_neurons, &new_pos);
    //total_num = total_num + 1;

    let mut new_neurons = new_pos.clone();

    for j in 0u64..(new_object_num - 1) {
        mdist = 0.0;
        while (1 == 1) {
            let rand_row_num = random_uniform_range(total_num as u64);

            let sel_pos = arrayfire::row(&total_neurons, rand_row_num as i64);
            new_pos = sel_pos + (gen_rad) * arrayfire::randn::<f32>(pos_dims);

            mdist = vec_min_dist(&new_pos, &total_neurons);

            if (neuron_sq < mdist) && (mdist < con_sq) {
                break;
            }
        }
        total_neurons = arrayfire::join::<f32>(0, &total_neurons, &new_pos);
        //total_num = total_num + 1;

        new_neurons = arrayfire::join::<f32>(0, &new_neurons, &new_pos);
    }

    let rand_dims = arrayfire::Dim4::new(&[new_neurons.dims()[0], 1, 1, 1]);

    let randarr = arrayfire::randu::<f32>(rand_dims);
    let c1 = arrayfire::le(&nratio, &randarr, false);
    let c2 = arrayfire::gt(&nratio, &randarr, false);
    let idx1 = arrayfire::locate(&c1);
    let idx2 = arrayfire::locate(&c2);

    let mut idxrs1 = arrayfire::Indexer::default();
    let seq1 = arrayfire::Seq::new(0.0, (space_dims - 1) as f32, 1.0);
    idxrs1.set_index(&idx1, 0, None);
    idxrs1.set_index(&seq1, 1, Some(false));
    *new_glia_pos = arrayfire::index_gen(&new_neurons, idxrs1);
    //*glia_pos = arrayfire::join::<f32>(0,glia_pos,&new_glia_pos);

    let mut idxrs2 = arrayfire::Indexer::default();
    let seq2 = arrayfire::Seq::new(0.0, (space_dims - 1) as f32, 1.0);
    idxrs2.set_index(&idx2, 0, None);
    idxrs2.set_index(&seq2, 1, Some(false));
    *new_neuron_pos = arrayfire::index_gen(&new_neurons, idxrs2);
    //*neuron_pos = arrayfire::join::<f32>(0,neuron_pos,&new_neuron_pos);
}

pub fn reposition(
    netdata: &network_metadata_type,
    glia_pos: &mut arrayfire::Array<f32>,
    neuron_pos: &mut arrayfire::Array<f32>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();

    let time_step: f32 = netdata.time_step.clone();
    let nratio: f32 = netdata.nratio.clone();
    let neuron_std: f32 = netdata.neuron_std.clone();
    let sphere_rad: f32 = netdata.sphere_rad.clone();
    let neuron_rad: f32 = netdata.neuron_rad.clone();
    let con_rad: f32 = netdata.con_rad.clone();
    let center_const: f32 = netdata.center_const.clone();
    let spring_const: f32 = netdata.spring_const.clone();
    let repel_const: f32 = netdata.repel_const.clone();

    let pos_dims = arrayfire::Dim4::new(&[1, space_dims, 1, 1]);

    let glia_dims = glia_pos.dims();
    let neuron_dims = neuron_pos.dims();

    let singlevec = arrayfire::constant::<f32>(1000000.0, pos_dims);

    let neuron_sq: f32 = 4.0 * neuron_rad * neuron_rad * neuron_rad_factor;
    let con_sq: f32 = 4.0 * con_rad * con_rad;

    let neuron_num: i64 = neuron_dims[0] as i64;

    let mut total_neurons = arrayfire::join::<f32>(0, &neuron_pos, &glia_pos);

    let mut new_pos = arrayfire::randn::<f32>(pos_dims);

    let gen_rad = con_rad / 2.0;

    let single_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let mut dist = arrayfire::constant::<f32>(0.0, single_dims);
    let mut magsq = arrayfire::constant::<f32>(0.0, single_dims);

    matrix_dist(&total_neurons, &mut dist, &mut magsq);

    set_diag(&mut magsq, high);

    let minarr = arrayfire::min(&magsq, 2);

    //let cmp1 = (con_sq < minarr );
    let cmp1 = arrayfire::lt(&con_sq, &minarr, false);
    let idx1 = arrayfire::locate(&cmp1);

    let idx1num = idx1.dims()[0];

    if (idx1num == 0) {
        return;
    }

    let mut idx1_cpu = vec![u32::default(); idx1.elements()];
    idx1.host(&mut idx1_cpu);

    let cmp2 = arrayfire::eq(&cmp1, &zero, false);
    let idx2 = arrayfire::locate(&cmp2);

    let mut idxrs = arrayfire::Indexer::default();
    let cols = arrayfire::Seq::new(0.0, (space_dims - 1) as f32, 1.0);
    idxrs.set_index(&idx2, 0, None);
    idxrs.set_index(&cols, 1, Some(false));
    let selected = arrayfire::index_gen(&total_neurons, idxrs);
    let selnum: i64 = selected.dims()[0] as i64;

    let mut i: i64 = 0;
    let mut mdist: f32 = 100000.0;
    let mut sel_pos = arrayfire::row(&total_neurons, 0);
    for vv in &idx1_cpu {
        i = vv.clone() as i64;

        while (1 == 1) {
            let rand_row_num = random_uniform_range(selnum as u64);

            sel_pos = arrayfire::row(&selected, rand_row_num as i64);
            new_pos = sel_pos + (gen_rad) * arrayfire::randn::<f32>(pos_dims);

            mdist = vec_min_dist(&new_pos, &total_neurons);

            if (neuron_sq < mdist) && (mdist < con_sq) {
                break;
            }
        }

        if (i < neuron_num) {
            arrayfire::set_row(neuron_pos, &new_pos, i);
        } else {
            arrayfire::set_row(glia_pos, &new_pos, (i - neuron_num));
        }

        arrayfire::set_row(&mut total_neurons, &new_pos, i);
    }
}

/*
pub fn create_connections(
    netdata: &network_metadata_type,
    neuron_pos: &arrayfire::Array<f32>,
    neuron_idx: &arrayfire::Array<i32>,

    newWValues: &mut arrayfire::Array<f32>,
    newWRowIdxCOO: &mut arrayfire::Array<i32>,
    newWColIdx: &mut arrayfire::Array<i32>
)
{
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();




    let time_step: f32 = netdata.time_step.clone();
    let nratio: f32 = netdata.nratio.clone();
    let neuron_std: f32 = netdata.neuron_std.clone();
    let sphere_rad: f32 = netdata.sphere_rad.clone();
    let neuron_rad: f32 = netdata.neuron_rad.clone();
    let con_rad: f32 = netdata.con_rad.clone();
    let mut init_prob: f32 = netdata.init_prob.clone();
    let center_const: f32 = netdata.center_const.clone();
    let spring_const: f32 = netdata.spring_const.clone();
    let repel_const: f32 = netdata.repel_const.clone();







    let neuron_dims = neuron_pos.dims();
    let neuron_num = neuron_dims[0];

    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);



    let neuron_sq: f32 = 4.0*neuron_rad*neuron_rad;
    let con_sq: f32 = 4.0*con_rad*con_rad;


    let alpha: f32 =  ((0.01 as f32).ln())/(con_sq-neuron_sq);




    let mut dist = arrayfire::constant::<f32>(0.0,single_dims);
    let mut magsq = arrayfire::constant::<f32>(0.0,single_dims);

    matrix_dist(
        &neuron_pos,
        &mut dist,
        &mut magsq
    );

    set_diag(
        &mut magsq,
        high
    );



    //Get neurons in con_sq
    let mut cmp1 = arrayfire::lt(&magsq , &con_sq, false);
    cmp1 = arrayfire::flat(&cmp1);
    let sel1 = arrayfire::locate(&cmp1);





    magsq = arrayfire::flat(&magsq);
    let mut idxrs1 = arrayfire::Indexer::default();
    idxrs1.set_index(&sel1, 0, None);
    let mut mg = arrayfire::index_gen(&magsq, idxrs1);

    mg = (mg-neuron_sq)*alpha;
    mg = init_prob*arrayfire::exp(&mg);

    let randarr = arrayfire::randu::<f32>(mg.dims());

    let cmp2 = arrayfire::lt(&randarr , &mg, false);
    let sel2 = arrayfire::locate(&cmp2);

    let sel2_dims = sel2.dims()[0];





    let mut idxrs2 = arrayfire::Indexer::default();
    idxrs2.set_index(&sel2, 0, None);
    let mut selidx = arrayfire::index_gen(&sel1, idxrs2);

    selidx = arrayfire::set_unique(&selidx, false);


    *newWValues = neuron_std*arrayfire::randn::<f32>(selidx.dims());







    let col = arrayfire::modulo(&selidx,&neuron_num,false);


    let mut idxrs3 = arrayfire::Indexer::default();
    idxrs3.set_index(&col, 0, None);
    *newWColIdx = arrayfire::index_gen(neuron_idx, idxrs3);







    let row = arrayfire::div(&selidx, &neuron_num,false);


    let mut idxrs4 = arrayfire::Indexer::default();
    idxrs4.set_index(&row, 0, None);
    *newWRowIdxCOO = arrayfire::index_gen(neuron_idx, idxrs4);




}
*/

pub fn add_neuron_to_existing(
    cube_pos: &arrayfire::Array<f32>,
    cube_radius: f32,

    new_neuron_num: u64,

    netdata: &mut network_metadata_type,
    WValues: &mut arrayfire::Array<f32>,
    WRowIdxCOO: &mut arrayfire::Array<i32>,
    WColIdx: &mut arrayfire::Array<i32>,

    neuron_pos: &mut arrayfire::Array<f32>,
    neuron_idx: &mut arrayfire::Array<i32>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();
    let batch_size: u64 = netdata.batch_size.clone();

    let del_unused_neuron: bool = netdata.del_unused_neuron.clone();

    let time_step: f32 = netdata.time_step.clone();
    let nratio: f32 = netdata.nratio.clone();
    let neuron_std: f32 = netdata.neuron_std.clone();
    let sphere_rad: f32 = netdata.sphere_rad.clone();
    let neuron_rad: f32 = netdata.neuron_rad.clone();
    let con_rad: f32 = netdata.con_rad.clone();
    let init_prob: f32 = netdata.init_prob.clone();
    let add_neuron_rate: f32 = netdata.add_neuron_rate.clone();
    let del_neuron_rate: f32 = netdata.del_neuron_rate.clone();
    let center_const: f32 = netdata.center_const.clone();
    let spring_const: f32 = netdata.spring_const.clone();
    let repel_const: f32 = netdata.repel_const.clone();

    //let COO_batch_size = 1 + ((COO_find_limit/WRowIdxCOO.dims()[0]) as u64);

    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);
    //let mut new_glia_pos = arrayfire::constant::<f32>(0.0,temp_dims);
    //let mut new_neuron_pos = arrayfire::constant::<f32>(0.0,temp_dims);

    let mut newWValues = arrayfire::constant::<f32>(0.0, temp_dims);
    let mut newWRowIdxCOO = arrayfire::constant::<i32>(0, temp_dims);
    let mut newWColIdx = arrayfire::constant::<i32>(0, temp_dims);

    let mut gidx1 = arrayfire::constant::<u64>(0, temp_dims);
    let mut gidx2 = arrayfire::constant::<u64>(0, temp_dims);
    let mut gidx3 = arrayfire::constant::<u64>(0, temp_dims);
    let mut gidx4 = arrayfire::constant::<u64>(0, temp_dims);

    let mut active_size = neuron_idx.dims()[0];
    let mut hidden_size = active_size - output_size - input_size;

    //Get input neurons
    let mut input_neurons = arrayfire::rows(neuron_pos, 0, (input_size - 1) as i64);

    let output_neurons = arrayfire::rows(
        neuron_pos,
        (active_size - output_size) as i64,
        (active_size - 1) as i64,
    );

    //Create new neurons from template
    let mut initial_sphere_radius = sphere_rad.clone();
    loop {
        *neuron_pos = NDsphere_from_NDcube(cube_pos, cube_radius, initial_sphere_radius);

        if (neuron_pos.dims()[0] < (hidden_size + new_neuron_num)) {
            initial_sphere_radius = initial_sphere_radius * sphere_rad_factor;
        } else {
            break;
        }
    }
    *neuron_pos = arrayfire::rows(neuron_pos, 0, (hidden_size + new_neuron_num - 1) as i64);

    sort_neuron_pos_sphere(neuron_pos);

    let sq = vec_norm(neuron_pos);

    let (m0, _) = arrayfire::max_all::<f32>(&sq);
    (*netdata).sphere_rad = m0;
    (*netdata).con_rad = (m0 / (proc_num as f32)) * con_rad_factor;

    //Reconstruct
    input_neurons = ((m0 + 0.2) / sphere_rad) * input_neurons;

    *neuron_pos = arrayfire::join(0, &input_neurons, neuron_pos);

    *neuron_pos = arrayfire::join(0, neuron_pos, &output_neurons);

    //Assign
    assign_neuron_idx(netdata, neuron_pos, neuron_idx);

    clear_input_to_hidden(WValues, WRowIdxCOO, WColIdx, input_size);

    /*
    //Get hidden neurons
    let mut temp_hidden_pos = arrayfire::rows(&neuron_pos, input_size as i64, ((active_size-output_size)-1) as i64);

    //Create new neurons
    create_objects(
        netdata,
        glia_pos,
        &mut temp_hidden_pos,
        new_object_num,

        &mut new_glia_pos,
        &mut new_neuron_pos
    );

    let new_neuron_pos_num = new_neuron_pos.dims()[0];

    if (new_neuron_pos_num + active_size) > neuron_size
    {
        new_neuron_pos = arrayfire::rows(&new_neuron_pos,0,(neuron_size - active_size -1) as i64);
    }


    //Add new to neurons
    *glia_pos = arrayfire::join::<f32>(0,glia_pos,&new_glia_pos);

    temp_hidden_pos = arrayfire::join::<f32>(0,&temp_hidden_pos,&new_neuron_pos);





    //Run simulation
    (*netdata).step_num = step_num*((new_neuron_pos_num+hidden_size)/hidden_size);

    run(
        netdata,
        glia_pos,
        &mut temp_hidden_pos);


    reposition(
        netdata,
        glia_pos,
        &mut temp_hidden_pos
    );


    //Find furthest neuron radius
    //let mut sq = arrayfire::pow(&temp_hidden_pos,&two,false);
    //sq = arrayfire::sum(&sq, 1);
    //sq = arrayfire::sqrt(&sq);


    let sq = vec_norm(&temp_hidden_pos);

    let (m0,_) = arrayfire::max_all::<f32>(&sq);
    (*netdata).sphere_rad = m0;
    (*netdata).con_rad = (m0/(proc_num as f32))*con_rad_factor;


    //Reconstruct
    let input_neurons = ((m0+0.2)/sphere_rad)*arrayfire::rows(&neuron_pos, 0, (input_size-1)  as i64);

    let output_neurons = arrayfire::rows(&neuron_pos, (active_size-output_size) as i64, (active_size-1)  as i64);

    *neuron_pos = arrayfire::join(0, &input_neurons, &temp_hidden_pos);

    *neuron_pos = arrayfire::join(0, neuron_pos, &output_neurons);






    //Reconstruct
    let input_neuron_idx = arrayfire::rows(neuron_idx,0, ((active_size-output_size)-1)  as i64);

    let output_neuron_idx = arrayfire::rows(neuron_idx,(active_size-output_size) as i64, (active_size-1)  as i64);


    let repeat_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    let tile_dims = arrayfire::Dim4::new(&[new_neuron_pos.dims()[0],1,1,1]);

    let new_neuron_idx = ((active_size-output_size) as i32) + arrayfire::iota::<i32>(tile_dims,repeat_dims);

    *neuron_idx = arrayfire::join(0, &input_neuron_idx, &new_neuron_idx);

    *neuron_idx = arrayfire::join(0, neuron_idx, &output_neuron_idx);

    */

    /*
    //Create new connections
    create_connections(
        netdata,
        neuron_pos,
        neuron_idx,

        &mut newWValues,
        &mut newWRowIdxCOO,
        &mut newWColIdx
    );
    */

    let mut hidden_neuron2 = neuron_pos.dims()[0] - output_size - input_size;

    if (neuron_pos.dims()[0] <= (output_size + input_size)) {
        hidden_neuron2 = 1;
    }

    let mut input_connections = (input_factor * (hidden_neuron2 as f32)) as u64;
    let mut output_connections = (output_factor * (hidden_neuron2 as f32)) as u64;

    if (input_connections <= 1) {
        input_connections = 1;
    }

    if (output_connections <= 1) {
        output_connections = 1;
    }

    input_and_output_layers(
        &netdata,
        &neuron_pos,
        &neuron_idx,
        input_connections,
        output_connections,
        &mut newWValues,
        &mut newWRowIdxCOO,
        &mut newWColIdx,
    );

    hidden_layers(
        &netdata,
        &neuron_pos,
        &neuron_idx,
        &mut newWValues,
        &mut newWRowIdxCOO,
        &mut newWColIdx,
    );

    //Clear input/output
    clear_input(
        &mut newWValues,
        &mut newWRowIdxCOO,
        &mut newWColIdx,
        input_size,
    );

    clear_output(
        &mut newWValues,
        &mut newWRowIdxCOO,
        &mut newWColIdx,
        neuron_size - output_size,
    );

    //Compute global index
    gidx1 = get_global_weight_idx(neuron_size, WRowIdxCOO, WColIdx);

    gidx2 = get_global_weight_idx(neuron_size, &newWRowIdxCOO, &newWColIdx);

    let gidx1dims = gidx1.dims()[0];
    let gidx2dims = (gidx2.dims()[0]).max(gidx1dims);

    let COO_batch_size = 1 + ((COO_find_limit / gidx2dims) as u64);

    gidx3 = COO_batch_find(&gidx2, &gidx1, COO_batch_size);

    //Filter out existing connections
    if gidx3.dims()[0] > 0 {
        let mut table = arrayfire::constant::<bool>(true, gidx2.dims());

        let inarr = arrayfire::constant::<bool>(false, gidx3.dims());

        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&gidx3, 0, None);
        arrayfire::assign_gen(&mut table, &idxrs, &inarr);

        let tempidx = arrayfire::locate(&table);

        newWValues = arrayfire::lookup(&newWValues, &tempidx, 0);
        newWRowIdxCOO = arrayfire::lookup(&newWRowIdxCOO, &tempidx, 0);
        newWColIdx = arrayfire::lookup(&newWColIdx, &tempidx, 0);
        gidx2 = arrayfire::lookup(&gidx2, &tempidx, 0);
    }

    //Insert new connections
    gidx4 = arrayfire::join(0, &gidx1, &gidx2);
    *WValues = arrayfire::join(0, WValues, &newWValues);
    *WRowIdxCOO = arrayfire::join(0, WRowIdxCOO, &newWRowIdxCOO);
    *WColIdx = arrayfire::join(0, WColIdx, &newWColIdx);

    let (_, idx) = arrayfire::sort_index(&gidx4, 0, true);

    *WValues = arrayfire::lookup(WValues, &idx, 0);
    *WRowIdxCOO = arrayfire::lookup(WRowIdxCOO, &idx, 0);
    *WColIdx = arrayfire::lookup(WColIdx, &idx, 0);

    (*netdata).active_size = neuron_idx.dims()[0];
}

pub fn reduce_network_size(new_active_size: u64, arch_search: &mut arch_search_type) {
    let mut input_size = (*arch_search).neural_network.netdata.input_size;
    let mut output_size = (*arch_search).neural_network.netdata.output_size;

    let min_prune_num = (*arch_search).evolution_info.min_prune_num;

    let mut active_size = (*arch_search).neural_network.neuron_idx.dims()[0];
    let mut con_num = (*arch_search).neural_network.WColIdx.dims()[0];
    let mut del_num = 0;

    let neuron_size: u64 = (*arch_search).neural_network.netdata.neuron_size.clone();

    let WValuesdims0 = (*arch_search).neural_network.WColIdx.dims()[0];

    let network_paramsdims0 = (*arch_search).neural_network.network_params.dims()[0];

    let Hdims0 = (network_paramsdims0 - WValuesdims0) / 6;

    let Wstart = 0;
    let Wend = (WValuesdims0 as i64) - 1;

    let Hstart = Wend + 1;
    let Hend = Hstart + (Hdims0 as i64) - 1;

    let Astart = Hend + 1;
    let Aend = Astart + (Hdims0 as i64) - 1;

    let Bstart = Aend + 1;
    let Bend = Bstart + (Hdims0 as i64) - 1;

    let Cstart = Bend + 1;
    let Cend = Cstart + (Hdims0 as i64) - 1;

    let Dstart = Cend + 1;
    let Dend = Dstart + (Hdims0 as i64) - 1;

    let Estart = Dend + 1;
    let Eend = Estart + (Hdims0 as i64) - 1;

    let Wseqs = [arrayfire::Seq::new(Wstart as i32, Wend as i32, 1i32)];
    let Hseqs = [arrayfire::Seq::new(Hstart as i32, Hend as i32, 1i32)];
    let Aseqs = [arrayfire::Seq::new(Astart as i32, Aend as i32, 1i32)];
    let Bseqs = [arrayfire::Seq::new(Bstart as i32, Bend as i32, 1i32)];
    let Cseqs = [arrayfire::Seq::new(Cstart as i32, Cend as i32, 1i32)];
    let Dseqs = [arrayfire::Seq::new(Dstart as i32, Dend as i32, 1i32)];
    let Eseqs = [arrayfire::Seq::new(Estart as i32, Eend as i32, 1i32)];

    let mut WValues = arrayfire::index(&((*arch_search).neural_network.network_params), &Wseqs);
    let H = arrayfire::index(&((*arch_search).neural_network.network_params), &Hseqs);
    let A = arrayfire::index(&((*arch_search).neural_network.network_params), &Aseqs);
    let B = arrayfire::index(&((*arch_search).neural_network.network_params), &Bseqs);
    let C = arrayfire::index(&((*arch_search).neural_network.network_params), &Cseqs);
    let D = arrayfire::index(&((*arch_search).neural_network.network_params), &Dseqs);
    let E = arrayfire::index(&((*arch_search).neural_network.network_params), &Eseqs);

    let mut WRowIdxCOO = CSR_to_COO(&((*arch_search).neural_network.WRowIdxCSR));

    let mut hiddennum = active_size - output_size - input_size;

    while new_active_size < active_size {
        hiddennum = active_size - output_size - input_size;
        del_num = ((hiddennum as f32) * min_prune_num) as u64;

        if del_num <= 10 {
            del_num = 1;
        }

        /*
        delete_smallest_neurons(
            &((*arch_search).neural_network.netdata),
            &((*arch_search).neural_network.neuron_idx),
            del_num,

            &mut WValues,
            &mut WRowIdxCOO,
            &mut ((*arch_search).neural_network.WColIdx),
        );
        */

        delete_smallest_neurons_with_prob(
            &((*arch_search).neural_network.netdata),
            &((*arch_search).neural_network.neuron_idx),
            del_num,
            &mut WValues,
            &mut WRowIdxCOO,
            &mut ((*arch_search).neural_network.WColIdx),
        );

        con_num = (*arch_search).neural_network.WColIdx.dims()[0];
        del_num = ((con_num as f32) * min_prune_num) as u64;

        if del_num <= 10 {
            del_num = 1;
        }

        /*
        delete_smallest_weights(
            &mut WValues,
            &mut WRowIdxCOO,
            &mut ((*arch_search).neural_network.WColIdx),
            del_num
        );
        */

        delete_weights_with_prob(
            &mut WValues,
            &mut WRowIdxCOO,
            &mut ((*arch_search).neural_network.WColIdx),
            del_num,
        );

        delete_unused_neurons(
            &((*arch_search).neural_network.netdata),
            &mut WValues,
            &mut WRowIdxCOO,
            &mut ((*arch_search).neural_network.WColIdx),
            &mut ((*arch_search).neural_network.glia_pos),
            &mut ((*arch_search).neural_network.neuron_pos),
            &mut ((*arch_search).neural_network.neuron_idx),
        );

        active_size = (*arch_search).neural_network.neuron_idx.dims()[0];
    }

    //Save values
    (*arch_search).neural_network.WRowIdxCSR = COO_to_CSR(&WRowIdxCOO, neuron_size);

    let total_param_size = WValues.dims()[0]
        + H.dims()[0]
        + A.dims()[0]
        + B.dims()[0]
        + C.dims()[0]
        + D.dims()[0]
        + E.dims()[0];
    let network_params_dims = arrayfire::Dim4::new(&[total_param_size, 1, 1, 1]);

    let Wstart = 0;
    let Wend = (WValues.dims()[0] as i64) - 1;

    let Hstart = Wend + 1;
    let Hend = Hstart + (H.dims()[0] as i64) - 1;

    let Astart = Hend + 1;
    let Aend = Astart + (A.dims()[0] as i64) - 1;

    let Bstart = Aend + 1;
    let Bend = Bstart + (B.dims()[0] as i64) - 1;

    let Cstart = Bend + 1;
    let Cend = Cstart + (C.dims()[0] as i64) - 1;

    let Dstart = Cend + 1;
    let Dend = Dstart + (D.dims()[0] as i64) - 1;

    let Estart = Dend + 1;
    let Eend = Estart + (E.dims()[0] as i64) - 1;

    let Wseqs = [arrayfire::Seq::new(Wstart as i32, Wend as i32, 1i32)];
    let Hseqs = [arrayfire::Seq::new(Hstart as i32, Hend as i32, 1i32)];
    let Aseqs = [arrayfire::Seq::new(Astart as i32, Aend as i32, 1i32)];
    let Bseqs = [arrayfire::Seq::new(Bstart as i32, Bend as i32, 1i32)];
    let Cseqs = [arrayfire::Seq::new(Cstart as i32, Cend as i32, 1i32)];
    let Dseqs = [arrayfire::Seq::new(Dstart as i32, Dend as i32, 1i32)];
    let Eseqs = [arrayfire::Seq::new(Estart as i32, Eend as i32, 1i32)];

    (*arch_search).neural_network.network_params =
        arrayfire::constant::<f32>(0.0, network_params_dims);
    arrayfire::assign_seq(
        &mut ((*arch_search).neural_network.network_params),
        &Wseqs,
        &WValues,
    );
    arrayfire::assign_seq(
        &mut ((*arch_search).neural_network.network_params),
        &Hseqs,
        &H,
    );
    arrayfire::assign_seq(
        &mut ((*arch_search).neural_network.network_params),
        &Aseqs,
        &A,
    );
    arrayfire::assign_seq(
        &mut ((*arch_search).neural_network.network_params),
        &Bseqs,
        &B,
    );
    arrayfire::assign_seq(
        &mut ((*arch_search).neural_network.network_params),
        &Cseqs,
        &C,
    );
    arrayfire::assign_seq(
        &mut ((*arch_search).neural_network.network_params),
        &Dseqs,
        &D,
    );
    arrayfire::assign_seq(
        &mut ((*arch_search).neural_network.network_params),
        &Eseqs,
        &E,
    );

    (*arch_search).neural_network.netdata.active_size =
        (*arch_search).neural_network.neuron_idx.dims()[0];
}

#[derive(Serialize, Deserialize)]
pub struct add_neuron_option_type {
    pub new_active_size: u64,
    pub init_connection_num: u64,
    pub input_neuron_con_rad: f32,
    pub hidden_neuron_con_rad: f32,
    pub output_neuron_con_rad: f32,
}

pub fn add_neuron_to_existing3(
    add_neuron_options: &add_neuron_option_type,

    arch_search: &mut arch_search_type,
) {
    let new_active_size = add_neuron_options.new_active_size.clone();
    let init_connection_num = add_neuron_options.init_connection_num.clone();
    let input_neuron_con_rad = add_neuron_options.input_neuron_con_rad.clone();
    let hidden_neuron_con_rad = add_neuron_options.hidden_neuron_con_rad.clone();
    let output_neuron_con_rad = add_neuron_options.output_neuron_con_rad.clone();

    let neuron_size: u64 = (*arch_search).neural_network.netdata.neuron_size.clone();
    let input_size: u64 = (*arch_search).neural_network.netdata.input_size.clone();
    let output_size: u64 = (*arch_search).neural_network.netdata.output_size.clone();
    let proc_num: u64 = (*arch_search).neural_network.netdata.proc_num.clone();
    let mut active_size: u64 = (*arch_search).neural_network.netdata.active_size.clone();
    let space_dims: u64 = (*arch_search).neural_network.netdata.space_dims.clone();
    let step_num: u64 = (*arch_search).neural_network.netdata.step_num.clone();
    let batch_size: u64 = (*arch_search).neural_network.netdata.batch_size.clone();

    let del_unused_neuron: bool = (*arch_search)
        .neural_network
        .netdata
        .del_unused_neuron
        .clone();

    let time_step: f32 = (*arch_search).neural_network.netdata.time_step.clone();
    let nratio: f32 = (*arch_search).neural_network.netdata.nratio.clone();
    let neuron_std: f32 = (*arch_search).neural_network.netdata.neuron_std.clone();
    let sphere_rad: f32 = (*arch_search).neural_network.netdata.sphere_rad.clone();
    let neuron_rad: f32 = (*arch_search).neural_network.netdata.neuron_rad.clone();
    let con_rad: f32 = (*arch_search).neural_network.netdata.con_rad.clone();
    let init_prob: f32 = (*arch_search).neural_network.netdata.init_prob.clone();
    let add_neuron_rate: f32 = (*arch_search)
        .neural_network
        .netdata
        .add_neuron_rate
        .clone();
    let del_neuron_rate: f32 = (*arch_search)
        .neural_network
        .netdata
        .del_neuron_rate
        .clone();
    let center_const: f32 = (*arch_search).neural_network.netdata.center_const.clone();
    let spring_const: f32 = (*arch_search).neural_network.netdata.spring_const.clone();
    let repel_const: f32 = (*arch_search).neural_network.netdata.repel_const.clone();

    let mut WRowIdxCOO = CSR_to_COO(&((*arch_search).neural_network.WRowIdxCSR));

    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

    let mut newWValues = arrayfire::constant::<f32>(0.0, temp_dims);
    let mut newWRowIdxCOO = arrayfire::constant::<i32>(0, temp_dims);
    let mut newWColIdx = arrayfire::constant::<i32>(0, temp_dims);

    let mut gidx1 = arrayfire::constant::<u64>(0, temp_dims);
    let mut gidx2 = arrayfire::constant::<u64>(0, temp_dims);
    let mut gidx3 = arrayfire::constant::<u64>(0, temp_dims);
    let mut gidx4 = arrayfire::constant::<u64>(0, temp_dims);

    let mut active_size = (*arch_search).neural_network.neuron_idx.dims()[0];
    let mut hidden_size = active_size - output_size - input_size;

    //Get input neurons
    let mut input_neurons = arrayfire::rows(
        &((*arch_search).neural_network.neuron_pos),
        0,
        (input_size - 1) as i64,
    );

    let output_neurons = arrayfire::rows(
        &((*arch_search).neural_network.neuron_pos),
        (active_size - output_size) as i64,
        (active_size - 1) as i64,
    );

    (*arch_search).neural_network.neuron_pos = arrayfire::rows(
        &((*arch_search).neural_network.neuron_pos),
        input_size as i64,
        (active_size - output_size - 1) as i64,
    );

    let mut current_volume = sphere_rad * sphere_rad * sphere_rad;

    let density = current_volume / (active_size as f32);

    let new_scale = (density * ((active_size + new_active_size) as f32)).cbrt() / sphere_rad;

    //Scale to new size
    (*arch_search).neural_network.neuron_pos =
        (*arch_search).neural_network.neuron_pos.clone() * new_scale;
    (*arch_search).neural_network.glia_pos =
        (*arch_search).neural_network.glia_pos.clone() * new_scale;
    (*arch_search).neural_network.netdata.sphere_rad =
        (*arch_search).neural_network.netdata.sphere_rad * new_scale;
    input_neurons = input_neurons.clone() * new_scale;

    (*arch_search).neural_network.netdata.active_size = new_active_size;

    spherical_existingV3(
        &((*arch_search).neural_network.netdata),
        &mut ((*arch_search).neural_network.glia_pos),
        &mut ((*arch_search).neural_network.neuron_pos),
    );

    if (*arch_search).neural_network.neuron_pos.dims()[0] > (active_size + new_active_size) {
        (*arch_search).neural_network.neuron_pos = arrayfire::rows(
            &((*arch_search).neural_network.neuron_pos),
            0,
            (active_size + new_active_size - 1) as i64,
        );
    }

    (*arch_search).neural_network.netdata.active_size =
        (*arch_search).neural_network.neuron_pos.dims()[0];
    active_size = (*arch_search).neural_network.neuron_pos.dims()[0];

    //sort_neuron_pos_sphere(&mut ((*arch_search).neural_network.neuron_pos));

    let sq = vec_norm(&((*arch_search).neural_network.neuron_pos));

    let (m0, _) = arrayfire::max_all::<f32>(&sq);
    (*arch_search).neural_network.netdata.sphere_rad = m0;
    (*arch_search).neural_network.netdata.con_rad = (m0 / (proc_num as f32)) * con_rad_factor;

    //Reconstruct
    //input_neurons = ((m0 + (neuron_rad*2.0f32)  )/sphere_rad)*input_neurons;

    (*arch_search).neural_network.neuron_pos = arrayfire::join(
        0,
        &input_neurons,
        &((*arch_search).neural_network.neuron_pos),
    );

    (*arch_search).neural_network.neuron_pos = arrayfire::join(
        0,
        &((*arch_search).neural_network.neuron_pos),
        &output_neurons,
    );

    if (*arch_search).neural_network.glia_pos.dims()[0]
        > (*arch_search).neural_network.neuron_pos.dims()[0]
    {
        (*arch_search).neural_network.glia_pos = arrayfire::rows(
            &((*arch_search).neural_network.glia_pos),
            0,
            ((*arch_search).neural_network.neuron_pos.dims()[0] - 1) as i64,
        );
    }

    assign_neuron_idx_with_buffer(
        (*arch_search).evolution_info.max_input_size,
        (*arch_search).evolution_info.max_output_size,
        &((*arch_search).neural_network.netdata),
        &((*arch_search).neural_network.neuron_pos),
        &mut ((*arch_search).neural_network.neuron_idx),
    );

    let WValuesdims0 = (*arch_search).neural_network.WColIdx.dims()[0];

    let network_paramsdims0 = (*arch_search).neural_network.network_params.dims()[0];

    let Hdims0 = (network_paramsdims0 - WValuesdims0) / 6;

    let Wstart = 0;
    let Wend = (WValuesdims0 as i64) - 1;

    let Hstart = Wend + 1;
    let Hend = Hstart + (Hdims0 as i64) - 1;

    let Astart = Hend + 1;
    let Aend = Astart + (Hdims0 as i64) - 1;

    let Bstart = Aend + 1;
    let Bend = Bstart + (Hdims0 as i64) - 1;

    let Cstart = Bend + 1;
    let Cend = Cstart + (Hdims0 as i64) - 1;

    let Dstart = Cend + 1;
    let Dend = Dstart + (Hdims0 as i64) - 1;

    let Estart = Dend + 1;
    let Eend = Estart + (Hdims0 as i64) - 1;

    let Wseqs = [arrayfire::Seq::new(Wstart as i32, Wend as i32, 1i32)];
    let Hseqs = [arrayfire::Seq::new(Hstart as i32, Hend as i32, 1i32)];
    let Aseqs = [arrayfire::Seq::new(Astart as i32, Aend as i32, 1i32)];
    let Bseqs = [arrayfire::Seq::new(Bstart as i32, Bend as i32, 1i32)];
    let Cseqs = [arrayfire::Seq::new(Cstart as i32, Cend as i32, 1i32)];
    let Dseqs = [arrayfire::Seq::new(Dstart as i32, Dend as i32, 1i32)];
    let Eseqs = [arrayfire::Seq::new(Estart as i32, Eend as i32, 1i32)];

    let mut WValues = arrayfire::index(&((*arch_search).neural_network.network_params), &Wseqs);
    let H = arrayfire::index(&((*arch_search).neural_network.network_params), &Hseqs);
    let A = arrayfire::index(&((*arch_search).neural_network.network_params), &Aseqs);
    let B = arrayfire::index(&((*arch_search).neural_network.network_params), &Bseqs);
    let C = arrayfire::index(&((*arch_search).neural_network.network_params), &Cseqs);
    let D = arrayfire::index(&((*arch_search).neural_network.network_params), &Dseqs);
    let E = arrayfire::index(&((*arch_search).neural_network.network_params), &Eseqs);

    let mut hidden_num2 =
        (*arch_search).neural_network.neuron_pos.dims()[0] - output_size - input_size;

    let raytrace_options: raytrace_option_type = raytrace_option_type {
        max_rounds: 10000,
        input_connection_num: init_connection_num * input_size,
        ray_neuron_intersect: false,
        ray_glia_intersect: false,
    };

    (*arch_search).neural_network.netdata.con_rad = input_neuron_con_rad;

    let neuron_num = arch_search.neural_network.neuron_pos.dims()[0];
    let hidden_idx_total = arrayfire::rows(
        &arch_search.neural_network.neuron_idx,
        input_size as i64,
        (neuron_num - output_size - 1) as i64,
    );
    let hidden_pos_total = arrayfire::rows(
        &arch_search.neural_network.neuron_pos,
        input_size as i64,
        (neuron_num - output_size - 1) as i64,
    );

    let input_idx = arrayfire::rows(
        &arch_search.neural_network.neuron_idx,
        0,
        (input_size - 1) as i64,
    );
    let input_pos = arrayfire::rows(
        &arch_search.neural_network.neuron_pos,
        0,
        (input_size - 1) as i64,
    );

    RT3_distance_limited_directly_connected(
        &raytrace_options,
        &arch_search.neural_network.netdata,
        &arch_search.neural_network.glia_pos,
        &input_pos,
        &input_idx,
        &hidden_pos_total,
        &hidden_idx_total,
        //&mut newWValues,
        &mut newWRowIdxCOO,
        &mut newWColIdx,
    );

    let neuron_num = arch_search.neural_network.neuron_pos.dims()[0];

    let hidden_idx_total = arrayfire::rows(
        &arch_search.neural_network.neuron_idx,
        input_size as i64,
        (neuron_num - output_size - 1) as i64,
    );
    let hidden_pos_total = arrayfire::rows(
        &arch_search.neural_network.neuron_pos,
        input_size as i64,
        (neuron_num - output_size - 1) as i64,
    );

    let input_idx = hidden_idx_total.clone();
    let input_pos = hidden_pos_total.clone();

    let raytrace_options: raytrace_option_type = raytrace_option_type {
        max_rounds: 10000,
        input_connection_num: init_connection_num * hidden_num2,
        ray_neuron_intersect: true,
        ray_glia_intersect: true,
    };

    (*arch_search).neural_network.netdata.con_rad = hidden_neuron_con_rad;

    RT3_distance_limited_directly_connected(
        &raytrace_options,
        &arch_search.neural_network.netdata,
        &arch_search.neural_network.glia_pos,
        &input_pos,
        &input_idx,
        &hidden_pos_total,
        &hidden_idx_total,
        //&mut newWValues,
        &mut newWRowIdxCOO,
        &mut newWColIdx,
    );

    let neuron_num = arch_search.neural_network.neuron_pos.dims()[0];

    let input_idx = arrayfire::rows(
        &arch_search.neural_network.neuron_idx,
        input_size as i64,
        (neuron_num - output_size - 1) as i64,
    );
    let input_pos = arrayfire::rows(
        &arch_search.neural_network.neuron_pos,
        input_size as i64,
        (neuron_num - output_size - 1) as i64,
    );

    let hidden_idx_total = arrayfire::rows(
        &arch_search.neural_network.neuron_idx,
        (arch_search.neural_network.neuron_idx.dims()[0] - output_size) as i64,
        (arch_search.neural_network.neuron_idx.dims()[0] - 1) as i64,
    );
    let hidden_pos_total = arrayfire::rows(
        &arch_search.neural_network.neuron_pos,
        (arch_search.neural_network.neuron_idx.dims()[0] - output_size) as i64,
        (arch_search.neural_network.neuron_idx.dims()[0] - 1) as i64,
    );

    (*arch_search).neural_network.netdata.con_rad = output_neuron_con_rad;

    let glia_pos_temp_dims = arrayfire::Dim4::new(&[4, space_dims, 1, 1]);

    let glia_pos_temp = arrayfire::constant::<f32>(-1000.0, glia_pos_temp_dims);

    let raytrace_options: raytrace_option_type = raytrace_option_type {
        max_rounds: 10000,
        input_connection_num: init_connection_num * output_size,
        ray_neuron_intersect: false,
        ray_glia_intersect: false,
    };

    RT3_distance_limited_directly_connected(
        &raytrace_options,
        &arch_search.neural_network.netdata,
        &glia_pos_temp,
        &input_pos,
        &input_idx,
        &hidden_pos_total,
        &hidden_idx_total,
        //&mut newWValues,
        &mut newWRowIdxCOO,
        &mut newWColIdx,
    );

    //Populate WValues with random values
    let rand_dims = arrayfire::Dim4::new(&[newWColIdx.dims()[0], 1, 1, 1]);
    newWValues = neuron_std * arrayfire::randn::<f32>(rand_dims);

    //Clear input/output
    clear_input(
        &mut newWValues,
        &mut newWRowIdxCOO,
        &mut newWColIdx,
        input_size,
    );

    clear_output(
        &mut newWValues,
        &mut newWRowIdxCOO,
        &mut newWColIdx,
        neuron_size - output_size,
    );

    let in_idx = arrayfire::rows(
        &((*arch_search).neural_network.neuron_idx),
        0,
        (input_size - 1) as i64,
    );

    /*
    if (*arch_search).evolution_info.traj_size > 1
    {
        self_loops(
            &((*arch_search).neural_network.netdata),

            &((*arch_search).neural_network.neuron_idx),


            &mut newWValues,
            &mut newWRowIdxCOO,
            &mut newWColIdx
        );
    }
    else
    {
        select_forward_sphere(
            &(*arch_search).neural_network.netdata,
            &mut newWValues,
            &mut newWRowIdxCOO,
            &mut newWColIdx,
            &(*arch_search).neural_network.neuron_pos,
            &(*arch_search).neural_network.neuron_idx
        );
    }
    */

    self_loops(
        &((*arch_search).neural_network.netdata),
        &((*arch_search).neural_network.neuron_idx),
        &mut newWValues,
        &mut newWRowIdxCOO,
        &mut newWColIdx,
    );

    delete_unused_neurons(
        &((*arch_search).neural_network.netdata),
        &mut newWValues,
        &mut newWRowIdxCOO,
        &mut newWColIdx,
        &mut ((*arch_search).neural_network.glia_pos),
        &mut ((*arch_search).neural_network.neuron_pos),
        &mut ((*arch_search).neural_network.neuron_idx),
    );

    let mut newH = H.clone();

    xavier_init(
        &in_idx,
        &newWRowIdxCOO,
        &newWColIdx,
        neuron_size,
        proc_num,
        &mut newWValues,
        &mut newH,
    );

    //Compute global index
    gidx1 = get_global_weight_idx(
        neuron_size,
        &WRowIdxCOO,
        &((*arch_search).neural_network.WColIdx),
    );

    gidx2 = get_global_weight_idx(neuron_size, &newWRowIdxCOO, &newWColIdx);

    let mut gidx1_cpu = vec![u64::default(); gidx1.elements()];
    gidx1.host(&mut gidx1_cpu);

    let mut gidx2_cpu = vec![u64::default(); gidx2.elements()];
    gidx2.host(&mut gidx2_cpu);

    let mut newWValues_cpu = vec![f32::default(); newWValues.elements()];
    newWValues.host(&mut newWValues_cpu);

    let mut newWRowIdxCOO_cpu = vec![i32::default(); newWRowIdxCOO.elements()];
    newWRowIdxCOO.host(&mut newWRowIdxCOO_cpu);

    let mut newWColIdx_cpu = vec![i32::default(); newWColIdx.elements()];
    newWColIdx.host(&mut newWColIdx_cpu);

    let mut WValues_cpu = vec![f32::default(); WValues.elements()];
    WValues.host(&mut WValues_cpu);

    let mut WRowIdxCOO_cpu = vec![i32::default(); WRowIdxCOO.elements()];
    WRowIdxCOO.host(&mut WRowIdxCOO_cpu);

    let mut WColIdx_cpu = vec![i32::default(); (*arch_search).neural_network.WColIdx.elements()];
    (*arch_search).neural_network.WColIdx.host(&mut WColIdx_cpu);

    let mut join_WValues = nohash_hasher::IntMap::default();
    let mut join_WColIdx = nohash_hasher::IntMap::default();
    let mut join_WRowIdxCOO = nohash_hasher::IntMap::default();

    for qq in 0..gidx2.elements() {
        let cur_gidx = gidx2_cpu[qq].clone();

        join_WValues.insert(cur_gidx, newWValues_cpu[qq].clone());
        join_WColIdx.insert(cur_gidx, newWColIdx_cpu[qq].clone());
        join_WRowIdxCOO.insert(cur_gidx, newWRowIdxCOO_cpu[qq].clone());
    }

    for qq in 0..gidx1.elements() {
        let cur_gidx = gidx1_cpu[qq].clone();

        join_WValues.insert(cur_gidx, WValues_cpu[qq].clone());
        join_WColIdx.insert(cur_gidx, WColIdx_cpu[qq].clone());
        join_WRowIdxCOO.insert(cur_gidx, WRowIdxCOO_cpu[qq].clone());
    }

    let mut gidx3: Vec<u64> = join_WValues.clone().into_keys().collect();
    gidx3.par_sort_unstable();

    WValues_cpu = Vec::new();
    WRowIdxCOO_cpu = Vec::new();
    WColIdx_cpu = Vec::new();

    for qq in gidx3 {
        WValues_cpu.push(join_WValues[&qq].clone());
        WColIdx_cpu.push(join_WColIdx[&qq].clone());
        WRowIdxCOO_cpu.push(join_WRowIdxCOO[&qq].clone());
    }

    WValues = arrayfire::Array::new(
        &WValues_cpu,
        arrayfire::Dim4::new(&[WValues_cpu.len() as u64, 1, 1, 1]),
    );
    (*arch_search).neural_network.WColIdx = arrayfire::Array::new(
        &WColIdx_cpu,
        arrayfire::Dim4::new(&[WValues_cpu.len() as u64, 1, 1, 1]),
    );
    WRowIdxCOO = arrayfire::Array::new(
        &WRowIdxCOO_cpu,
        arrayfire::Dim4::new(&[WValues_cpu.len() as u64, 1, 1, 1]),
    );

    //Save values
    (*arch_search).neural_network.WRowIdxCSR = COO_to_CSR(&WRowIdxCOO, neuron_size);

    let total_param_size = WValues.dims()[0]
        + H.dims()[0]
        + A.dims()[0]
        + B.dims()[0]
        + C.dims()[0]
        + D.dims()[0]
        + E.dims()[0];
    let network_params_dims = arrayfire::Dim4::new(&[total_param_size, 1, 1, 1]);

    let Wstart = 0;
    let Wend = (WValues.dims()[0] as i64) - 1;

    let Hstart = Wend + 1;
    let Hend = Hstart + (H.dims()[0] as i64) - 1;

    let Astart = Hend + 1;
    let Aend = Astart + (A.dims()[0] as i64) - 1;

    let Bstart = Aend + 1;
    let Bend = Bstart + (B.dims()[0] as i64) - 1;

    let Cstart = Bend + 1;
    let Cend = Cstart + (C.dims()[0] as i64) - 1;

    let Dstart = Cend + 1;
    let Dend = Dstart + (D.dims()[0] as i64) - 1;

    let Estart = Dend + 1;
    let Eend = Estart + (E.dims()[0] as i64) - 1;

    let Wseqs = [arrayfire::Seq::new(Wstart as i32, Wend as i32, 1i32)];
    let Hseqs = [arrayfire::Seq::new(Hstart as i32, Hend as i32, 1i32)];
    let Aseqs = [arrayfire::Seq::new(Astart as i32, Aend as i32, 1i32)];
    let Bseqs = [arrayfire::Seq::new(Bstart as i32, Bend as i32, 1i32)];
    let Cseqs = [arrayfire::Seq::new(Cstart as i32, Cend as i32, 1i32)];
    let Dseqs = [arrayfire::Seq::new(Dstart as i32, Dend as i32, 1i32)];
    let Eseqs = [arrayfire::Seq::new(Estart as i32, Eend as i32, 1i32)];

    (*arch_search).neural_network.network_params =
        arrayfire::constant::<f32>(0.0, network_params_dims);
    arrayfire::assign_seq(
        &mut ((*arch_search).neural_network.network_params),
        &Wseqs,
        &WValues,
    );
    arrayfire::assign_seq(
        &mut ((*arch_search).neural_network.network_params),
        &Hseqs,
        &H,
    );
    arrayfire::assign_seq(
        &mut ((*arch_search).neural_network.network_params),
        &Aseqs,
        &A,
    );
    arrayfire::assign_seq(
        &mut ((*arch_search).neural_network.network_params),
        &Bseqs,
        &B,
    );
    arrayfire::assign_seq(
        &mut ((*arch_search).neural_network.network_params),
        &Cseqs,
        &C,
    );
    arrayfire::assign_seq(
        &mut ((*arch_search).neural_network.network_params),
        &Dseqs,
        &D,
    );
    arrayfire::assign_seq(
        &mut ((*arch_search).neural_network.network_params),
        &Eseqs,
        &E,
    );

    (*arch_search).neural_network.netdata.active_size =
        (*arch_search).neural_network.neuron_idx.dims()[0];
}

pub fn add_neuron_to_existing2(
    cube_pos: &arrayfire::Array<f32>,
    cube_radius: f32,

    new_active_size: u64,

    arch_search: &mut arch_search_type,
) {
    let neuron_size: u64 = (*arch_search).neural_network.netdata.neuron_size.clone();
    let input_size: u64 = (*arch_search).neural_network.netdata.input_size.clone();
    let output_size: u64 = (*arch_search).neural_network.netdata.output_size.clone();
    let proc_num: u64 = (*arch_search).neural_network.netdata.proc_num.clone();
    let active_size: u64 = (*arch_search).neural_network.netdata.active_size.clone();
    let space_dims: u64 = (*arch_search).neural_network.netdata.space_dims.clone();
    let step_num: u64 = (*arch_search).neural_network.netdata.step_num.clone();
    let batch_size: u64 = (*arch_search).neural_network.netdata.batch_size.clone();

    let del_unused_neuron: bool = (*arch_search)
        .neural_network
        .netdata
        .del_unused_neuron
        .clone();

    let time_step: f32 = (*arch_search).neural_network.netdata.time_step.clone();
    let nratio: f32 = (*arch_search).neural_network.netdata.nratio.clone();
    let neuron_std: f32 = (*arch_search).neural_network.netdata.neuron_std.clone();
    let sphere_rad: f32 = (*arch_search).neural_network.netdata.sphere_rad.clone();
    let neuron_rad: f32 = (*arch_search).neural_network.netdata.neuron_rad.clone();
    let con_rad: f32 = (*arch_search).neural_network.netdata.con_rad.clone();
    let init_prob: f32 = (*arch_search).neural_network.netdata.init_prob.clone();
    let add_neuron_rate: f32 = (*arch_search)
        .neural_network
        .netdata
        .add_neuron_rate
        .clone();
    let del_neuron_rate: f32 = (*arch_search)
        .neural_network
        .netdata
        .del_neuron_rate
        .clone();
    let center_const: f32 = (*arch_search).neural_network.netdata.center_const.clone();
    let spring_const: f32 = (*arch_search).neural_network.netdata.spring_const.clone();
    let repel_const: f32 = (*arch_search).neural_network.netdata.repel_const.clone();

    let mut WRowIdxCOO = CSR_to_COO(&((*arch_search).neural_network.WRowIdxCSR));

    //let COO_batch_size = 1 + ((COO_find_limit/WRowIdxCOO.dims()[0]) as u64);

    let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);
    //let mut new_glia_pos = arrayfire::constant::<f32>(0.0,temp_dims);
    //let mut new_neuron_pos = arrayfire::constant::<f32>(0.0,temp_dims);

    let mut newWValues = arrayfire::constant::<f32>(0.0, temp_dims);
    let mut newWRowIdxCOO = arrayfire::constant::<i32>(0, temp_dims);
    let mut newWColIdx = arrayfire::constant::<i32>(0, temp_dims);

    let mut gidx1 = arrayfire::constant::<u64>(0, temp_dims);
    let mut gidx2 = arrayfire::constant::<u64>(0, temp_dims);
    let mut gidx3 = arrayfire::constant::<u64>(0, temp_dims);
    let mut gidx4 = arrayfire::constant::<u64>(0, temp_dims);

    let mut active_size = (*arch_search).neural_network.neuron_idx.dims()[0];
    let mut hidden_size = active_size - output_size - input_size;

    //Get input neurons
    let mut input_neurons = arrayfire::rows(
        &((*arch_search).neural_network.neuron_pos),
        0,
        (input_size - 1) as i64,
    );

    let output_neurons = arrayfire::rows(
        &((*arch_search).neural_network.neuron_pos),
        (active_size - output_size) as i64,
        (active_size - 1) as i64,
    );

    //Create new neurons from template
    let mut initial_sphere_radius = sphere_rad.clone();
    loop {
        (*arch_search).neural_network.neuron_pos =
            NDsphere_from_NDcube(cube_pos, cube_radius, initial_sphere_radius);

        if ((*arch_search).neural_network.neuron_pos.dims()[0]
            < (new_active_size - input_size - output_size))
        {
            initial_sphere_radius = initial_sphere_radius * sphere_rad_factor;
        } else {
            break;
        }
    }
    (*arch_search).neural_network.neuron_pos = arrayfire::rows(
        &((*arch_search).neural_network.neuron_pos),
        0,
        (new_active_size - input_size - output_size - 1) as i64,
    );

    sort_neuron_pos_sphere(&mut ((*arch_search).neural_network.neuron_pos));

    let sq = vec_norm(&((*arch_search).neural_network.neuron_pos));

    let (m0, _) = arrayfire::max_all::<f32>(&sq);
    (*arch_search).neural_network.netdata.sphere_rad = m0;
    (*arch_search).neural_network.netdata.con_rad = (m0 / (proc_num as f32)) * con_rad_factor;

    //Reconstruct
    input_neurons = ((m0 + (neuron_rad * 2.0f32)) / sphere_rad) * input_neurons;

    (*arch_search).neural_network.neuron_pos = arrayfire::join(
        0,
        &input_neurons,
        &((*arch_search).neural_network.neuron_pos),
    );

    (*arch_search).neural_network.neuron_pos = arrayfire::join(
        0,
        &((*arch_search).neural_network.neuron_pos),
        &output_neurons,
    );

    assign_neuron_idx_with_buffer(
        (*arch_search).evolution_info.max_input_size,
        (*arch_search).evolution_info.max_output_size,
        &((*arch_search).neural_network.netdata),
        &((*arch_search).neural_network.neuron_pos),
        &mut ((*arch_search).neural_network.neuron_idx),
    );

    let WValuesdims0 = (*arch_search).neural_network.WColIdx.dims()[0];

    let network_paramsdims0 = (*arch_search).neural_network.network_params.dims()[0];

    let Hdims0 = (network_paramsdims0 - WValuesdims0) / 6;

    let Wstart = 0;
    let Wend = (WValuesdims0 as i64) - 1;

    let Hstart = Wend + 1;
    let Hend = Hstart + (Hdims0 as i64) - 1;

    let Astart = Hend + 1;
    let Aend = Astart + (Hdims0 as i64) - 1;

    let Bstart = Aend + 1;
    let Bend = Bstart + (Hdims0 as i64) - 1;

    let Cstart = Bend + 1;
    let Cend = Cstart + (Hdims0 as i64) - 1;

    let Dstart = Cend + 1;
    let Dend = Dstart + (Hdims0 as i64) - 1;

    let Estart = Dend + 1;
    let Eend = Estart + (Hdims0 as i64) - 1;

    let Wseqs = [arrayfire::Seq::new(Wstart as i32, Wend as i32, 1i32)];
    let Hseqs = [arrayfire::Seq::new(Hstart as i32, Hend as i32, 1i32)];
    let Aseqs = [arrayfire::Seq::new(Astart as i32, Aend as i32, 1i32)];
    let Bseqs = [arrayfire::Seq::new(Bstart as i32, Bend as i32, 1i32)];
    let Cseqs = [arrayfire::Seq::new(Cstart as i32, Cend as i32, 1i32)];
    let Dseqs = [arrayfire::Seq::new(Dstart as i32, Dend as i32, 1i32)];
    let Eseqs = [arrayfire::Seq::new(Estart as i32, Eend as i32, 1i32)];

    let mut WValues = arrayfire::index(&((*arch_search).neural_network.network_params), &Wseqs);
    let H = arrayfire::index(&((*arch_search).neural_network.network_params), &Hseqs);
    let A = arrayfire::index(&((*arch_search).neural_network.network_params), &Aseqs);
    let B = arrayfire::index(&((*arch_search).neural_network.network_params), &Bseqs);
    let C = arrayfire::index(&((*arch_search).neural_network.network_params), &Cseqs);
    let D = arrayfire::index(&((*arch_search).neural_network.network_params), &Dseqs);
    let E = arrayfire::index(&((*arch_search).neural_network.network_params), &Eseqs);

    /*
    clear_input_to_hidden(
        &mut WValues,
        &mut WRowIdxCOO,
        &mut ((*arch_search).neural_network.WColIdx),
        input_size
    );
    */

    let mut hidden_num2 =
        (*arch_search).neural_network.neuron_pos.dims()[0] - output_size - input_size;

    let mut init_connection_num = 10;

    if init_connection_num >= hidden_num2 {
        init_connection_num = hidden_num2;
    } else {
        let temp_num = ((hidden_num2 as f32) * INOUT_FACTOR) as u64;
        if temp_num > init_connection_num {
            init_connection_num = temp_num;
        }
    }

    input_and_output_layers(
        &((*arch_search).neural_network.netdata),
        &((*arch_search).neural_network.neuron_pos),
        &((*arch_search).neural_network.neuron_idx),
        init_connection_num,
        init_connection_num,
        &mut newWValues,
        &mut newWRowIdxCOO,
        &mut newWColIdx,
    );

    fully_connected_hidden_layers(
        &((*arch_search).neural_network.neuron_pos),
        &((*arch_search).neural_network.neuron_idx),
        &mut ((*arch_search).neural_network.netdata),
        &mut newWValues,
        &mut newWRowIdxCOO,
        &mut newWColIdx,
    );

    //Clear input/output
    clear_input(
        &mut newWValues,
        &mut newWRowIdxCOO,
        &mut newWColIdx,
        input_size,
    );

    clear_output(
        &mut newWValues,
        &mut newWRowIdxCOO,
        &mut newWColIdx,
        neuron_size - output_size,
    );

    let in_idx = arrayfire::rows(
        &((*arch_search).neural_network.neuron_idx),
        0,
        (input_size - 1) as i64,
    );

    if (*arch_search).evolution_info.traj_size > 1 {
        self_loops(
            &((*arch_search).neural_network.netdata),
            //&((*arch_search).neural_network.neuron_pos),
            &((*arch_search).neural_network.neuron_idx),
            &mut newWValues,
            &mut newWRowIdxCOO,
            &mut newWColIdx,
        );
    } else {
        select_forward_sphere(
            &(*arch_search).neural_network.netdata,
            &mut newWValues,
            &mut newWRowIdxCOO,
            &mut newWColIdx,
            &(*arch_search).neural_network.neuron_pos,
            &(*arch_search).neural_network.neuron_idx,
        );
    }

    delete_unused_neurons(
        &((*arch_search).neural_network.netdata),
        &mut newWValues,
        &mut newWRowIdxCOO,
        &mut newWColIdx,
        &mut ((*arch_search).neural_network.glia_pos),
        &mut ((*arch_search).neural_network.neuron_pos),
        &mut ((*arch_search).neural_network.neuron_idx),
    );

    let mut newH = H.clone();

    xavier_init(
        &in_idx,
        &newWRowIdxCOO,
        &newWColIdx,
        neuron_size,
        proc_num,
        &mut newWValues,
        &mut newH,
    );

    /*
    if (*arch_search).evolution_info.traj_size > 1
    {
        assign_self_loop_value(
            &((*arch_search).neural_network.netdata),
            //&((*arch_search).neural_network.neuron_pos),
            &((*arch_search).neural_network.neuron_idx),


            &mut newWValues,
            &mut newWRowIdxCOO,
            &mut newWColIdx
        );
    }
    */

    //Compute global index
    gidx1 = get_global_weight_idx(
        neuron_size,
        &WRowIdxCOO,
        &((*arch_search).neural_network.WColIdx),
    );

    gidx2 = get_global_weight_idx(neuron_size, &newWRowIdxCOO, &newWColIdx);

    let mut gidx1_cpu = vec![u64::default(); gidx1.elements()];
    gidx1.host(&mut gidx1_cpu);

    let mut gidx2_cpu = vec![u64::default(); gidx2.elements()];
    gidx2.host(&mut gidx2_cpu);

    let mut newWValues_cpu = vec![f32::default(); newWValues.elements()];
    newWValues.host(&mut newWValues_cpu);

    let mut newWRowIdxCOO_cpu = vec![i32::default(); newWRowIdxCOO.elements()];
    newWRowIdxCOO.host(&mut newWRowIdxCOO_cpu);

    let mut newWColIdx_cpu = vec![i32::default(); newWColIdx.elements()];
    newWColIdx.host(&mut newWColIdx_cpu);

    let mut WValues_cpu = vec![f32::default(); WValues.elements()];
    WValues.host(&mut WValues_cpu);

    let mut WRowIdxCOO_cpu = vec![i32::default(); WRowIdxCOO.elements()];
    WRowIdxCOO.host(&mut WRowIdxCOO_cpu);

    let mut WColIdx_cpu = vec![i32::default(); (*arch_search).neural_network.WColIdx.elements()];
    (*arch_search).neural_network.WColIdx.host(&mut WColIdx_cpu);

    let mut join_WValues = nohash_hasher::IntMap::default();
    let mut join_WColIdx = nohash_hasher::IntMap::default();
    let mut join_WRowIdxCOO = nohash_hasher::IntMap::default();

    for qq in 0..gidx2.elements() {
        let cur_gidx = gidx2_cpu[qq].clone();

        join_WValues.insert(cur_gidx, newWValues_cpu[qq].clone());
        join_WColIdx.insert(cur_gidx, newWColIdx_cpu[qq].clone());
        join_WRowIdxCOO.insert(cur_gidx, newWRowIdxCOO_cpu[qq].clone());
    }

    for qq in 0..gidx1.elements() {
        let cur_gidx = gidx1_cpu[qq].clone();

        join_WValues.insert(cur_gidx, WValues_cpu[qq].clone());
        join_WColIdx.insert(cur_gidx, WColIdx_cpu[qq].clone());
        join_WRowIdxCOO.insert(cur_gidx, WRowIdxCOO_cpu[qq].clone());
    }

    let mut gidx3: Vec<u64> = join_WValues.clone().into_keys().collect();
    gidx3.par_sort_unstable();

    WValues_cpu = Vec::new();
    WRowIdxCOO_cpu = Vec::new();
    WColIdx_cpu = Vec::new();

    for qq in gidx3 {
        WValues_cpu.push(join_WValues[&qq].clone());
        WColIdx_cpu.push(join_WColIdx[&qq].clone());
        WRowIdxCOO_cpu.push(join_WRowIdxCOO[&qq].clone());
    }

    WValues = arrayfire::Array::new(
        &WValues_cpu,
        arrayfire::Dim4::new(&[WValues_cpu.len() as u64, 1, 1, 1]),
    );
    (*arch_search).neural_network.WColIdx = arrayfire::Array::new(
        &WColIdx_cpu,
        arrayfire::Dim4::new(&[WValues_cpu.len() as u64, 1, 1, 1]),
    );
    WRowIdxCOO = arrayfire::Array::new(
        &WRowIdxCOO_cpu,
        arrayfire::Dim4::new(&[WValues_cpu.len() as u64, 1, 1, 1]),
    );

    //Save values
    (*arch_search).neural_network.WRowIdxCSR = COO_to_CSR(&WRowIdxCOO, neuron_size);

    /*
    let gidx1dims = gidx1.dims()[0];
    let gidx2dims = (gidx2.dims()[0]).max(gidx1dims);


    let COO_batch_size = 1 + ((COO_find_limit/gidx2dims) as u64);


    gidx3 = COO_batch_find(&gidx2, &gidx1, COO_batch_size);


    //Filter out existing connections
    if gidx3.dims()[0] > 0
    {

        let mut table = arrayfire::constant::<bool>(true,gidx2.dims());


        let inarr = arrayfire::constant::<bool>(false, gidx3.dims());

        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&gidx3, 0, None);
        arrayfire::assign_gen(&mut table, &idxrs, &inarr);

        let tempidx = arrayfire::locate(&table);


        newWValues = arrayfire::lookup(&newWValues, &tempidx, 0);
        newWRowIdxCOO = arrayfire::lookup(&newWRowIdxCOO, &tempidx, 0);
        newWColIdx = arrayfire::lookup(&newWColIdx, &tempidx, 0);
        gidx2 = arrayfire::lookup(&gidx2, &tempidx, 0);
    }








    //Insert new connections
    gidx4 = arrayfire::join(0, &gidx1, &gidx2);
    WValues = arrayfire::join(0, &WValues, &newWValues);
    WRowIdxCOO = arrayfire::join(0, &WRowIdxCOO, &newWRowIdxCOO);
    (*arch_search).neural_network.WColIdx = arrayfire::join(0, &((*arch_search).neural_network.WColIdx), &newWColIdx);


    let (_,idx) = arrayfire::sort_index(&gidx4,0,true);

    WValues = arrayfire::lookup(&WValues, &idx, 0);
    WRowIdxCOO = arrayfire::lookup(&WRowIdxCOO, &idx, 0);
    (*arch_search).neural_network.WColIdx = arrayfire::lookup(&((*arch_search).neural_network.WColIdx), &idx, 0);


    //Save values
    (*arch_search).neural_network.WRowIdxCSR = COO_to_CSR(&WRowIdxCOO,neuron_size);
    */

    let total_param_size = WValues.dims()[0]
        + H.dims()[0]
        + A.dims()[0]
        + B.dims()[0]
        + C.dims()[0]
        + D.dims()[0]
        + E.dims()[0];
    let network_params_dims = arrayfire::Dim4::new(&[total_param_size, 1, 1, 1]);

    let Wstart = 0;
    let Wend = (WValues.dims()[0] as i64) - 1;

    let Hstart = Wend + 1;
    let Hend = Hstart + (H.dims()[0] as i64) - 1;

    let Astart = Hend + 1;
    let Aend = Astart + (A.dims()[0] as i64) - 1;

    let Bstart = Aend + 1;
    let Bend = Bstart + (B.dims()[0] as i64) - 1;

    let Cstart = Bend + 1;
    let Cend = Cstart + (C.dims()[0] as i64) - 1;

    let Dstart = Cend + 1;
    let Dend = Dstart + (D.dims()[0] as i64) - 1;

    let Estart = Dend + 1;
    let Eend = Estart + (E.dims()[0] as i64) - 1;

    let Wseqs = [arrayfire::Seq::new(Wstart as i32, Wend as i32, 1i32)];
    let Hseqs = [arrayfire::Seq::new(Hstart as i32, Hend as i32, 1i32)];
    let Aseqs = [arrayfire::Seq::new(Astart as i32, Aend as i32, 1i32)];
    let Bseqs = [arrayfire::Seq::new(Bstart as i32, Bend as i32, 1i32)];
    let Cseqs = [arrayfire::Seq::new(Cstart as i32, Cend as i32, 1i32)];
    let Dseqs = [arrayfire::Seq::new(Dstart as i32, Dend as i32, 1i32)];
    let Eseqs = [arrayfire::Seq::new(Estart as i32, Eend as i32, 1i32)];

    (*arch_search).neural_network.network_params =
        arrayfire::constant::<f32>(0.0, network_params_dims);
    arrayfire::assign_seq(
        &mut ((*arch_search).neural_network.network_params),
        &Wseqs,
        &WValues,
    );
    arrayfire::assign_seq(
        &mut ((*arch_search).neural_network.network_params),
        &Hseqs,
        &H,
    );
    arrayfire::assign_seq(
        &mut ((*arch_search).neural_network.network_params),
        &Aseqs,
        &A,
    );
    arrayfire::assign_seq(
        &mut ((*arch_search).neural_network.network_params),
        &Bseqs,
        &B,
    );
    arrayfire::assign_seq(
        &mut ((*arch_search).neural_network.network_params),
        &Cseqs,
        &C,
    );
    arrayfire::assign_seq(
        &mut ((*arch_search).neural_network.network_params),
        &Dseqs,
        &D,
    );
    arrayfire::assign_seq(
        &mut ((*arch_search).neural_network.network_params),
        &Eseqs,
        &E,
    );

    (*arch_search).neural_network.netdata.active_size =
        (*arch_search).neural_network.neuron_idx.dims()[0];
}

pub fn randomly_delete_neuron_pos(neuron_pos: &mut arrayfire::Array<f32>, new_neuron_num: u64) {
    let space_dims = neuron_pos.dims()[1];

    let random_vector_dims = arrayfire::Dim4::new(&[neuron_pos.dims()[0], 1, 1, 1]);

    let random_vector = arrayfire::randu::<f32>(random_vector_dims);
    let (_, mut random_vector_idx) = arrayfire::sort_index(&random_vector, 0, false);
    random_vector_idx = arrayfire::rows(&random_vector_idx, 0, (new_neuron_num - 1) as i64);

    let colseq = arrayfire::Seq::new(0.0, (space_dims - 1) as f32, 1.0);

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&random_vector_idx, 0, None);
    idxrs.set_index(&colseq, 1, Some(false));
    *neuron_pos = arrayfire::index_gen(neuron_pos, idxrs);
}
