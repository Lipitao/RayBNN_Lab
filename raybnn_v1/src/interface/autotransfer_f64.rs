extern crate arrayfire;

use nohash_hasher;
use std::collections::HashMap;

use crate::neural::network_f64::neural_network_type;

use crate::graph::adjacency_f64::add_random_weights;
use crate::graph::adjacency_f64::delete_weights_with_prob;
use crate::interface::automatic_f64::arch_search_type;

use crate::graph::large_sparse_i32::CSR_to_COO;

use crate::graph::large_sparse_i32::COO_to_CSR;

use crate::graph::path_f64::find_path_backward_group2;

use crate::neural::network_f64::state_space_forward_batch;

use crate::physics::construct_f64::plane_surface_on_NDsphere;

use crate::physics::initial_f64::assign_neuron_idx_with_buffer;

use crate::physics::initial_f64::create_spaced_input_neuron_on_sphere;

use crate::export::dataloader_u64::find_cube_paths;

use crate::export::dataloader_f64::extract_file_info2;

use crate::interface::automatic_f64::network_info_seed_type;
use crate::interface::automatic_f64::set_network_seed;

use crate::export::dataloader_f64::load_network_structure;

use crate::physics::update_f64::add_neuron_to_existing2;

use crate::physics::update_f64::reduce_network_size;

pub fn shuffle_weights(i: u64, arch_search: &mut arch_search_type) {
    let neuron_size: u64 = (*arch_search).neural_network.netdata.neuron_size.clone();
    let input_size: u64 = (*arch_search).neural_network.netdata.input_size.clone();
    let output_size: u64 = (*arch_search).neural_network.netdata.output_size.clone();
    let proc_num: u64 = (*arch_search).neural_network.netdata.proc_num.clone();
    let mut active_size: u64 = (*arch_search).neural_network.netdata.active_size.clone();
    let space_dims: u64 = (*arch_search).neural_network.netdata.space_dims.clone();
    let step_num: u64 = (*arch_search).neural_network.netdata.step_num.clone();

    let del_unused_neuron: bool = (*arch_search)
        .neural_network
        .netdata
        .del_unused_neuron
        .clone();

    let time_step: f64 = (*arch_search).neural_network.netdata.time_step.clone();
    let nratio: f64 = (*arch_search).neural_network.netdata.nratio.clone();
    let neuron_std: f64 = (*arch_search).neural_network.netdata.neuron_std.clone();
    let sphere_rad: f64 = (*arch_search).neural_network.netdata.sphere_rad.clone();
    let neuron_rad: f64 = (*arch_search).neural_network.netdata.neuron_rad.clone();
    let con_rad: f64 = (*arch_search).neural_network.netdata.con_rad.clone();
    let center_const: f64 = (*arch_search).neural_network.netdata.center_const.clone();
    let spring_const: f64 = (*arch_search).neural_network.netdata.spring_const.clone();
    let repel_const: f64 = (*arch_search).neural_network.netdata.repel_const.clone();

    let batch_size: u64 = (*arch_search).neural_network.netdata.batch_size.clone();

    let mut WRowIdxCOO = CSR_to_COO(&((*arch_search).neural_network.WRowIdxCSR));

    let connection_num = WRowIdxCOO.dims()[0];

    let network_info = network_info_seed_type {
        i: i.clone(),
        proc_num: proc_num.clone(),
        connection_num: connection_num.clone(),
        active_size: active_size.clone(),
    };

    set_network_seed(&network_info);

    /*
    //Make the neural network smaller
    reduce_network_size(
        active_size-1,

        arch_search,
    );
    (*arch_search).neural_network.netdata.active_size = (*arch_search).neural_network.neuron_idx.dims()[0];


    WRowIdxCOO = CSR_to_COO(&((*arch_search).neural_network.WRowIdxCSR));
    */

    let mut input_size = (*arch_search).neural_network.netdata.input_size;
    let mut output_size = (*arch_search).neural_network.netdata.output_size;

    let max_prune_num = (*arch_search).evolution_info.max_prune_num;

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

    con_num = (*arch_search).neural_network.WColIdx.dims()[0];
    del_num = ((con_num as f64) * max_prune_num) as u64;

    if del_num <= 10 {
        del_num = 1;
    }

    delete_weights_with_prob(
        &mut WValues,
        &mut WRowIdxCOO,
        &mut ((*arch_search).neural_network.WColIdx),
        del_num,
    );

    add_random_weights(
        &(*arch_search).neural_network.netdata,
        &(*arch_search).neural_network.neuron_idx,
        &mut WValues,
        &mut WRowIdxCOO,
        &mut ((*arch_search).neural_network.WColIdx),
        del_num,
    );

    /*
    //Find all cube paths
    let cube_path_vec = find_cube_paths(&arch_search.evolution_info.dir_path  );

    let cube_path = cube_path_vec[0].clone();

    let cube_radius= extract_file_info2(&cube_path)[0];

    //Load cube pos
    let temp_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    let mut cube_glia_pos = arrayfire::constant::<f64>(0.0,temp_dims);
    let mut cube_neuron_pos = arrayfire::constant::<f64>(0.0,temp_dims);
    let mut cube_neuron_idx = arrayfire::constant::<i32>(0,temp_dims);

    load_network_structure(
        &cube_path,

        &mut cube_glia_pos,
        &mut cube_neuron_pos,
        &mut cube_neuron_idx
    );
    drop(cube_glia_pos);
    drop(cube_neuron_idx);


    //Add new neurons
    add_neuron_to_existing2(
        &cube_neuron_pos,
        cube_radius,

        active_size,

        arch_search,
    );
    (*arch_search).neural_network.netdata.active_size = (*arch_search).neural_network.neuron_idx.dims()[0];


    WRowIdxCOO = CSR_to_COO(&((*arch_search).neural_network.WRowIdxCSR));
    */

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
        arrayfire::constant::<f64>(0.0, network_params_dims);
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

pub fn reduce_weights(del_num: u64, arch_search: &mut arch_search_type) {
    let neuron_size: u64 = (*arch_search).neural_network.netdata.neuron_size.clone();
    let input_size: u64 = (*arch_search).neural_network.netdata.input_size.clone();
    let output_size: u64 = (*arch_search).neural_network.netdata.output_size.clone();
    let proc_num: u64 = (*arch_search).neural_network.netdata.proc_num.clone();
    let mut active_size: u64 = (*arch_search).neural_network.netdata.active_size.clone();
    let space_dims: u64 = (*arch_search).neural_network.netdata.space_dims.clone();
    let step_num: u64 = (*arch_search).neural_network.netdata.step_num.clone();

    let del_unused_neuron: bool = (*arch_search)
        .neural_network
        .netdata
        .del_unused_neuron
        .clone();

    let time_step: f64 = (*arch_search).neural_network.netdata.time_step.clone();
    let nratio: f64 = (*arch_search).neural_network.netdata.nratio.clone();
    let neuron_std: f64 = (*arch_search).neural_network.netdata.neuron_std.clone();
    let sphere_rad: f64 = (*arch_search).neural_network.netdata.sphere_rad.clone();
    let neuron_rad: f64 = (*arch_search).neural_network.netdata.neuron_rad.clone();
    let con_rad: f64 = (*arch_search).neural_network.netdata.con_rad.clone();
    let center_const: f64 = (*arch_search).neural_network.netdata.center_const.clone();
    let spring_const: f64 = (*arch_search).neural_network.netdata.spring_const.clone();
    let repel_const: f64 = (*arch_search).neural_network.netdata.repel_const.clone();

    let batch_size: u64 = (*arch_search).neural_network.netdata.batch_size.clone();

    let mut WRowIdxCOO = CSR_to_COO(&((*arch_search).neural_network.WRowIdxCSR));

    let connection_num = WRowIdxCOO.dims()[0];

    let mut input_size = (*arch_search).neural_network.netdata.input_size;
    let mut output_size = (*arch_search).neural_network.netdata.output_size;

    let max_prune_num = (*arch_search).evolution_info.max_prune_num;

    let mut active_size = (*arch_search).neural_network.neuron_idx.dims()[0];
    let mut con_num = (*arch_search).neural_network.WColIdx.dims()[0];

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

    if (del_num + 10) > (*arch_search).neural_network.WColIdx.dims()[0] {
        return;
    }

    delete_weights_with_prob(
        &mut WValues,
        &mut WRowIdxCOO,
        &mut ((*arch_search).neural_network.WColIdx),
        del_num,
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
        arrayfire::constant::<f64>(0.0, network_params_dims);
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

pub fn resize_input(new_input_size: u64, arch_search: &mut arch_search_type) {
    let neuron_size: u64 = (*arch_search).neural_network.netdata.neuron_size.clone();
    let mut input_size: u64 = (*arch_search).neural_network.netdata.input_size.clone();
    let output_size: u64 = (*arch_search).neural_network.netdata.output_size.clone();
    let proc_num: u64 = (*arch_search).neural_network.netdata.proc_num.clone();
    let active_size: u64 = (*arch_search).neural_network.netdata.active_size.clone();
    let space_dims: u64 = (*arch_search).neural_network.netdata.space_dims.clone();
    let step_num: u64 = (*arch_search).neural_network.netdata.step_num.clone();

    let sphere_rad: f64 = (*arch_search).neural_network.netdata.sphere_rad.clone();
    let neuron_rad: f64 = (*arch_search).neural_network.netdata.neuron_rad.clone();

    let max_input_size = (*arch_search).evolution_info.max_input_size;
    let max_output_size = (*arch_search).evolution_info.max_output_size;

    (*arch_search).neural_network.neuron_pos = arrayfire::rows(
        &(*arch_search).neural_network.neuron_pos,
        input_size as i64,
        ((*arch_search).neural_network.neuron_pos.dims()[0] - 1) as i64,
    );

    let sqrt_input = (new_input_size as f64).sqrt().ceil() as u64;
    let plane_shape = vec![sqrt_input, sqrt_input];

    let mut input_neurons =
        plane_surface_on_NDsphere(&plane_shape, sphere_rad + (neuron_rad * 2.0));

    if input_neurons.dims()[0] > new_input_size {
        input_neurons = arrayfire::rows(&input_neurons, 0, (new_input_size - 1) as i64);
    }

    (*arch_search).neural_network.neuron_pos =
        arrayfire::join(0, &input_neurons, &(*arch_search).neural_network.neuron_pos);

    (*arch_search).neural_network.netdata.active_size =
        (*arch_search).neural_network.neuron_pos.dims()[0];

    input_size = new_input_size;
    (*arch_search).neural_network.netdata.input_size = new_input_size;

    assign_neuron_idx_with_buffer(
        max_input_size,
        max_output_size,
        &(*arch_search).neural_network.netdata,
        &(*arch_search).neural_network.neuron_pos,
        &mut (*arch_search).neural_network.neuron_idx,
    );
}

pub fn resize_input_with_channels(
    Nx: u64,
    Ny: u64,

    new_channels: u64,

    arch_search: &mut arch_search_type,
) {
    let neuron_size: u64 = (*arch_search).neural_network.netdata.neuron_size.clone();
    let mut input_size: u64 = (*arch_search).neural_network.netdata.input_size.clone();
    let output_size: u64 = (*arch_search).neural_network.netdata.output_size.clone();
    let proc_num: u64 = (*arch_search).neural_network.netdata.proc_num.clone();
    let active_size: u64 = (*arch_search).neural_network.netdata.active_size.clone();
    let space_dims: u64 = (*arch_search).neural_network.netdata.space_dims.clone();
    let step_num: u64 = (*arch_search).neural_network.netdata.step_num.clone();

    let sphere_rad: f64 = (*arch_search).neural_network.netdata.sphere_rad.clone();
    let neuron_rad: f64 = (*arch_search).neural_network.netdata.neuron_rad.clone();

    let max_input_size = (*arch_search).evolution_info.max_input_size;
    let max_output_size = (*arch_search).evolution_info.max_output_size;

    (*arch_search).neural_network.neuron_pos = arrayfire::rows(
        &(*arch_search).neural_network.neuron_pos,
        input_size as i64,
        ((*arch_search).neural_network.neuron_pos.dims()[0] - 1) as i64,
    );

    //let sqrt_input = (new_input_size as f64).sqrt().ceil() as u64 ;
    //let plane_shape = vec![sqrt_input, sqrt_input];

    let mut input_neurons =
        create_spaced_input_neuron_on_sphere(sphere_rad + (neuron_rad * 2.0), Nx, Ny);

    /*
    if input_neurons.dims()[0] > new_input_size
    {
        input_neurons = arrayfire::rows(&input_neurons, 0, (new_input_size-1)  as i64);
    }
    */

    let tile_dims = arrayfire::Dim4::new(&[new_channels, 1, 1, 1]);
    input_neurons = arrayfire::tile(&input_neurons, tile_dims);

    /*
    let template_neurons = input_neurons.clone();

    let increment = std::f64::consts::TAU/(new_channels as f64);
    let mut angle = increment.clone();
    for ridx in 1..new_channels
    {
        let mut new_pos = template_neurons.clone();



        let rotation_cpu: Vec<f64> = vec![1.0, 0.0, 0.0,       0.0, angle.cos(), -angle.sin(),        0.0, angle.sin(), angle.cos()   ];
        let mut rotation = arrayfire::Array::new(&rotation_cpu, arrayfire::Dim4::new(&[3, 3, 1, 1]));



        rotation = arrayfire::transpose(&rotation, false);

        new_pos = arrayfire::matmul(
            &new_pos,
            &rotation,
            arrayfire::MatProp::NONE,
            arrayfire::MatProp::NONE
        );


        angle = angle + increment;

        input_neurons = arrayfire::join(0, &input_neurons, &new_pos);
    }
    */

    //let tile_dims = arrayfire::Dim4::new(&[new_channels,1,1,1]);

    //input_neurons =  arrayfire::tile(&input_neurons, tile_dims);

    (*arch_search).neural_network.neuron_pos =
        arrayfire::join(0, &input_neurons, &(*arch_search).neural_network.neuron_pos);

    (*arch_search).neural_network.netdata.active_size =
        (*arch_search).neural_network.neuron_pos.dims()[0];

    input_size = Nx * Ny * new_channels;
    (*arch_search).neural_network.netdata.input_size = Nx * Ny * new_channels;

    assign_neuron_idx_with_buffer(
        max_input_size,
        max_output_size,
        &(*arch_search).neural_network.netdata,
        &(*arch_search).neural_network.neuron_pos,
        &mut (*arch_search).neural_network.neuron_idx,
    );
}

pub fn reduce_dataset(
    input_size: u64,
    max_input_size: u64,

    TOTAL_DATASET: &nohash_hasher::IntMap<u64, Vec<f64>>,
) -> nohash_hasher::IntMap<u64, Vec<f64>> {
    let mut tempRSSI = nohash_hasher::IntMap::default();

    for (key, value) in TOTAL_DATASET {
        let mut tempvec = value.clone();

        let veclen = tempvec.len() as u64;
        let mut formatarr = arrayfire::Array::new(
            &tempvec,
            arrayfire::Dim4::new(&[max_input_size, veclen / max_input_size, 1, 1]),
        );

        formatarr = arrayfire::rows(&formatarr, 0, (input_size - 1) as i64);

        tempvec = vec![f64::default(); formatarr.elements()];
        formatarr.host(&mut tempvec);

        tempRSSI.insert(key.clone(), tempvec.clone());
    }

    tempRSSI
}
