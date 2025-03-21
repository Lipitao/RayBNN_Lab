/*
Plot  2f Probability Density Function of the Number of Connections per Neuron

This code uses RT-3 to plot the probability density function of the number of neural connections per neuron compared to the density of the neural network sphere

Generated files
./fig2f_WRowIdxCOO_*.csv     Row index of the COO Sparse Matrix
./fig2f_WColIdx_*.csv        Column index of the COO Sparse Matrix
./fig2f_neuron_pos_*.csv     Neuron positions
./fig2f_neuron_idx_*.csv     Indexes of the neurons


*/

extern crate arrayfire;
extern crate raybnn;

use std::time::{Duration, Instant};

use raybnn::physics::raytrace_f32::raytrace_option_type;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;

#[allow(unused_must_use)]
fn main() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);

    //List of densities to sweep through
    let density_list = vec![
        512.0e-5, 1024.0e-5, 2048.0e-5, 3048.0e-5, 4001.0e-5, 5001.0e-5, 6001.0e-5, 8001.0e-5,
        12001.0e-5,
    ];

    //Save Collision Runtime
    raybnn::export::dataloader_f32::write_vec_cpu_to_csv("./fig2f_density_list.csv", &density_list);

    for ii in 0..density_list.len() {
        let density = density_list[ii].clone();

        //Neural Network Initialization constants

        let mut sphere_rad = 80.0;
        let mut test_num =
            (4.0 / 3.0) * std::f32::consts::PI * sphere_rad * sphere_rad * sphere_rad * density;

        println!("testnum {}", test_num);
        let dir_path = "/tmp/".to_string();

        let input_size: u64 = 1;
        let max_input_size: u64 = 1;

        let max_output_size: u64 = 1;
        let output_size: u64 = 1;

        let max_neuron_size: u64 = (test_num as u64) + 200;

        let mut batch_size: u64 = 100;
        let mut traj_size = 20;

        //Create Neural Network

        let mut arch_search = raybnn::interface::automatic_f32::create_start_archtecture(
            input_size,
            max_input_size,
            output_size,
            max_output_size,
            max_neuron_size,
            batch_size,
            traj_size,
            &dir_path,
        );

        arch_search.neural_network.netdata.nratio = 0.5;

        arch_search.neural_network.netdata.neuron_rad = 1.0;

        arch_search.neural_network.netdata.sphere_rad = sphere_rad;

        arch_search.neural_network.netdata.active_size = (test_num as u64);

        //Add new neurons
        arrayfire::sync(0);
        let start = Instant::now();
        raybnn::physics::initial_f32::sphere_cell_collision_minibatch(
            &arch_search.neural_network.netdata,
            &mut arch_search.neural_network.glia_pos,
            &mut arch_search.neural_network.neuron_pos,
        );

        let mut active_size = arch_search.neural_network.neuron_pos.dims()[0];
        arch_search.neural_network.netdata.active_size = active_size;

        raybnn::physics::initial_f32::assign_neuron_idx_with_buffer(
            max_input_size,
            max_output_size,
            &arch_search.neural_network.netdata,
            &arch_search.neural_network.neuron_pos,
            &mut arch_search.neural_network.neuron_idx,
        );

        arrayfire::sync(0);
        let duration = start.elapsed();
        println!(
            "Add neurons {}, {}",
            duration.as_secs_f64(),
            arch_search.neural_network.neuron_pos.dims()[0]
        );

        let mut WRowIdxCOO =
            raybnn::graph::large_sparse_i32::CSR_to_COO(&arch_search.neural_network.WRowIdxCSR);

        let WValuesdims0 = arch_search.neural_network.WColIdx.dims()[0];

        let network_paramsdims0 = arch_search.neural_network.network_params.dims()[0];

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

        let mut WValues = arrayfire::index(&(arch_search.neural_network.network_params), &Wseqs);

        let gidx2 = raybnn::graph::adjacency_f32::get_global_weight_idx(
            max_neuron_size,
            &WRowIdxCOO,
            &arch_search.neural_network.WColIdx,
        );
        let mut gidx2_cpu = vec![u64::default(); gidx2.elements()];
        gidx2.host(&mut gidx2_cpu);
        drop(gidx2);

        let neuron_num2 = arch_search.neural_network.neuron_pos.dims()[0];

        let gidx1 = raybnn::graph::adjacency_f32::get_global_weight_idx(
            max_neuron_size,
            &WRowIdxCOO,
            &arch_search.neural_network.WColIdx,
        );
        let mut gidx1_cpu = vec![u64::default(); gidx1.elements()];
        gidx1.host(&mut gidx1_cpu);
        drop(gidx1);

        let gidx2 = raybnn::graph::adjacency_f32::get_global_weight_idx(
            max_neuron_size,
            &WRowIdxCOO,
            &arch_search.neural_network.WColIdx,
        );
        let mut gidx2_cpu = vec![u64::default(); gidx2.elements()];
        gidx2.host(&mut gidx2_cpu);
        drop(gidx2);

        let neuron_num2 = arch_search.neural_network.neuron_pos.dims()[0];

        let hidden_idx_total = arrayfire::rows(
            &arch_search.neural_network.neuron_idx,
            input_size as i64,
            (neuron_num2 - output_size - 1) as i64,
        );
        let hidden_pos_total = arrayfire::rows(
            &arch_search.neural_network.neuron_pos,
            input_size as i64,
            (neuron_num2 - output_size - 1) as i64,
        );

        let input_idx = hidden_idx_total.clone();
        let input_pos = hidden_pos_total.clone();

        // RT-3 40 Neuron Radius
        arch_search.neural_network.netdata.con_rad =
            40.0 * arch_search.neural_network.netdata.neuron_rad;

        println!("hidden_pos_total {}", hidden_pos_total.dims()[0]);
        println!("input_pos {}", input_pos.dims()[0]);

        let temp_dims = arrayfire::Dim4::new(&[1, 1, 1, 1]);

        WValues = arrayfire::constant::<f32>(0.0, temp_dims);
        WRowIdxCOO = arrayfire::constant::<i32>(0, temp_dims);
        arch_search.neural_network.WColIdx = arrayfire::constant::<i32>(0, temp_dims);

        // Raytrace as many neural connections as possible
        let raytrace_options: raytrace_option_type = raytrace_option_type {
            max_rounds: 100000,
            input_connection_num: 100000,
            ray_neuron_intersect: true,
            ray_glia_intersect: true,
        };

        // Raytrace neural connections

        arrayfire::sync(0);
        let start = Instant::now();
        raybnn::physics::raytrace_f32::RT3_distance_limited_directly_connected(
            &raytrace_options,
            &arch_search.neural_network.netdata,
            &arch_search.neural_network.glia_pos,
            &input_pos,
            &input_idx,
            &hidden_pos_total,
            &hidden_idx_total,
            &mut WRowIdxCOO,
            &mut arch_search.neural_network.WColIdx,
        );
        arrayfire::sync(0);
        let duration = start.elapsed();
        println!(
            "Ray trace {}, {}",
            duration.as_secs_f64(),
            WRowIdxCOO.dims()[0]
        );

        // Save Data into CSV file

        let filename = format!("fig2f_WRowIdxCOO_{}.csv", density);
        raybnn::export::dataloader_i32::write_arr_to_csv(&filename, &WRowIdxCOO);

        let filename = format!("fig2f_WColIdx_{}.csv", density);
        raybnn::export::dataloader_i32::write_arr_to_csv(
            &filename,
            &arch_search.neural_network.WColIdx,
        );

        let filename = format!("fig2f_neuron_pos_{}.csv", density);
        raybnn::export::dataloader_f32::write_arr_to_csv(
            &filename,
            &arch_search.neural_network.neuron_pos,
        );

        let filename = format!("fig2f_neuron_idx_{}.csv", density);
        raybnn::export::dataloader_i32::write_arr_to_csv(
            &filename,
            &arch_search.neural_network.neuron_idx,
        );
    }
}
