/*
Plot Figure 2c Distribution of Cells as a function of radius

This code generates 240,000 neurons and 240,000 glial cells in a 739.81 radius network
It is intended to plot the distribution of cells as a function of radius

Generated files
./neuron_pos.csv      Neuron Positions
./glia_pos.csv        Glial Cell Positions


*/

extern crate arrayfire;
extern crate raybnn;

// Use CUDA GPU and GPU device 0
const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;

#[allow(unused_must_use)]
fn main() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);

    //Create Initial Neural Network

    let neuron_size: u64 = 510000;
    let input_size: u64 = 4;
    let output_size: u64 = 3;
    let proc_num: u64 = 3;
    let active_size: u64 = 500000;
    let space_dims: u64 = 3;
    let sim_steps: u64 = 1;
    let mut batch_size: u64 = 105;
    let neuron_rad = 1.0; //Cell radius
    let sphere_rad = 739.81; //Neural Network sphere radius

    let mut netdata: raybnn::neural::network_f32::network_metadata_type =
        raybnn::neural::network_f32::network_metadata_type {
            neuron_size: neuron_size,
            input_size: input_size,
            output_size: output_size,
            proc_num: proc_num,
            active_size: active_size,
            space_dims: space_dims,
            step_num: sim_steps,
            batch_size: batch_size,
            del_unused_neuron: true,

            time_step: 0.3,
            nratio: 0.5,
            neuron_std: 0.3,
            sphere_rad: sphere_rad,
            neuron_rad: neuron_rad,
            con_rad: 0.6,
            init_prob: 0.5,
            add_neuron_rate: 0.0,
            del_neuron_rate: 0.0,
            center_const: 0.005,
            spring_const: 0.01,
            repel_const: 0.01,
        };

    let temp_dims = arrayfire::Dim4::new(&[4, 1, 1, 1]);

    //Create 240,000 neurons and 240,000 glial cells

    let new_active_size = 240000;
    netdata.active_size = new_active_size.clone();

    let mut glia_pos = arrayfire::constant::<f32>(0.0, temp_dims);
    let mut neuron_pos = arrayfire::constant::<f32>(0.0, temp_dims);

    //Create the neural network sphere
    raybnn::physics::initial_f32::sphere_cell_collision_minibatch(
        &netdata,
        &mut glia_pos,
        &mut neuron_pos,
    );

    //Save the glial cell positions
    raybnn::export::dataloader_f32::write_arr_to_csv("./glia_pos.csv", &glia_pos);

    //Save the neuron positions
    raybnn::export::dataloader_f32::write_arr_to_csv("./neuron_pos.csv", &neuron_pos);
}
