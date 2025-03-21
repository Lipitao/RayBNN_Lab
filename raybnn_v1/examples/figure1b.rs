/*
Plot Figure 1b Generating an Example Neural Network Sphere


Generates these files

./figure1_neural_network.csv    Neural network file containing the cell positions and the weights of the neural connections


*/

// Import RayBNN
use arrayfire;
use raybnn;

use raybnn::physics::update_f32::add_neuron_option_type;

// Use CUDA GPU and GPU device 0
const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;

#[allow(unused_must_use)]
fn main() {
    //Use time for random seed
    let sys_time = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs() as u64;
    arrayfire::set_seed(sys_time);

    // Use CUDA GPU and GPU device 0
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);

    arrayfire::sync(DEVICE);
    arrayfire::device_gc();

    //Create Small Initial Network Network

    //With Parameters
    let dir_path = "/tmp/".to_string();
    let max_input_size: u64 = 10;
    let mut input_size: u64 = 10;
    let max_output_size: u64 = 1;
    let output_size: u64 = 1;
    let max_neuron_size: u64 = 200;
    let mut batch_size: u64 = 32;
    let mut traj_size = 1;
    let mut proc_num = 5;

    //Create Neural Network object
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

    //Add 30 neurons to existing neural network
    //Raytrace radius of 40 neuron radius
    let add_neuron_options: add_neuron_option_type = add_neuron_option_type {
        new_active_size: 30,
        init_connection_num: 1000,
        input_neuron_con_rad: 40.0 * arch_search.neural_network.netdata.neuron_rad,
        hidden_neuron_con_rad: 40.0 * arch_search.neural_network.netdata.neuron_rad,
        output_neuron_con_rad: 40.0 * arch_search.neural_network.netdata.neuron_rad,
    };

    //Add 30 neurons to existing neural network
    raybnn::physics::update_f32::add_neuron_to_existing3(&add_neuron_options, &mut arch_search);

    //Save neural network to CSV file
    let filename = "./figure1_neural_network.csv";
    raybnn::export::dataloader_f32::save_network2(&filename, &arch_search.neural_network);
}
