extern crate arrayfire;

/*

extern crate mpi;

use mpi::datatype::{MutView, UserDatatype, View};
use mpi::traits::*;
use mpi::Count;
use mpi::topology::*;



pub fn broadcast_array(
    MPI_root_proc: &mpi::topology::Process<mpi::topology::SystemCommunicator>,

    target_arr: &mut arrayfire::Array<f64>
)
{
    //TO CPU
    let mut target_arr_cpu = vec!(f64::default();target_arr.elements());
    target_arr.host(&mut target_arr_cpu);


    //BROADCAST
    MPI_root_proc.broadcast_into(&mut target_arr_cpu[..]);


    //TO GPU
    *target_arr = arrayfire::Array::new(&target_arr_cpu, target_arr.dims());


}








pub fn sum_array(
    MPI_rank: i32,
    MPI_size: i32,


    ROOT_RANK: i32,
    MPI_root_proc: &mpi::topology::Process<mpi::topology::SystemCommunicator>,



    target_arr: &mut arrayfire::Array<f64>
)
{

    //TO CPU
    let mut target_arr_cpu = vec!(f64::default();target_arr.elements());
    target_arr.host(&mut target_arr_cpu);


    let input_dim0 = target_arr.dims()[0];


    let mut temp_cpu = vec![0f64; (input_dim0 as usize) * (MPI_size as usize)];

    if MPI_rank == ROOT_RANK
    {
        //GATHER INTO ROOT
        MPI_root_proc.gather_into_root(&target_arr_cpu[..], &mut temp_cpu[..]);




        //Summation
        *target_arr = arrayfire::Array::new(&temp_cpu, arrayfire::Dim4::new(&[input_dim0, MPI_size as u64, 1, 1]));

        *target_arr = arrayfire::sum(target_arr, 1);



        //TO CPU
        target_arr.host(&mut target_arr_cpu);
    }
    else
    {
        //GATHER INTO ROOT
        MPI_root_proc.gather_into(&target_arr_cpu[..]);
    }





    //BROADCAST
    MPI_root_proc.broadcast_into(&mut target_arr_cpu[..]);

    //TO GPU
    *target_arr = arrayfire::Array::new(&target_arr_cpu, target_arr.dims());



}

*/
