mod idat_processing;

extern crate mpi;
use crate::mpi::topology::Communicator;
use mpi::traits::*;
use mpi::request::WaitGuard;

fn main() {

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();
    let processed_data: Vec<Vec<(f64, f64)>>;
    let mut store_red: Vec<Vec<f64>> = Vec::new();
    let mut store_grn: Vec<Vec<f64>> = Vec::new();
    let mut number_of_individuals: i32 = 0;
    let mut store_lines: Vec<i32> = Vec::new();

    // Function reads the intensity data for each individual and perform within BeadSetID normalisation
    match idat_processing::main_processing(&world, rank, size, &mut number_of_individuals, &mut store_lines) {
        Ok(data) => {
            processed_data = data;
            println!("Program Completed Executing");

        }
        Err(err) => {
            eprintln!("Error: {:?}", err);
            return;
        }
    }
    return;
    // Determine the number of times each node will send data...
    // Basically determine the number of data each node has to send to the master node
    let max_number = store_lines.iter().max();

    match max_number {
        Some(&max) => {
            
        }
        None => {
            println!("The vector is empty.");
        }
    }

    // Converting data type to i32
    let mut max: usize = *max_number.unwrap_or(&0) as usize;
    let mut num = max;

    let root_rank = 0;
    let root_process = world.process_at_rank(root_rank);

    //Share the number of SNPs within each individual across the nodes
    root_process.broadcast_into(&mut max);

    // Split the processed_data into two vectors of Vec<Vec<f64>
    let (mut processed_data_red, mut processed_data_grn): (Vec<Vec<f64>>, Vec<Vec<f64>>) = processed_data
        .iter()
        .map(|tuples| tuples.iter().cloned().unzip())
        .unzip();

    // Store the red and green data from the master process
    if rank == 0 {
        store_red.extend(processed_data_red.drain(..));
        store_grn.extend(processed_data_grn.drain(..));
    }

    world.barrier();

    let mut store_lines = store_lines.clone();

    mpi::request::scope(|scope| {
        
        for k in 0..max {
            // Send the processed_green intensities to the root node
            if rank != 0 {
                if num != 0 {
                    let _sreq = WaitGuard::from(
                        world
                            .process_at_rank(0)
                            .immediate_send(scope, &processed_data_red[k as usize][..]),
                    );
                }
            }

            // Root node recieiving data from the other nodes
            if rank == 0 {
                for _i in 1..size {
                    if store_lines[(_i-1) as usize] != 0 {
                        let (msg, _status) = world.process_at_rank(_i).receive_vec::<f64>();
                        store_red.push(msg);
                    }
                }

            }
            // Ensures that every node has arrived at this point before proceeding to send the green data intensities
            world.barrier();

            // Send the processed_green intensities to the root node
            if rank != 0 {
                if num != 0 {
                    let _sreq = WaitGuard::from(
                        world
                            .process_at_rank(0)
                            .immediate_send(scope, &processed_data_grn[k as usize][..]),
                    );
                }
            }

            // root node recieves the data from the other nodes
            if rank == 0 {
                for _i in 1..size {
                    if store_lines[(_i-1) as usize] != 0 {
                        let (msg, _status) = world.process_at_rank(_i).receive_vec::<f64>();
                        store_grn.push(msg);
                        store_lines[(_i-1) as usize] = store_lines[(_i-1) as usize] - 1;
                    }
                }

            }

            if rank != 0 {
                num = num - 1;
            }
            world.barrier();
        }  
    }); 
    
    world.barrier();

    // Transposing the matrix 
    if rank == 0 {
        idat_processing::transpose_matrix_in_place(&mut store_grn);
        idat_processing::transpose_matrix_in_place(&mut store_red);
    }

    let mut snp_number = store_grn.len();

    let root_rank = 0;
    let root_process = world.process_at_rank(root_rank);

    //Share the number of SNPs within each individual across the nodes
    root_process.broadcast_into(&mut snp_number);
    
    world.barrier();
    let mut start = 0 as usize;
    let mut finish = size as usize;
    let mut store_recieved_grn: Vec<Vec<f64>> = Vec::new();
    let mut difference = 0;

    world.barrier();

    // Scattering the processed data across the mpi processes
    for _i in 2..((snp_number/(size as usize)) + 3){
        let mut chunk: Vec<f64> = vec![0.0; number_of_individuals as usize];

        //When the number of data to be split is less than the MPI processes
        //The scatter method in mpi processes, assumes that the data can be evenly distributed 
        if finish > (snp_number as usize) {
            difference = (finish - (snp_number as usize)) as i32;
            finish = (snp_number) as usize;
        }

        if rank < (size - difference) {
            if world.rank() == 0 {
            
                // Flatten the 2D grid into a 1D vector before scattering
                let flat_grid: Vec<f64> = store_grn[start..finish]
                    .iter()
                    .flatten()
                    .map(|&x| x) // Map the references to owned values
                    .collect();

                world.process_at_rank(0).scatter_into_root(flat_grid.as_slice(), chunk.as_mut_slice());
            } else {
                world.process_at_rank(0).scatter_into(chunk.as_mut_slice());
            }
            store_recieved_grn.push(chunk.clone());
            start = finish;
            finish = (start + size as usize)as usize;
        }
    }

    world.barrier();

    let mut start = 0 as usize;
    let mut finish = size as usize;
    let mut store_recieved_red: Vec<Vec<f64>> = Vec::new();
    let mut difference = 0;

    // Scattering the processed data across the mpi processes
    for _i in 2..((snp_number/(size as usize)) + 3){
        let mut chunk: Vec<f64> = vec![0.0; number_of_individuals as usize];

        //When the number of data to be split is less than the MPI processes
        //The scatter method in mpi processes, assumes that the data can be evenly distributed 
        if finish > (snp_number as usize) {
            difference = (finish - (snp_number as usize)) as i32;
            finish = (snp_number) as usize;
        }

        if rank < (size - difference) {
            if world.rank() == 0 {
            
                // Flatten the 2D grid into a 1D vector before scattering
                let flat_grid: Vec<f64> = store_red[start..finish]
                    .iter()
                    .flatten()
                    .map(|&x| x) // Map the references to owned values
                    .collect();

                world.process_at_rank(0).scatter_into_root(flat_grid.as_slice(), chunk.as_mut_slice());
            } else {
                world.process_at_rank(0).scatter_into(chunk.as_mut_slice());
            }
            store_recieved_red.push(chunk.clone());
            start = finish;
            finish = (start + size as usize)as usize;
        }
    }
    world.barrier();

    // Combining the recieved data into a tuple of red and green intensities
    let combined: Vec<Vec<(f64, f64)>> = store_recieved_red
        .iter()
        .zip(store_recieved_grn.iter())
        .map(|(v1, v2)| v1.iter().zip(v2.iter()).map(|(&a, &b)| (a, b)).collect())
        .collect();

    // Normalisation within SNP across the individuals
    let _ = idat_processing::snp_normalisation(&combined);

    println!("Program Finished Running Rank {}", rank);
}

