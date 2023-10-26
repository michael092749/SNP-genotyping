//porting Crates and Modules
extern crate mpi;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead};
use std::io::Read;
use byteorder::LittleEndian;
use byteorder::ReadBytesExt;
use std::io::Seek;
use std::env;
use std::sync::{Arc, Mutex};
use std::thread;
use std::thread::JoinHandle;
use mpi::topology::SystemCommunicator;
use std::fs;
use normalisation::apply_normalisation::Normalise;
use crate::mpi::collective::CommunicatorCollectives;
use crate::mpi::topology::Communicator;
use crate::mpi::point_to_point::Source;
use crate::mpi::point_to_point::Destination;
use rayon::prelude::*;

// Initialising Constant Variable and New Data types
const FID_N_SNPS_READ: u16 = 1000;
const FID_BARCODE: u16 = 402;
const FID_ILLUMINAID: u16 = 102;
const FID_MEAN: u16 = 104;

fn read_u32_array<R: Read>(reader: &mut R, count: usize) -> io::Result<Vec<u32>> {
    let mut buffer = vec![0u32; count];
    reader.read_u32_into::<LittleEndian>(&mut buffer)?;
    Ok(buffer)
}

fn read_u16_array<R: Read>(reader: &mut R, count: usize) -> io::Result<Vec<u16>> {
    let mut buffer = vec![0u16; count];
    reader.read_u16_into::<LittleEndian>(&mut buffer)?;
    Ok(buffer)
}

// Function processes the idat files of the individuals
fn read_idat_values(fname: &str, _illumina_type: &str) -> Result<(Vec<u32>, Vec<f64>),io::Error> {
    let mut file = File::open(fname)?;

    // Read as a string
    let mut magic_number = [0u8; 4];
    file.read_exact(&mut magic_number)?;

    if magic_number != b"IDAT"[..] {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Magic number is {:?}", magic_number),
        ));
    }

    // Read the IDAT version (a 64-bit integer in little-endian byte order)
    let version = file.read_u64::<LittleEndian>()?;

    if version != 3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("IDAT version 3 supported only, found {}", version),
        ));
    }

    // Read the field count
    let fcount = file.read_u32::<LittleEndian>()?;

    let mut field_val = HashMap::new();
    for _ in 0..fcount {
        let fcode = file.read_u16::<LittleEndian>()?;
        let offset = file.read_u64::<LittleEndian>()?;
        field_val.insert(fcode, offset);
    }

    // Seek to the position of FID_N_SNPS_READ
    let num_markers_offset = field_val[&FID_N_SNPS_READ];
    file.seek(io::SeekFrom::Start(num_markers_offset))?;

    // Read the number of markers
    let num_markers = file.read_u32::<LittleEndian>()?;

    // Seek to the position of FID_BARCODE
    let barcode_offset = field_val[&FID_BARCODE];
    file.seek(io::SeekFrom::Start(barcode_offset))?;

    // Read the barcode
    let _bcode = file.read_u32::<LittleEndian>()?;

    // Seek to the position of FID_ILLUMINAID
    let illumina_id_offset = field_val[&FID_ILLUMINAID];
    file.seek(io::SeekFrom::Start(illumina_id_offset))?;

    // Read Illumina IDs as u32 array
    let iids = read_u32_array(&mut file, num_markers as usize)?;

    // Seek to the position of FID_MEAN
    let mean_offset = field_val[&FID_MEAN];
    file.seek(io::SeekFrom::Start(mean_offset))?;

    // Read means as u16 array
    let vals = read_u16_array(&mut file, num_markers as usize)?;
    // let grn_vals = read_u16_array(&mut file, num_markers as usize)?;
    let f64_vals: Vec<f64> = vals.iter().map(|&u16_val| u16_val as f64).collect();

    Ok((iids, f64_vals))
}

// Get the directory paths to the idat files to process
fn process_sample_sheet_line(
    line: &str,
    idat_directory: &str,
    rank: i32,
    _size: i32,
    ids: &Arc<Mutex<Vec<u32>>>,
    read_ids: &bool,
    batch_comment_index: &Option<usize>,
    array_info_s_index: &Option<usize>,
    sentrix_id_index: &Option<usize>,
) -> Vec<(f64, f64)>{
    let mut data: Vec<(f64, f64)> = Vec::new();
    // Split the line into fields
    let record: Vec<&str> = line.split(',').collect();
    if let Some(batch_comment_index) = batch_comment_index {
        if let Some(array_info_s_index) = array_info_s_index {
            if let Some(sentrix_id_index) = sentrix_id_index {
                // Check if the record has enough fields
                if record.len() >= 20 {
                    let batch_comment = record[*batch_comment_index];
                    let array_info_s = record[*array_info_s_index];
                    let sentrix_id = record[*sentrix_id_index];

                    // Remove spaces and ensure that batch_comment ends with "_iDATS"
                    let mut batch_comment_cleaned = batch_comment.replace(" ", "").trim().to_string();
                    batch_comment_cleaned.push_str("_iDATS");

                    // Construct the file paths for Red and Grn IDAT files
                    let red_idat_path = construct_red_idat_path(idat_directory, &batch_comment_cleaned, array_info_s, sentrix_id);
                    let grn_idat_path = construct_grn_idat_path(idat_directory, &batch_comment_cleaned, array_info_s, sentrix_id);

                    // Check the existence of Red and Grn IDAT files
                    if fs::metadata(&red_idat_path).is_ok() && fs::metadata(&grn_idat_path).is_ok() {
                        //println!("Node {}: Found Red IDAT file: {}", rank, red_idat_path);
                        //println!("Node {}: Found Grn IDAT file: {}", rank, grn_idat_path);

                        // Read data from Red and Grn IDAT files
                        match (
                            read_idat_values(&red_idat_path, "Red"),
                            read_idat_values(&grn_idat_path, "Green"),
                        ) {
                            (Ok((probe_ids, red_data)), Ok((_, grn_data))) => {
                                // Lock the shared data vector and push the IdatData
                                let store: Vec<(f64, f64)> = red_data.iter().zip(grn_data.iter()).map(|(&a, &b)| (a, b)).collect();
                                data = store;

                                if *read_ids {
                                    let mut store_id = ids.lock().unwrap();
                                    *store_id = probe_ids;
                                }
                            }
                            _ => {
                                println!("Error reading IDAT files.");
                            }
                        }

                    } else {
                        println!("Node {}: Red IDAT file not found: {}", rank, red_idat_path);
                        println!("Node {}: Grn IDAT file not found: {}", rank, grn_idat_path);
                    }
                }
            }
        }
    }
    data
}

// Function construct the directory path for the red intensity idat for an individial
fn construct_red_idat_path(
    idat_directory: &str,
    batch_comment_cleaned: &str,
    array_info_s: &str,
    sentrix_id: &str,
) -> String {
    format!(
        "{}/{}/{}/{}_{}_Red.idat",
        idat_directory, batch_comment_cleaned, array_info_s, array_info_s, sentrix_id
    )
}

// Function construct the directory path for the green intensity idat for an individial
fn construct_grn_idat_path(
    idat_directory: &str,
    batch_comment_cleaned: &str,
    array_info_s: &str,
    sentrix_id: &str,
) -> String {
    format!(
        "{}/{}/{}/{}_{}_Grn.idat",
        idat_directory, batch_comment_cleaned, array_info_s, array_info_s, sentrix_id
    )
}

fn read_manifest_file(manifest_directory: &str, addresses: &mut Vec<u32>, bead_set_id: &mut Vec<i32>, unique_bead_set_ids: &mut  Vec<i32>) -> Result<(), std::io::Error> {
    let manifest_file = std::fs::File::open(manifest_directory)?;
    let mut is_first_line = true;
    let manifest_lines = std::io::BufReader::new(manifest_file).lines().skip(7);
    let mut address_a_id_index = None;
    let mut address_b_id_index = None;
    let mut bead_set_id_index = None;
    let mut number_of_fields = None;

    for manifest_line in manifest_lines {
        let manifest_line = manifest_line?;
        let manifest_fields: Vec<&str> = manifest_line.split(',').collect();

        if is_first_line {
            // Iterate over the manifest fields to find indexes
            for (index, field) in manifest_fields.iter().enumerate() {
                match *field {
                    "AddressA_ID" => {
                        address_a_id_index = Some(index);
                    }
                    "AddressB_ID" => {
                        address_b_id_index = Some(index);
                    }
                    "BeadSetID" => {
                        bead_set_id_index = Some(index);
                    }
                    _ => {}
                }
            }
            number_of_fields = Some(manifest_fields.len());
            is_first_line = false; // Set is_first_line to false after processing the first line
            continue;
        }

        if let Some(number_of_fields) = number_of_fields {
            if manifest_fields.len() >= number_of_fields {
                // Determining the unique individual beadsetIDs for address A
                if let Some(address_a_id_index) = address_a_id_index {
                    if let Ok(address) = manifest_fields.get(address_a_id_index).unwrap().parse::<u32>() {
                        if let Some(bead_set_id_index) = bead_set_id_index {
                            if let Ok(beadset_id) = manifest_fields.get(bead_set_id_index).unwrap().parse::<i32>() {
                                addresses.push(address);
                                bead_set_id.push(beadset_id);

                                if !unique_bead_set_ids.contains(&beadset_id) {
                                    unique_bead_set_ids.push(beadset_id);
                                }
                            }
                        }
                    }
                }

                // Determining the unique individual beadsetIDs for address B if it exists
                if !manifest_fields[address_b_id_index.unwrap()].is_empty() {
                    if let Some(address_b_id_index) = address_b_id_index {
                        if let Ok(address_col6) = manifest_fields.get(address_b_id_index).unwrap().parse::<u32>() {
                            if let Some(bead_set_id_index) = bead_set_id_index {
                                if let Ok(beadset_id) = manifest_fields.get(bead_set_id_index).unwrap().parse::<i32>() {
                                    addresses.push(address_col6);
                                    bead_set_id.push(beadset_id);

                                    if !unique_bead_set_ids.contains(&beadset_id) {
                                        unique_bead_set_ids.push(beadset_id);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Create a new vector that combines addresses and bead_set_id
    let mut combined: Vec<(u32, i32)> = addresses.iter().cloned().zip(bead_set_id.iter().cloned()).collect();

    // Sort the combined vector by addresses in ascending order
    combined.sort_by_key(|&(address, _)| address);

    // Update the addresses and bead_set_id vectors with the sorted values
    addresses.clear();
    bead_set_id.clear();

    for (addr, bead_id) in combined {
        addresses.push(addr);
        bead_set_id.push(bead_id);
    }

    Ok(())
}

// Function initialises the vectors in a hash map according to their Beadset and stores the indexes of the probe IDs in the idat file for each beadset
fn initialize_vectors(vector_names: &mut Vec<i32>) -> (
    Arc<Mutex<HashMap<i32, Vec<i32>>>>,
    Arc<Mutex<HashMap<i32, Vec<u32>>>>,
) {
    let vectors_ind_shared: HashMap<i32, Vec<i32>> = HashMap::new();
    let vectors_ind = Arc::new(Mutex::new(vectors_ind_shared));

    let vectors_id_shared: HashMap<i32, Vec<u32>> = HashMap::new();
    let vectors_ids = Arc::new(Mutex::new(vectors_id_shared));

    for name in vector_names {
        let initial_vector2: Vec<i32> = Vec::new();
        vectors_ind.lock().unwrap().insert(*name, initial_vector2);

        let intial_vector3: Vec<u32> = Vec::new();
        vectors_ids.lock().unwrap().insert(*name, intial_vector3);
    }

    (Arc::clone(&vectors_ind), Arc::clone(&vectors_ids))
}

// Intialise memory to store the data intensities into their associated beadsetIDs
fn initialize_storage(vector_names: &Arc<Mutex<Vec<i32>>>) -> 
    HashMap<i32, Vec<(f64, f64)>>
{
    let mut vectors: HashMap<i32, Vec<(f64, f64)>> = HashMap::new();
    let vector_names = vector_names.lock().unwrap();

    for name in vector_names.iter() {
        let initial_vector1: Vec<(f64, f64)> = Vec::new();
        vectors.insert(*name, initial_vector1);
    }

    vectors
}


// Function populate the group vectors for indexes with the indexes they should extract data from in the color intensity files
fn process_vectors(
    vectors: &Arc<Mutex<HashMap<i32, Vec<i32>>>>,
    vector_ids: &Arc<Mutex<HashMap<i32, Vec<u32>>>>,
    idds: &Arc<Mutex<Vec<u32>>>,
    address: &Vec<u32>,
    bead_set_id: &Vec<i32>,
) {
    let ids = idds.lock().unwrap();
    let mut vectors_ind = vectors.lock().unwrap();
    let mut vectors_ids = vector_ids.lock().unwrap();

    let mut found: usize = 0;
    for (index, iid) in ids.iter().enumerate() {
        for (addr_index, addr) in address.iter().enumerate().skip(found) {
            if iid < addr {
                break;
            }

            if iid == addr {
                //Store the index at which the particular address is found in associated beadset vector
                if let Some(index_vec) = vectors_ind.get_mut(&bead_set_id[addr_index]) {
                    index_vec.push(index as i32);
                }

                // Store the actual address for that index in the associated beadset vector
                if let Some(index_vector) = vectors_ids.get_mut(&bead_set_id[addr_index]) {
                    index_vector.push(*iid);
                }
                break;
            }
            found += 1;
        }
    }
}

// Function split the individual data and store them according to their BeadSetIDs
fn populate_vectors(
    vectors_grp: &mut HashMap<i32, Vec<(f64, f64)>>,
    vectors_ind_map: &Arc<Mutex<HashMap<i32, Vec<i32>>>>,
    idat_data: &Vec<(f64,f64)>,
) {

    // let mut vectors_grp = vector_store.lock().unwrap();
    let vectors_ind_map = vectors_ind_map.lock().unwrap();

    //The positions of the data for each beadsetid is pre-defined (already determined)
    for (group, positions) in vectors_ind_map.iter() {
        let red_val: Vec<f64> = positions.iter().map(|&index| idat_data[index as usize].0).collect();
        let grn_val: Vec<f64> = positions.iter().map(|&index| idat_data[index as usize].1).collect();
        let store: Vec<(f64, f64)> = red_val.iter().zip(grn_val.iter()).map(|(&a, &b)| (a, b)).collect();
        vectors_grp.entry(*group).or_insert(Vec::new()).extend(store);
    }

}

fn recontruct_individual_vector(
    vector_store: &mut HashMap<i32, Vec<(f64, f64)>>,
    vectors_ids: &Arc<Mutex<HashMap<i32, Vec<u32>>>>,
    vector_names: &Arc<Mutex<Vec<i32>>>,
) -> Vec<(f64, f64)> {
    // Create a single vector to store all the combined data
    let mut combined_data: Vec<(u32, f64, f64)> = Vec::new();

    // Lock the data structures outside the loop to improve efficiency and avoid deadlocks
    let mut vectors_ids = vectors_ids.lock().unwrap();
    let vector_names = vector_names.lock().unwrap();

    for vector_name in vector_names.iter() {
        if let Some(vector) = vector_store.get_mut(vector_name) {
            if let Some(ids) = vectors_ids.get_mut(vector_name) {
                // Combine the data with their corresponding IDs and collect into a separate vector
                let combined: Vec<(u32, f64, f64)> = ids
                    .iter()
                    .cloned()
                    .zip(vector.iter().cloned())
                    .map(|(id, data)| (id, data.0, data.1))
                    .collect();
                    
                // Extend the combined data into the combined_data vector
                combined_data.extend_from_slice(&combined);
            }
        }
    }

    // Sort the combined_data in ascending order of IDs
    combined_data.sort_by_key(|(id, _, _)| *id);

    // Remove the IDs column and return a Vec<(f64, f64)>
    let data_without_ids: Vec<(f64, f64)> = combined_data.iter().map(|(_, x, y)| (*x, *y)).collect();

    data_without_ids
}


pub fn main_processing(_world: &SystemCommunicator, rank: i32, size: i32, line_count: &mut i32, store_lines: &mut Vec<i32>) -> Result<Vec<Vec<(f64, f64)>>, io::Error> {

    // Code is set to requie atleast 4 nodes
    if size < 4 {
        println!("Error: This program requires at least 4 processes.");
    }

    // Read command-line arguments
    let args: Vec<String> = env::args().collect();

    // Command-line arguments
    let sample_sheet_file = &args[1];
    let idat_directory = &args[2];
    let manifest_directory = &args[3];

    let mut addresses: Vec<u32> = Vec::new(); // Stores the probe addresses from the manifest file
    let mut bead_set_id: Vec<i32> = Vec::new(); // Stores the BeadSetID associated with the probe addresses from the manifest file

    let ids: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new())); // Stores the probe ids from the idat files
    // Each node only need to store when processing the first individual

    let shared_idat_directory = Arc::new(idat_directory.to_string());
    let shared_sample_sheet_file = Arc::new(sample_sheet_file.to_string());
    let mut is_first_line = true; // Flag to check if it's the first line
    let shared_sample_sheet_file = Arc::clone(&shared_sample_sheet_file);
    let mut shared_bool = true; // Flag variable used to determine if the nodes are processing the first individual or not
    let mut vector_names: Vec<i32> = Vec::new(); // Stores the different beadsetIDs without repeatition

    // Opted for each node to read the manifest file on its own, rather than having the root node reading and sharing.
    // All the other nodes must wait for the root node to process and also increase computation time since the master node must now broadcast its results
    let _ = read_manifest_file(&manifest_directory, &mut addresses, &mut bead_set_id, &mut vector_names);
    
    // Vectors_ind_map stores the indexes of the probe addresses each beadsetID group will be extracting from each individual
    // Vector_ids stores the actual addresses from at that specific index for each beadSetID. These ids are used to reconstruct the data of the individual from beadsetID groups
    let (vectors_ind_map, vector_ids) = initialize_vectors(&mut vector_names);

    // Variables from the sample sheet used to construct the directory for each individual
    let mut batch_comment  = None;
    let mut array_info_s = None;
    let mut sentrix_id = None;

    // Sharing across threads
    let vector_names: Arc<Mutex<Vec<i32>>> = Arc::new(Mutex::new(vector_names)); 
    let mut number_of_lines_per_node = 0;

    // Stores the data for the individuals processed by a node
    let all_individuals: Arc<Mutex<Vec<Vec<(f64, f64)>>>> = Arc::new(Mutex::new(Vec::new())); 

    if let Ok(file) = File::open(&*shared_sample_sheet_file) {
        let reader = std::io::BufReader::new(file);
        // Open and read the sample sheet file
        println!("Node {}: Processing the Sample Sheet...", rank);
        // Create a vector to store thread handles with explicit type annotation
        let mut handles: Vec<JoinHandle<()>> = Vec::new();
        let mut num = 1;
        for line in reader.lines() {
            let line = line.unwrap();

            // Use the first line to determine the columns to be used to construct the directory to individuals data
            if is_first_line {
                is_first_line = false; // Set the flag to false after processing the first line
                let file_columns: Vec<&str> = line.split(',').collect();

                for (index, field) in file_columns.iter().enumerate() {
                    match *field {
                        "Batch Comment" => {
                            batch_comment = Some(index);
                        }
                        "Array Info.S" => {
                            array_info_s = Some(index);
                        }
                        "Sentrix ID" => {
                            sentrix_id = Some(index);
                        }
                        _ => {}
                    }
                }

                continue; // Skip the first line (it is the row with the column names)
            }

            if *line_count > 20 {
                break;
            }

            // Each mpi process will process a set number of inviduals 
            // Some mpi processes might process at most one more individual than the other processes since the individuals might not be evenly distributive across the mpi processes
            if *line_count % size == rank {
                // Initialising the variables for sharing across threads
                let vector_names = Arc::clone(&vector_names);
                let shared_idat_directory = Arc::clone(&shared_idat_directory);
                let ids = Arc::clone(&ids);
                let all_individuals = Arc::clone(&all_individuals);
                let vectors_ind_map = Arc::clone(&vectors_ind_map);
                let vector_ids = Arc::clone(&vector_ids);

                // Use the first individual in the to process the vectors_ind_map, vectors_ids, and ids
                // There is no need to perform this operation more than once
                if shared_bool {
                    // Initialise the vector to store the data in beadsetID
                    // Initialising every time because we had a problem when the vector was being shared across the threads which is a problem
                    let mut vectors = initialize_storage(&vector_names);
                    let shared_data = process_sample_sheet_line(&line, &shared_idat_directory, rank, size, &ids, &shared_bool, &batch_comment, &array_info_s, &sentrix_id);
                    shared_bool = false;

                    // Not necessary but just redundancy in ensuring that the process_vectors is only called once 
                    if num == 1 {
                        process_vectors(&vectors_ind_map, &vector_ids, &ids, &addresses, &bead_set_id);
                    }

                    // Populate the vectors variable with the data for each beadsetID
                    populate_vectors(&mut vectors, &vectors_ind_map, &shared_data);
                    num = 2;

                    // Normalise the data intensities across beadSet
                    let _ = Normalise::within_beadset_normalisation(&mut vectors, &vector_names);

                    // Combine the data to make one individual given the data in beadsetIDs for that individual
                    let ind_vec = recontruct_individual_vector(&mut vectors, &vector_ids, &vector_names);

                    // Store the processed individual
                    let mut all_individuals =  all_individuals.lock().unwrap();
                    all_individuals.push(ind_vec.clone());
                    println!("Print Done");
                }else{

                    let handle = thread::spawn(move || {
                        let mut vectors = initialize_storage(&vector_names);
                        let shared_data = process_sample_sheet_line(&line, &shared_idat_directory, rank, size, &ids, &false, &batch_comment, &array_info_s, &sentrix_id);
                        println!("Print {}", shared_data.len());
                        populate_vectors(&mut vectors, &vectors_ind_map, &shared_data);
                        let _ = Normalise::within_beadset_normalisation(&mut vectors, &vector_names);
                        let ind_vec = recontruct_individual_vector(&mut vectors, &vector_ids, &vector_names);
                        let mut all_individuals =  all_individuals.lock().unwrap();
                        all_individuals.push(ind_vec.clone());
                    });
                    handles.push(handle);
                    
                }
                // Records the number of individuals the node processed
                number_of_lines_per_node = number_of_lines_per_node + 1;
            }
            
            *line_count += 1;
        }
        
        // Wait for all threads to finish
        for handle in handles {
            handle.join().unwrap();
        }
        
    }

    _world.barrier();
    println!("Node {}: Sample Sheet successfully processed...", rank);

    // Node send to the master node, the number of individuals that it processed
    // Vital for the master node to have this information so that it knows how many individuals it will recieve from the particular node ahead of time
    if rank != 0 {
        // Send a single variable from process with rank != 0 to process 0
        _world.process_at_rank(0).send(&number_of_lines_per_node);
    }

    if rank == 0 {
        // Receive the single variable sent by other processes
        for i in 1..size {
            let (msg, _status) = _world.process_at_rank(i).receive::<i32>();
            store_lines.push(msg);
        }
    }

    store_lines.push(number_of_lines_per_node);
    _world.barrier();
    let result = Ok(all_individuals.lock().unwrap().clone());

    //Return Hashmaps of vectors
    result
}

//Function for normalisation within SNP across all the individuals
pub fn snp_normalisation(individuals: &Vec<Vec<(f64,f64)>>) {
    println!("Normalising Across SNPs...");

    for single_individual in individuals {
        let people: Arc<Mutex<Vec<(f64,f64)>>> = Arc::new(Mutex::new(single_individual.to_vec()));
        let _ = Normalise::within_snp_normalisation(&people);
    }

    println!("Normalisation Across SNPs complete...");
}

// pub fn snp_normalisation(individuals: &Vec<Vec<(f64,f64)>>) {
//     println!("Normalising Across SNPs... {}", individuals.len());

//     individuals.par_iter().for_each(|single_individual| {
//         let people = Arc::new(Mutex::new(single_individual.to_vec()));
//         let _ = Normalise::within_snp_normalisation(&people);
//     });

//     println!("Normalisation Across SNPs complete...");
// }

// Function to transpose the matrix of individuals data
pub fn transpose_matrix_in_place(matrix: &mut Vec<Vec<f64>>) {

    println!("Transposition of All individuals Matrix...");
    // Check if the matrix is empty
    if matrix.is_empty() {
        return;
    }

    let rows = matrix.len();
    let cols = matrix[0].len();

    let mut transposed = vec![vec![0.0; rows]; cols]; // Create a new matrix with transposed dimensions

    for i in 0..rows {
        for j in 0..cols {
            transposed[j][i] = matrix[i][j]; // Transpose elements
        }
    }

    *matrix = transposed; // Update the original matrix with the transposed data

    println!("Transposition of All individuals Matrix Completed...");
}
