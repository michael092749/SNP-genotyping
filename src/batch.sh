#!/bin/bash
#SBATCH --job-name=mpi_job
#SBATCH --nodes=2  # Number of nodes
#SBATCH --ntasks-per-node=2  # Number of MPI processes per node
#SBATCH --cpus-per-task=6  # Number of CPU cores per MPI process
#SBATCH --output=output.txt  # Output file
#SBATCH --error=error.txt  # Error file

# Load any necessary modules or set environment variables here
cargo build --release

# Run your Rust MPI program with mpirun
mpirun -n 4 ../target/release/final_code  /dataE/AWIGenGWAS/aux/sample_sheet.csv /dataE/AWIGenGWAS/idats /dataE/AWIGenGWAS/aux/H3Africa_2017_20021485_A3.csv
