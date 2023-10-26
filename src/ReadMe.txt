>> Set up the node: srun --pty bash
>> then import mpi module: module load mpi/openmpi-4
>> then run the script: sbatch batch.sh

-> So I set the code to only read in 8 lines so basically, each node will read only 2 individuals because I set 
up the process to have 2 nodes and 2 mpi processes per node and 6 cores per mpi process.
-> I did this to make it run fast during testing on your side.
-> The variable called vectors is the one you have to use (basically it storing the data that you need)

Ohh and P.S: Do not edit any file name, and Yes you can edit the code, but do not edit file names


let result = Normalise::apply_normalisation_parallelised();


-----------------------------

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rustup-init.sh

chmod +x rustup-init.sh
./rustup-init.sh --default-toolchain none -y


export PATH="$HOME/.cargo/bin:$PATH"

rustup install nightly

rustup default nightly

