#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --time=2:00:00
#SBATCH --export=NONE
#SBATCH --job-name=serial-runtime
#SBATCH --output=serial-runtime.o

# Create the results directory if it doesn't exist
module load python/mpi4py-3.1.1py3.9

mkdir -p results

values_dim=(1024 2048 4096 8192 16384 32768 65536)

output_file="results/serial_runtime_200.csv"
echo "n, trunc, l, norm, time, decay, comms" > "$output_file"  # Write the header to the CSV file

for dim in "${values_dim[@]}"; do

    output=$(srun --cpu-freq=2000000 -n1 python src/mpi_main.py "$dim" 200)
    echo "$output" >> "$output_file"
done
    

echo "All results have been saved in the 'results' directory."