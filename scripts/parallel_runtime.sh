#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --time=2:00:00
#SBATCH --export=NONE
#SBATCH --job-name=stability-bshrt
#SBATCH --output=stability-bshrt_log.o

# Create the results directory if it doesn't exist
module load python/mpi4py-3.1.1py3.9

mkdir -p results

values_proc=(1 4 16 64)

output_file="results/parallel_runtime_65536_200.csv"
echo "n, trunc, l, norm, time, decay, comms" > "$output_file"  # Write the header to the CSV file

for proc in "${values_proc[@]}"; do

    output=$(srun -n"$proc" python src\\mpi_main.py 65536 200 )
    echo "$output" >> "$output_file"
    fi
done
    

echo "All results have been saved in the 'results' directory."