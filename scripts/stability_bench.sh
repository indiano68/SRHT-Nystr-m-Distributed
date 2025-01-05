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

# Define the values for a and b
values_l=(600 1000 2000)
values_trunc=(200 400 600 800 1000)

# Loop through each value of a
for l in "${values_l[@]}"; do
    # Create a results file for the current a
    output_file="results/result_${l}.csv"
    echo "n, trunc, l, norm, time, decay, comms" > "$output_file"  # Write the header to the CSV file
    
    # Loop through each value of b
    for trunc in "${values_trunc[@]}"; do
        if [ "$trunc" -le "$l" ]; then
            # Run the Python script with the current values of a and b
            output=$(srun -n64 python src/mpi_main.py 65536 "$trunc" "$l")

            # Append the result to the CSV file
            echo "$output" >> "$output_file"
        fi
    done
    
done

echo "All results have been saved in the 'results' directory."