#!/bin/bash


cmd="python training.py --device cuda"

networks=("unet" "att_unet" "seg_net")

# Loop over all subdirectories in ./data
for dir_path in ./data/*/
do
    # Extract the directory name from the path
    dir_name=$(basename "$dir_path")

    # Loop over all networks and add them to the Python command
    for network in "${networks[@]}"
    do
        # Create the --data argument
        arg="--data $dir_path"

        # Add the current network to the argument
        arg+=" --network $network"

        # Execute the Python command with the data argument and current network
        eval "$cmd $arg"
    done
done
