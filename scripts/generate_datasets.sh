#!/bin/bash

# Default path to "datasets" if no argument is provided
Path="./datasets"
if [ "$1" ]; then
    Path="$1"
fi

Path_raw="$Path/raw"
Path_full="$Path/full"
Path_tmp="$Path/_tmp"

Download_Archive() {
    local Url_endpoint="$1"
    local OutputFileName="$2"
    local TempPath="$3"

    local BaseUrl="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/"
    local Url="$BaseUrl$Url_endpoint"

    # Ensure the temp path exists
    if [ ! -d "$TempPath" ]; then
        mkdir -p "$TempPath"
    fi

    # Construct the full output path
    local OutputFilePath="$TempPath/$OutputFileName"

    # Download the file
    curl -o "$OutputFilePath" "$Url"
    echo "Downloaded archive to '$OutputFilePath'."

    echo "$OutputFilePath"
}

if [ ! -d "$Path" ]; then
    # Create directories if they do not exist
    mkdir -p "$Path" "$Path_raw" "$Path_full"
    echo "Created directories at $Path."

    # Download archives
    Download_Archive "mnist.scale.bz2" "mnist.scale.bz2" "$Path_tmp"
    Download_Archive "mnist.scale.t.bz2" "mnist.scale.t.bz2" "$Path_tmp"

    echo "Uncompressing..."
    bunzip2 -ck "$Path_tmp/mnist.scale.bz2" > "$Path_raw/mnist.scale"
    bunzip2 -ck "$Path_tmp/mnist.scale.t.bz2" > "$Path_raw/mnist.scale.t"

    echo "Joining..."
    cat "$Path_raw/mnist.scale" > "$Path_full/mnist.scale.full"
    cat "$Path_raw/mnist.scale.t" >> "$Path_full/mnist.scale.full"

    # Remove temporary files
    rm -rf "$Path_tmp"
    echo "Datasets built successfully."
else
    echo "Error: Folder '$Path' already exists."
fi
