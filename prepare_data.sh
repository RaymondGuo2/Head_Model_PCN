#!/bin/bash

# Bash script to place all existing 2C images in Headspace into an unprocessed folder ready for MTCNN detection
base_dir="$HOME/../../vol/bitbucket/rqg23/headspacePngTka/subjects"
target_dir="$HOME/../../vol/bitbucket/rqg23/unprocessed_2c"

# Check if the target directory exists, otherwise make it
mkdir -p "$target_dir"
# Loop through all files/directories in the base directory
for dir in "$base_dir"/*/; do
  # Check that it is a directory (otherwise not a participant)
  if [ -d "$dir" ]; then
    # Save the ID of the study participant
    participant_id=$(basename "$dir")
    # Check for existence of 2C.png in the subdirectory of this directory
    file_to_copy=$(find "$dir" -type d -exec sh -c 'find "$1" -maxdepth 1 -name "2C.png" -print -quit' _ {} \;)

    # If it exists then copy that file and change the name to the participant id
    if [ -n "$file_to_copy" ]; then
      cp "$file_to_copy" "$target_dir/$participant_id.png"
      echo "Copied $file_to_copy to $target_dir/$participant_id.png"
    else
      echo "File 2C.png not found in $dir"
    fi
  else
    echo "$dir does not exist"
  fi
done