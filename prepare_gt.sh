#!/bin/bash

# Bash script to place all obj files in HeadspaceOnline into an unprocessed folder ready for obj preprocessing
base_dir="$HOME/../../vol/bitbucket/rqg23/headspaceOnline/subjects"
target_dir="$HOME/../../vol/bitbucket/rqg23/ground_truth_faceCompletion"

# Check if the target directory exists, otherwise make it
mkdir -p "$target_dir"
# Loop through all files/directories in the base directory
for dir in "$base_dir"/*/; do
  # Check that it is a directory (otherwise not a participant)
  if [ -d "$dir" ]; then
    # Save the ID of the study participant
    participant_id=$(basename "$dir")
    # Check for existence of .obj file in this directory
    file_to_copy=$(find "$dir" -type d -exec sh -c 'find "$1" -maxdepth 1 -name "*.obj" -print -quit' _ {} \;)

    # If it exists then copy that file and change the name to the participant_id_gt
    if [ -n "$file_to_copy" ]; then
      cp "$file_to_copy" "$target_dir/${participant_id}_gt.obj"
      echo "Copied $file_to_copy to $target_dir/${participant_id}_gt.obj"
    else
      echo ".obj file not found in $dir"
    fi
  else
    echo "$dir does not exist"
  fi
done