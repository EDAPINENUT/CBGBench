#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 --method <method> --task <task> --tag <tag> [--checkpoint <checkpoint>]"
    exit 1
}

# Initialize parameters
method=""
task=""
tag=""
checkpoint=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --method) method="$2"; shift ;;
        --task) task="$2"; shift ;;
        --tag) tag="$2"; shift ;;
        --checkpoint) checkpoint="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Check if all required arguments are provided
if [ -z "$method" ] || [ -z "$task" ] || [ -z "$tag" ]; then
    usage
fi

checkpoint_path=./logs/${task}/${method}/${tag}/checkpoints

# If no checkpoint is provided, find the latest checkpoint
if [ -z "$checkpoint" ]; then
    pt_files=($(ls ${checkpoint_path}/*.pt 2>/dev/null))

    if [ ${#pt_files[@]} -eq 0 ]; then
        echo "No .pt files found in the checkpoints directory."
        exit 1
    fi

    max_checkpoint=$(for f in "${pt_files[@]}"; do
        basename "$f" | grep -o -E '[0-9]+' | sort -n | tail -1
    done | sort -n | tail -1)
    echo "max_checkpoint is ${max_checkpoint}"
    checkpoint=$max_checkpoint
fi

# If checkpoint is found or provided, run the sample.py script
if [ -n "$checkpoint" ]; then
    python sample.py --config ./configs/${task}/test/${method}.yml --out_root ./results/${task}/ --tag ${tag} --checkpoint ${checkpoint}
else
    echo "No valid checkpoint found."
    exit 1
fi

##