#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 --method <method> --tasks <tasks> --tag <tag>"
    exit 1
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --method) method="$2"; shift ;;
        --tasks) tasks="$2"; shift ;;
        --tag) tag="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Function to run evaluate_chem_folder.py with a given method and timestamp
run_evaluation() {
    local method=$1
    local tasks=$2
    local tag=$3 
    local base_result_path="../results/${tasks}/${method}/${tag}"

    echo "Running evaluation for method: ${method}"
    echo "Base result path: ${base_result_path}"
    
    python evaluate_geom_folder.py --base_result_path ${base_result_path} 

    echo "Evaluation for method: ${method} completed."
    echo "--------------------------------------------------"

    python cal_geom_results.py --base_result_path ${base_result_path}
    echo "Calculation for method: ${method} completed."
    echo "--------------------------------------------------"
}

# Check if all required arguments are provided
if [ -z "$method" ] || [ -z "$tasks" ] || [ -z "$tag" ]; then
    usage
fi

# Run the evaluation for the given method, task, and tag
run_evaluation ${method} ${tasks} ${tag}
