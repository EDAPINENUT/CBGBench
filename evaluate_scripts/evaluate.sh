# Function to display usage
usage() {
    echo "Usage: $0 --method <method> --task <task> --tag <tag>"
    exit 1
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --method) method="$2"; shift ;;
        --task) task="$2"; shift ;;
        --tag) tag="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Check if all required arguments are provided
if [ -z "$method" ] || [ -z "$task" ] || [ -z "$tag" ]; then
    usage
fi



bash evaluate_chem.sh --method ${method} --tasks ${task} --tag ${tag}  
bash evaluate_geom.sh --method ${method} --tasks ${task} --tag ${tag}  
bash evaluate_interact.sh --method ${method} --tasks ${task} --tag ${tag}  
bash evaluate_substruct.sh --method ${method} --tasks ${task} --tag ${tag}


## e.g.
# bash evaluate.sh --method targetdiff --tasks denovo --tag selftrain
# bash evaluate.sh --method targetdiff --tasks frag --tag selftrain
# bash evaluate.sh --method targetdiff --tasks linker --tag selftrain
# bash evaluate.sh --method targetdiff --tasks scaffold --tag selftrain
# bash evaluate.sh --method targetdiff --tasks sidechain --tag selftrain