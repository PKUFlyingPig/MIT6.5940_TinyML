#!/bin/bash

# List of implementation
# 0: reference
# 1: loop_unrolling
# 2: multithreading
# 3: simd_programming
# 4: multithreading_loop_unrolling
# 5: all_techniques
keys=("reference" "loop_unrolling" "multithreading" "simd_programming" "multithreading_loop_unrolling" "all_techniques")
values=("0" "1" "2" "3" "4" "5")

# If a implementation is provided to the script, map it to the corresponding argument
if [ "$#" -eq 1 ]; then
    found=0
    for i in "${!keys[@]}"; do
        if [ "${keys[$i]}" = "$1" ]; then
            test_args=("${values[$i]}")
            found=1
            break
        fi
    done
    if [ "$found" -eq 0 ]; then
        echo "Invalid implementation. Please provide a valid key from the mapping."
        exit 1
    fi
else
    # If no argument is provided, use all values
    test_args=("${values[@]}")
fi


# Run the program with different arguments
for arg in "${test_args[@]}"; do
    make clean
    make chat test_linear -j IMP="$arg"
    # Check if make was successful
    if [ $? -ne 0 ]; then
        echo "Compilation failed!"
        exit 1
    fi
    ./test_linear
    echo ""
done

echo "All tests completed!"
