#!/bin/bash

# This script runs the data generation and training Python scripts
# in a loop for a specified number of iterations.

NUM_ITERATIONS=100

echo "Starting training loop for $NUM_ITERATIONS iterations..."

for (( i=1; i<=$NUM_ITERATIONS; i++ ))
do
  echo "--- Iteration $i of $NUM_ITERATIONS ---"
  
  # Step 1: Run the data generation script
  echo "Running data_gcp.py..."
  python data_gcp.py
  
  # Check if the data script failed
  if [ $? -ne 0 ]; then
    echo "Error: data_gcp.py failed on iteration $i. Exiting."
    exit 1
  fi
  
  # Step 2: Run the training script
  echo "Running train_gcp.py..."
  python train_gcp.py
  
  # Check if the training script failed
  if [ $? -ne 0 ]; then
    echo "Error: train_gcp.py failed on iteration $i. Exiting."
    exit 1
  fi
  
  echo "--- Completed iteration $i ---"
  echo "" # Add a blank line for readability
done

echo "Training loop completed all $NUM_ITERATIONS iterations."