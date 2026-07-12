#!/bin/bash

SHARE_DIR="/mnt/hgfs/share"
# Define lists of input and output files
INPUT_FILES=("inequ" "eq_transp.F")
OUTPUT_FILES=("psi_xz.dat" "q_p_g.dat" "wch.dat" "ploteq.pdf")

while true; do
    # --- Phase A: Wait for input files to appear ---
    # Break the wait loop only when both input files exist
    while true; do
        ls "$SHARE_DIR" > /dev/null 2>&1
        if [[ -f "$SHARE_DIR/inequ" && -f "$SHARE_DIR/eq_transp.F" ]]; then
            echo "input file have been prepared, starting calculation"
            break
        fi
        # Check every 2 seconds to avoid excessive resource consumption
        sleep 2
    done

    # --- Phase B: Fetch files and prepare environment ---
    # Copy files from share to the current directory
    cp "$SHARE_DIR/inequ" ./
    cp "$SHARE_DIR/eq_transp.F" ./
    rm "$SHARE_DIR/inequ"
    rm "$SHARE_DIR/eq_transp.F"
    
    # Clean up folder to prevent interference from old data
    for f in "${INPUT_FILES[@]}" "${OUTPUT_FILES[@]}"; do
        rm -f "$f"
    done

    # --- Phase C: Execute calculation script ---
    echo "----------------------------------------"
    echo "run equilibrium.sh ..."
    sh equilibrium.sh

    # --- Phase D: Send back results ---
    echo "calculation completed, copying results to share ..."
    
    # Check if result files exist; copy if they do
    SUCCESS_COUNT=0
    for f in "${OUTPUT_FILES[@]}"; do
        if [ -f "$f" ]; then
            cp "$f" "$SHARE_DIR/"
            ((SUCCESS_COUNT++))
        else
            echo "Warning: Generated file $f not found"
        fi
    done

    if [ $SUCCESS_COUNT -eq ${#OUTPUT_FILES[@]} ]; then
        echo ">>> Current iteration successful, continuing to listen for the next set..."
    else
        echo ">>> Current iteration finished (some files missing), continuing to listen..."
    fi
    
    echo "----------------------------------------"
done