#!/bin/bash

tests=(
    "conv1d_test"
    "relu_test"
    "dropout_test"
    "residual_block_test"
    "tcn_test"
)

echo "========================================"
echo "    PRISMTCN LAYER VERIFICATION SUITE   "
echo "========================================"
echo

all_passed=true

for test_bin in "${tests[@]}"; do
    if [ -f "./$test_bin" ]; then
        echo -n "Testing $test_bin ... "
        
        # Run the test and capture output, silence stdout unless error
        output=$(./$test_bin 2>&1)
        exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            echo "✅ PASS"
        else
            echo "❌ FAIL"
            echo "--- Error Output ---"
            echo "$output"
            echo "--------------------"
            all_passed=false
        fi
    else
        echo "⚠️  Skipping $test_bin (Binary not found. Did compilation succeed?)"
        all_passed=false
    fi
done

echo
echo "========================================"
if [ "$all_passed" = true ]; then
    echo "  ALL SYSTEMS GO. ENGINE READY."
else
    echo "  VERIFICATION FAILED."
    exit 1
fi
echo "========================================"
