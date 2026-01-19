#!/bin/bash

BUILD_DIR="./build"

tests=(
    "conv1d_test"
    "relu_test"
    "dropout_test"
    "residual_block_test"
    "tcn_test"
    "cuda_test"
)

echo "========================================"
echo "    PRISMTCN LAYER VERIFICATION SUITE   "
echo "========================================"

if [ ! -d "$BUILD_DIR" ]; then
    echo "❌ Error: Build directory '$BUILD_DIR' not found."
    echo "   Please compile first:"
    echo "   mkdir build && cd build && cmake .. && make -j\$(nproc)"
    exit 1
fi

all_passed=true

for test_bin in "${tests[@]}"; do
    executable="$BUILD_DIR/$test_bin"
    
    if [ -f "$executable" ]; then
        echo -n "Testing $test_bin ... "
        
        output=$("$executable" 2>&1)
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
        if [ "$test_bin" == "cuda_test" ]; then
            echo "⚠️  Skipping cuda_test (Binary not found - CUDA disabled?)"
        else
            echo "❌ Error: $test_bin not found in $BUILD_DIR"
            all_passed=false
        fi
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
