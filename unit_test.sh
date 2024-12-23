source env.sh

echo "Running CPP tests..."

# echo "==============CPP Compilation Test==============="
# python benchmark/evaluation/cpp_test/compilation.py
# echo "==============CPP Computation Test==============="
# python benchmark/evaluation/cpp_test/result_test.py
# Check for NVIDIA GPU presence
if cnmon > /dev/null 2>&1; then
    # echo "Cambricon MLU detected. Running BANG tests..."

    # echo "==============MLU Compilation Test==============="
    # python benchmark/evaluation/mlu_test/compilation.py
    # echo "==============MLU Computation Test==============="
    # python benchmark/evaluation/mlu_test/result_test.py

    echo "==============DL Boost Compilation Test==============="
    python benchmark/evaluation/dlboost_test/compilation.py
    echo "==============DL Boost Computation Test==============="
    python benchmark/evaluation/dlboost_test/result_test.py
fi
# Check for NVIDIA GPU presence
if nvidia-smi > /dev/null 2>&1; then
    echo "NVIDIA GPU detected. Running CUDA tests..."

    echo "==============GPU Compilation Test==============="
    python benchmark/evaluation/cuda_test/compilation.py
    echo "==============GPU Computation Test==============="
    python benchmark/evaluation/cuda_test/result_test.py
fi
