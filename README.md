# cuda_irregular_ops

The **cuda_irregular_ops** repository focuses on cross-conversion between programming languages for different deep learning processors. It includes implementations of irregular tensor operators in **CUDA C**, **AVX VNNI**, and **BANG C**. 
The overall workflow involves: 
1. **Preprocessing**: Converting platform-specific code into standard C code. This step removes features tied to particular hardware platforms, allowing for more generic handling. 

2. **Loop Processing**: Applying various loop transformations to improve the performance of operators, such as loop fusion, loop unrolling, and reordering. 

3. **Post-processing**: Converting the processed C code back into platform-specific code, this time adding SIMD (Single Instruction, Multiple Data) instructions where applicable. Final optimizations are applied to maximize performance on the target hardware. 

This repository streamlines the translation process across different processor architectures, ensuring efficient computation on deep learning processors through both architecture-specific and generic optimizations.

# Install
Install is similar to tvm. First, fill in USE_CUDA and USE_LLVM in cmake/config.cmake, like this:

set(USE_LLVM "/path/to/llvm-config --link-static")
set(HIDE_PRIVATE_SYMBOLS ON)
set(USE_CUDA /usr/local/cuda)

Furthermore, to optimize the MCTS algorithm, the following Python packages are necessary:
```
pip install chex jax mctx
```

# Language reference
Still in progress.

See test cases and tutorials.