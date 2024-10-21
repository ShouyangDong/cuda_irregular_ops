import re
import openai

from src.prompt.prompt import SYSTEM_PROMPT
from src.post_processing.post_processing_prompt import (
    THREAD_BINDING_DEMO_BANG,
    THREAD_BINDING_DEMO_CUDA,
)

model_name = """gpt-3.5-turbo"""
openai.api_key = "sk-JmlwEmWiNtFqSD7IDaF981Dd8a7447FfBcE768755cB38010"
openai.api_base = "https://api.keya.pw/v1"


THREAD_BINDING_PROMPT = """
Thread Binding

Function Overview:
This prompt is designed to identify parallelizable loops or axes in C++ 
and bind them to the available threads or cores on a GPU or NPU. The prompt helps 
transform the input code by mapping the loops onto specific hardware resources like GPU threads or NPU cores 
to enable parallel computation.

Application Scenario:
Use this prompt when you want to parallelize a computational task by binding one or more axes of a loop (e.g., batch size, spatial dimensions, etc.) 
to the available threads or cores in a GPU/NPU. This process accelerates the computation by exploiting the parallel nature of hardware accelerators.

Input:
The input is a C++/CUDA code snippet containing loops that can be parallelized, with the goal of binding these loops to threads or cores on a GPU/NPU. 
The target hardware may have specific clusters, cores, or threads, and the prompt will help map the loop dimensions accordingly.

Output:
The transformed code with appropriate `#pragma` or thread binding directives inserted into the loops, 
ensuring that each iteration of the loop is handled by different threads or cores for parallel execution.


### Steps for Insertion:
1. Identify loops or axes that are candidates for parallel execution. Typically, outer loops or large iterations are ideal for parallelization.
2. Bind these loops to available hardware threads or cores using directives such as `#pragma thread_binding` or directly using CUDA constructs like `threadIdx` and `blockIdx`.
3. For NPU hardware, bind the loops to clusters and cores (e.g., clusterId, coreId).
4. Maintain the code logic, ensuring that the transformed code remains functionally equivalent while parallelizing the computation.

### Example 
{LOOP_RECOVERY_DEMO}

### GPT Task:
Please transform the following C++ or CUDA code by binding the parallel loops to GPU threads or NPU clusters and cores for efficient parallel computation. Insert `#pragma thread_binding` or equivalent GPU/NPU constructs where appropriate.

#### Input Code:
{cpp_code}

#### Output Code with Thread/Cluster Binding:
```

### Notes:
- `{cpp_code}` should be replaced with the actual input C++/CUDA code containing loops that are suitable for parallelization.
- The output should map parallel loops to the hardware resources available on the target device (e.g., GPU threads, NPU clusters/cores).
- The prompt is flexible enough to handle both GPU (CUDA) and NPU-specific architectures.

"""


def run_THREAD_BINDING(code, target):
    PROMPT = """
    {SYSTEM_PROMPT}
    
    {THREAD_BINDING_PROMPT}
    
    Please return the output kernel function without any additional information.
    """

    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)
    prompt_demo = None
    if target == "CUDA":
        prompt_demo = THREAD_BINDING_DEMO_CUDA
    elif target == "BANG":
        prompt_demo = THREAD_BINDING_DEMO_BANG

    PROMPT = PROMPT.replace("{THREAD_BINDING_PROMPT}", THREAD_BINDING_PROMPT)
    PROMPT = PROMPT.replace("{LOOP_RECOVERY_DEMO}", prompt_demo)
    PROMPT = PROMPT.replace("{cpp_code}", code)
    transformation_completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": PROMPT}],
    )

    content = transformation_completion.choices[0].message["content"]
    match = re.search(r"\`\`\`(.*?)\`\`\`", content, re.DOTALL)
    if match:
        code_content = match.group(1)
        return code_content
    return None


if __name__ == "__main__":
    code = """
    extern "C" void add_kernel(float* output, float* input1, float* input2) {
        int dim1 = 4;
        int dim2 = 4;
        int dim3 = 4;
        int dim4 = 64;
        
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                for (int k = 0; k < dim3; k++) {
                    for (int l = 0; l < dim4; l++) {
                        int index = i * dim2 * dim3 * dim4 + j * dim3 * dim4 + k * dim4 + l;
                        output[index] = input1[index] + input2[index];
                    }
                }
            }
        }
    }
    """
    output_code = run_THREAD_BINDING(code, "BANG")
    print(output_code)

    code = """
    extern "C" void  add_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_add) {
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 1024; j++) {
                if (((i * 1024) + j) < 2309) {
                    T_add[((i * 1024) + j)] = (A[((i * 1024) + j)] + B[((i * 1024) + j)]);
                }
            }
        }
    }
    """
    code = run_loop_recovery(code, target="CUDA")
    print(code)
