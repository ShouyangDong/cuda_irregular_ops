import re
import openai

model_name = """gpt-3.5-turbo"""
openai.api_key = "sk-JmlwEmWiNtFqSD7IDaF981Dd8a7447FfBcE768755cB38010"
openai.api_base = "https://api.keya.pw/v1"


DECORATION_PROMPT = """
Operation Recognition

Function Overview:
Operation Recognition is designed to identify element-wise or matrix multiplication arithmetic 
operations inside for loops in C++ code and insert the corresponding `#pragma operation( )` directives. 
The inserted pragmas are intended to mark operations for future SIMD (Single Instruction, Multiple Data) vectorization. 
This ensures that element-wise or matrix multiplication calculations can be efficiently transformed into SIMD instructions 
during a later code transformation stage.

### Application Scenario:
Use this prompt when preparing C++ code for SIMD tensorization. It helps identify and mark arithmetic operations inside for loops that operate on individual elements of arrays or matrices. These operations will be optimized and vectorized in the later stages.

### Input:
The input is a C++ code snippet containing for loops with element-wise or matrix multiplication arithmetic operations, where you want to insert `#pragma operation( )` directives before each operation for SIMD vectorization purposes.

### Output:
The transformed C++ code with the `#pragma operation( )` directives inserted before the detected operations inside loops, which marks them for SIMD vectorization.

### Example:

#### Input C++ Code:
```cpp
for (int i = 0; i < 64; i++) {
    C[i] = A[i] + B[i];
}

for (int i = 0; i < 64; i++) {
    C[i] = C[i] * D[i];
}

for (int i = 0; i < 64; i++) {
    E[i] = C[i] - D[i];
}
```

#### Desired Output C++ Code with Pragmas for SIMD Preparation:
```cpp 
#pragma operation(add)
for (int i = 0; i < 64; i++) {
    C[i] = A[i] + B[i];
}
#pragma operation(mul)
for (int i = 0; i < 64; i++) {
    C[i] = C[i] * D[i];
}
#pragma operation(sub)
for (int i = 0; i < 64; i++) {
    E[i] = C[i] - D[i];
}
```

### Steps for Insertion:
1. Identify element-wise or matrix multiplication arithmetic operations inside the for loop such as addition (`+`), subtraction (`-`), multiplication (`*`), and division (`/`).
2. Insert the corresponding `#pragma operation( )` directive directly above each identified operation, specifying the operation type in parentheses (e.g., `#pragma operation(add)` for addition).
3. Focus only on the operations inside loops, as these are the target for SIMD tensorization.
4. Ensure that the structure and logic of the code are not altered, and only relevant element-wise or matrix multiplication operations are annotated.

### GPT Task:
Please transform the following C++ code by inserting `#pragma operation( )` directives above each element-wise or matrix multiplication arithmetic operation inside for loops. These pragmas will be used to prepare the code for SIMD vectorization in a later stage.

#### Input C++ Code:
{cpp_code}

#### Output C++ Code with Pragmas for SIMD Preparation:
```

### Notes:
- The input should be replaced with the actual input C++ code containing loops with element-wise or matrix multiplication operations.
- The output should focus on identifying and marking operations inside loops that are candidates for SIMD vectorization.
"""


def code_decoration(code):
    PROMPT = DECORATION_PROMPT.replace("{cpp_code}", code)
    decoration_completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": PROMPT}],
    )

    content = decoration_completion.choices[0].message["content"]

    match = re.search(r"\`\`\`(.*?)\`\`\`", content, re.DOTALL)
    if match:
        code_content = match.group(1)
        return code_content
    return None


if __name__ == "__main__":
    code = """
    for (int col = 0; col < 64; col++) {
        for (int i = 0; i < 512; i++) {
            B_wram[i * 64 + col] = B[i * 64 + col];

    for (int i = 0; i < 512; i++) {
        A_nram[i] = A[(clusterId * 4 + coreId) * 512 + i];
    }


    for (int col = 0; col < 64; col++) {
        C_nram[(clusterId * 4 + coreId) * 64 + col] = 0.0f;
        for (int i = 0; i < 512; i++) {
            C_nram[col] += A_nram[i] * B_wram[i * 64 + col];
        }
    }

    for (int col = 0; col < 64; col++) {
        C[(clusterId * 4 + coreId) * 64 + col] = C_nram[col];
    }
    """
    code = code_decoration(code)
    print(code)
