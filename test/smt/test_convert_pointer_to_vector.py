import re


def convert_pointer_to_vector(code):
    # Step 1: Replace pointer declarations with vector declarations
    code = re.sub(
        r"float\s*\*([A-Za-z_]\w*)", r"vector<vector<float>> \1", code
    )

    # Step 2: Replace output pointer initialization with vector initialization
    # Identify the result variable and initialize a vector of vectors for it.
    match = re.search(r"vector<vector<float>>\s+(\w+);", code)
    if match:
        result_var = match.group(1)
        init_statement = f"vector<vector<float>> {result_var}(rows, vector<float>(cols, 0));\n"
        code = re.sub(
            rf"vector<vector<float>>\s+{result_var};", init_statement, code
        )

    # Step 3: Convert matrix access
    # Replace accesses like A[i * N + j] with A[i][j]
    code = re.sub(
        r"(\w+)\[(\w+)\s*\*\s*(\w+)\s*\+\s*(\w+)\]", r"\1[\2][\4]", code
    )

    # Step 4: Replace for loop initializations to automatically use vector size
    code = re.sub(
        r"for\s*\(\s*int\s+(\w+)\s*=\s*0;\s*\1\s*<\s*(\d+);\s*\1\+\+\s*\)",
        r"for (int \1 = 0; \1 < \2; \1++)",
        code,
    )

    return code


# Original C++ pointer-based code
pointer_code = """
extern "C" void gemm_kernel(float *A, float *B, float *result) {
  for (int j = 0; j < 32; j++) {
    for (int k = 0; k < 128; k++) {
      result[j * 128 + k] = 0;
      for (int l = 0; l < 128; l++) {
        result[j * 128 + k] += A[j * 128 + l] * B[l * 128 + k];
      }
    }
  }
}
"""

# Convert pointer-based code to vector-based code
vector_code = convert_pointer_to_vector(pointer_code)
print(vector_code)
