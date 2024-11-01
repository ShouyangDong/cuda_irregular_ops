from smt.auto_cache import ast_auto_cache

if __name__ == "__main__":
    code = """
    void add_kernel(float *output, float *input1, float *input2)
    {
        for (int k = 0; k < 4; k++)
        {
            #pragma intrinsic(__bang_add(input[Nram, Nram], output[Nram])))
            for (int l = 0; l < 64; l++)
            {
            output[(((((clusterId * dim2) * dim3) * dim4) + ((coreId * dim3) * dim4)) + (k * dim4)) + l] = input1[(((((clusterId * dim2) * dim3) * dim4) + ((coreId * dim3) * dim4)) + (k * dim4)) + l] + input2[(((((clusterId * dim2) * dim3) * dim4) + ((coreId * dim3) * dim4)) + (k * dim4)) + l];
            }
        }
    }
    """
    space_map = [
        {"input": {"input1": "Nram", "input2": "Nram"}, "output": {"output": "Nram"}}
    ]
    code = ast_auto_cache(code, space_map)
    print(code)
