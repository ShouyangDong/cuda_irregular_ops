from smt.tensorization.tensorization import ast_tensorization

if __name__ == "__main__":
    code = """
    void add_kernel(float *output, float *input1, float *input2)
    {
    float input1_Nram[64];
    float input2_Nram[64];
    float output_Nram[64];
    for (int k = 0; k < 4; k++)
    {
        for (int l = 0; l < 64; l++)
        {
        input1_Nram[l] = input1[(((((clusterId * dim2) * dim3) * dim4) + ((coreId * dim3) * dim4)) + (k * dim4)) + l];
        }

        for (int l = 0; l < 64; l++)
        {
        input2_Nram[l] = input2[(((((clusterId * dim2) * dim3) * dim4) + ((coreId * dim3) * dim4)) + (k * dim4)) + l];
        }

        #pragma operation(add(input[input1_Nram, input2_Nram], output[output_Nram]))
        for (int l = 0; l < 64; l++)
        {
        output_Nram[l] = input1_Nram[l] + input2_Nram[l];
        }

        for (int l = 0; l < 64; l++)
        {
        output[(((((clusterId * dim2) * dim3) * dim4) + ((coreId * dim3) * dim4)) + (k * dim4)) + l] = output_Nram[l];
        }

    }
    }

    """
    code = ast_tensorization(code)
    print(code)
