from falcon.smt.tensorization.tensorization import ast_tensorization

if __name__ == "__main__":
    code = """
    void add_kernel(float *input1, float *input2, float *output)
    {
        __nram__ float input1_Nram[64];
        __nram__ float input2_Nram[64];
        __nram__ float output_Nram[64];
        for (int k = 0; k < 4; k++)
        {
            #pragma operation(memory(input[input1], output[input1_Nram]))
            for (int l = 0; l < 64; l++)
            {
                input1_Nram[l] = input1[(((((clusterId * 4) * 4) * 64) + ((coreId * 4) * 64)) + (k * 64)) + l];
            }

            #pragma operation(memory(input[input2], output[input2_Nram]))
            for (int l = 0; l < 64; l++)
            {
                input2_Nram[l] = input2[(((((clusterId * 4) * 4) * 64) + ((coreId * 4) * 64)) + (k * 64)) + l];
            }

            #pragma operation(add(input[input1_Nram, input2_Nram], output[output_Nram]))
            for (int l = 0; l < 64; l++)
            {
                output_Nram[l] = input1_Nram[l] + input2_Nram[l];
            }

            #pragma operation(memory(input[output_Nram], output[output]))
            for (int l = 0; l < 64; l++)
            {
                output[(((((clusterId * 4) * 4) * 64) + ((coreId * 4) * 64)) + (k * 64)) + l] = output_Nram[l];
            }

        }
    }
    """
    code = ast_tensorization(code, target="BANG")
    print(code)
