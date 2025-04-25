from falcon.src.post_processing.post_processing import run_tensorization

if __name__ == "__main__":
    code = """
    #pragma operation(memory(input[B], output[B_wram]))
    for (int col = 0; col < 64; col++) {
        for (int i = 0; i < 512; i++) {
            B_wram[i * 64 + col] = B[i * 64 + col];

    #pragma operation(memory(input[A], output[A_nram]))
    for (int i = 0; i < 512; i++) {
        A_nram[i] = A[(clusterId * 4 + coreId) * 512 + i];
    }


    #pragma operation(matmul(input[A_nram, B_wram], output[C_nram]))
    for (int col = 0; col < 64; col++) {
        C_nram[(clusterId * 4 + coreId) * 64 + col] = 0.0f;
        for (int i = 0; i < 512; i++) {
            C_nram[col] += A_nram[i] * B_wram[i * 64 + col];
        }
    }

    #pragma operation(memory(input[C_nram], output[C]))
    for (int col = 0; col < 64; col++) {
        C[(clusterId * 4 + coreId) * 64 + col] = C_nram[col];
    }
    """
    code = run_tensorization(code, target="mlu")
    print(code)

    code = """
    __mlu_global__ void add(float *A, float *B, float *T_add)
    {
    __nram__ float A_Nram[64];
    __nram__ float B_Nram[64];
    __nram__ float T_add_Nram[64];
    if (((clusterId * 4) + coreId) < 64)
    {
        #pragma operation(memory(input[A], output[A_Nram]))
        for (int j = 0; j < 64; j++)
        {
        A_Nram[j] = A[(((clusterId * 4) + coreId) * 64) + j];
        }

        #pragma operation(memory(input[B], output[B_Nram]))
        for (int j = 0; j < 64; j++)
        {
        B_Nram[j] = B[(((clusterId * 4) + coreId) * 64) + j];
        }

        #pragma operation(add(input[A_Nram, B_Nram], output[T_add_Nram]))
        for (int j = 0; j < 64; j++)
        {
        T_add_Nram[j] = A_Nram[j] + B_Nram[j];
        }

        #pragma operation(memory(input[T_add_Nram], output[T_add]))
        for (int j = 0; j < 64; j++)
        {
        T_add[(((clusterId * 4) + coreId) * 64) + j] = T_add_Nram[j];
        }

    }
    }
    """
    code = run_tensorization(code, target="mlu")
    print(code)
