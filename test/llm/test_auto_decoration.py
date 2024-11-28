from falcon.src.post_processing.post_processing import run_code_decoration

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
    code = run_code_decoration(code)
    print(code)

    code = """
    __mlu_global__ void add(float *A, float *B, float *T_add)
    {
        if (((clusterId * 4) + coreId) < 64)
        {
            for (int j = 0; j < 64; j++)
            {
                T_add[(k * 64) + j] = A[(k * 64) + j] + B[(k * 64) + j];
            }

        }
    }
    """
    code = run_code_decoration(code)
    print(code)
