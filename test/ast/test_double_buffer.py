from smt.software_pipeline import smt_double_buffer

if __name__ == "__main__":
    code = """void add(float* INPUT0, float* INPUT1, float* OUTPUT) {
        float INPUT0_N[64];
        float INPUT1_N[64];
        float OUTPUT_N[64];
        #pragma software_pipeline
        for (int i = 0; i < 2048; ++i) {
            __memcpy(INPUT0_N, INPUT0 + (i * 64), 256, GDRAM2NRAM);
            __memcpy(INPUT1_N, INPUT1 + (i * 64), 256, GDRAM2NRAM);
            __bang_add(OUTPUT_N, INPUT0_N , INPUT1_N, 64);
            __memcpy(OUTPUT + (i * 64), OUTPUT_N, 256, NRAM2GDRAM);
        }
    }
    """
    code = smt_double_buffer(code)
    print(code)

    code = """void tanh(float* INPUT0, float* INPUT1, float* OUTPUT) {
        float INPUT0_N[64];
        float OUTPUT_N[64];
        #pragma software_pipeline
        for (int i = 0; i < 2048; ++i) {
            __memcpy(INPUT0_N, INPUT0 + (i * 64), 256, GDRAM2NRAM);
            __bang_active_tanh(OUTPUT_N, INPUT0_N, 64);
            __memcpy(OUTPUT + (i * 64), OUTPUT_N, 256, NRAM2GDRAM);
        }
    }
    """
    code = smt_double_buffer(code)
    print(code)
