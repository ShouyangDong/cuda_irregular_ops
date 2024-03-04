void transpose_kernel(float*  ins, float*  outs) {
    for (int i = 0; i < 8192; ++i) {
        for (int j = 0; j < 8192; ++j) {
            outs[j * 8192 + i] = ins[i * 8192 + j];
        }
    }
}
