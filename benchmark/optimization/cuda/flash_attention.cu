// shape [64, 2048, 12, 256]

__global__
void forward_kernel(float* Q, float* K, float* V,
                    float* l, float *m, float* O) {
    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (blockIdx.x * gridDim.y * 12 * 256) + (blockIdx.y * 12 * 256);  // gridDim.y = nh
    int lm_offset = (blockIdx.x * gridDim.y * 12) + (blockIdx.y * 12);  // offset for l and m

    // Define SRAM for Q,K,V,S
    __shared__ float sram[102400];
    int tile_size = 32 * 256;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    for (int j = 0; j < 1; j++) {
        // Load Kj, Vj to SRAM
        for (int x = 0; x < 256; x++) {
            Kj[(threadIdx.x * 256) + x] = K[qkv_offset + (tile_size * j) + (threadIdx.x * 256) + x];
            Vj[(threadIdx.x * 256) + x] = V[qkv_offset + (tile_size * j) + (threadIdx.x * 256) + x];
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < 1; i++)  {

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < 256; x++) {
                Qi[(threadIdx.x * 256) + x] = Q[qkv_offset + (tile_size * i) + (threadIdx.x * 256) + x];
            }
            float row_m_prev = m[lm_offset + (32 * i) + threadIdx.x];
            float row_l_prev = l[lm_offset + (32 * i) + threadIdx.x];

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < 32; y++) {
                float sum = 0;
                for (int x = 0; x < 256; x++) {
                    sum += Qi[(threadIdx.x * 256) + x] * Kj[(y * 256) + x];
                }
                sum *= 1.0 / sqrt(256);
                S[(32 * threadIdx.x) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < 32; y++) {
                S[(32 * threadIdx.x) + y] = __expf(S[(32 * threadIdx.x) + y] - row_m);
                row_l += S[(32 * threadIdx.x) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < 256; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < 32; y++) {
                    pv += S[(32 * threadIdx.x) + y] * Vj[(y * 256) + x];
                }
                O[qkv_offset + (tile_size * i) + (threadIdx.x * 256) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (threadIdx.x * 256) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (32 * i) + threadIdx.x] = row_m_new;
            l[lm_offset + (32 * i) + threadIdx.x] = row_l_new;
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}
