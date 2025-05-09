__global__ void forward_kernel_v2(const float* Q, const float* K, const float* V, float* O, float* L) {
    int tid = threadIdx.x;
    int bid_x = blockIdx.x;
    int bid_y = blockIdx.y;
    if (tid < 32 && bid_x < 1 && bid_y < 32) {
    int qkv_offset = (bid_x * gridDim.y * 32 * 1024) + (bid_y * 32 * 1024);

    __shared__ float sram[528384];

    int tile_size = 32 * 1024;
    float scale = 1.0/sqrt(1024);
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[2 * tile_size];
    float* S = &sram[3 * tile_size];

    for (int i = 0; i < 1; i++) {
        for (int x = 0; x < 1024; x++) {
            Qi[(tid * 1024) + x] = Q[qkv_offset + (tile_size * i) + (tid * 1024) + x];
        }

        __syncthreads();

        float row_m_prev = -INFINITY;
        float row_l_prev = 0.0f;
        float row_m_new ;
        float row_l_new;

        for (int j = 0; j < 1; j++) {
            for (int x = 0; x < 1024; x++) {
                Kj[(tid * 1024) + x] = K[qkv_offset + (tile_size * j) + (tid * 1024) + x];
                Vj[(tid * 1024) + x] = V[qkv_offset + (tile_size * j) + (tid * 1024) + x];
            }

            float row_m = -INFINITY;

            for (int y = 0; y < 32; y++) {
                float sum = 0;
                for (int x = 0; x < 1024; x++) {
                    sum += Qi[(tid * 1024) + x] * Kj[(y * 1024) + x];
                }
                sum *= scale;
                S[(32 * tid) + y] = sum;
                if (sum > row_m) {
                    row_m = sum;
                }
            }

            row_m_new = max(row_m_prev, row_m);

            float row_l = 0;
            for (int y = 0; y < 32; y++) {
                S[(32 * tid) + y] = __expf(S[(32 * tid) + y] - row_m_new);
                row_l += S[(32 * tid) + y];
            }

            row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + row_l;

            for (int x = 0; x < 1024; x++) {
                float pv = 0;
                for (int y = 0; y < 32; y++) {
                    pv += S[(32 * tid) + y] * Vj[(y * 1024) + x];
                }
                O[qkv_offset + (tile_size * i) + (tid * 1024) + x] = (__expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tid * 1024) + x] + pv);
            }

            row_m_prev = row_m_new;
            row_l_prev = row_l_new;
        }

        __syncthreads();

        for (int x = 0; x < 1024; x++) {
            O[qkv_offset + (tile_size * i) + (tid * 1024) + x] *= 1 / row_l_prev;
        }
        L[qkv_offset + (tile_size * i) + (tid * 1024)] = row_m_new + __logf(row_l_new);
    }
    }
}
