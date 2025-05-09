__global__ void flashattenv1(const float* Q,const float* K,const float* V, float* l,float* m,float* O){
    int tid = threadIdx.x;

    int bid_x = blockIdx.x;
    int bid_y = blockIdx.y;
    if (tid < 64 && bid_x < 1 && bid_y < 16) {
        

    int qkv_offset = (bid_x * gridDim.y * 64 * 1024) + (bid_y * 64 * 1024);
    int lm_offset = (bid_x * gridDim.y * 64) + (bid_y * 64);
    __shared__ float sram[528384];

    int tile_size = 64*1024;
    float scale  = 1.0/sqrt(1024);
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size*2];
    float* S = &sram[tile_size*3];

    for(int j=0;j<2;j++){
        for(int x=0;x<1024;x++){
            Kj[(tid*1024)+x] = K[qkv_offset + (tile_size*j) + (tid*1024)+x];
            Vj[(tid*1024)+x] = V[qkv_offset + (tile_size*j) + (tid*1024)+x];
        }

        __syncthreads();

        for(int i=0;i<2;i++){
            for(int x=0;x<1024;x++){
            Qi[(tid*1024)+x] = Q[qkv_offset + (tile_size*i) + (tid*1024)+x];
        }
        float row_m_prev = m[lm_offset + (64*i) + tid];
        float row_l_prev = l[lm_offset + (64*i) + tid];

        float row_m = -INFINITY;
        for(int y=0;y<64;y++){
            float sum = 0;
            for(int x = 0;x<d;x++){
                sum+=  Qi[(tid*1024)+x] * Kj[(y*1024)+x];
            }
            sum*=scale;
            S[(64*tid) + y] = sum;
            if(sum>row_m){
                row_m = sum;
                }
            }
        float row_l = 0;
        for(int y =0;y<64;y++){
            S[(64 * tid) + y] = __expf(S[(64 * tid) + y] - row_m);
            row_l +=  S[(64 * tid) + y];
            }
        
        float row_m_new = max(row_m_prev,row_m);
        float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

        for(int x = 0; x < 1024;x++){
                float pv = 0;
                for(int y=0;y<64;y++){
                    pv += S[(64 * tid) + y] * Vj[(y*1024)+x];
                }
                O[qkv_offset + (tile_size*i) + (tid*1024)+x] = (1/(row_l_new)) * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size*i) + (tid*1024)+x]) + (__expf(row_m - row_m_new)*pv));
            }
            m[lm_offset + (64*i) + tid] = row_m_new;
            l[lm_offset + (64*i) + tid] = row_l_new;
        }

        __syncthreads();
    }
    }

}