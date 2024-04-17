
extern "C" void multiHeadAttentionForward_kernel(
    float* Q,      //[batch, seq_len, heads, dim]
    float* K,      //[batch, seq_len, heads, dim]
    float* V,      //[batch, seq_len, heads, dim]
    float* output  //[batch, seq_len, heads, dim]
) {
  uint8_t arr_a[16];
  uint8_t arr_b[16];
  uint32_t arr_src[4];
  float score[6 *16];
  // The dimension 1, 4096,16, 512
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 4096; j++) {
      for (int m = 0; m <16; m++) {
        for (int n = 0; n <16; n++) {
          // score[m *16 + n] = 0.0;
          uint32_t sum = 0;
          for (int local_s = 0; local_s < 32; local_s++) {
            for (int local_i = 0; local_i < 16; local_i++) {
                arr_a[local_i] = (uint8_t)Q[i * 4096 *16 * 512 + j * 16 * 512 + m * 512 + (local_s * 16 + local_i)];
                arr_b[local_i] = (uint8_t)K[i * 4096 *16 * 512 + j *16 * 512 + n * 512 + local_s * 16 +local_i];
            }
            for (int i_src =0; i_src<NUM_32B_INT_IN_M6; i_src++){
                arr_src[i_src] = (uint32_t)0;
            }
            
            __m128i A = _mm_loadu_si128((__m128i*)&arr_a);
            __m128i B = _mm_loadu_si128((__m128i*)&arr_b);
            __m128i src = _mm_loadu_si128((__m128i*)&arr_src);
            __m128i local_result =  _mm_dpbusds_epi32(src, A, B);
            uint32_t *val = (uint32_t*) &local_result;
            for(int i = 0; i < NUM_32B_INT_IN_M128; i++){
                sum += val[1];
            }
        }
        score[m *16 + n] = float(sum); 
        }
      }

      // score
      for (int m_sc = 0; m_sc <16; m_sc++) {
        for (int n_sc = 0; n_sc <16; n_sc++) {
          score[m_sc *16 + n_sc] = score[m_sc *16 + n_sc] / sqrt(512);
        }
      }

      // The Softmax code:
      for (int j_sf = 0; j_sf <16; ++j_sf) {
        float sum = 0;

        for (int i_ex = 0; i_ex <16; ++i_ex) {
          score[j_sf *16 + i_ex] = expf(score[j_sf *16 + i_ex]);
        }
        for (int i_sf = 0; i_sf <16; ++i_sf) {
          sum += score[j_sf *16 + i_sf];
        }
        for (int k_sf = 0; k_sf <16; ++k_sf) {
          score[j_sf *16 + k_sf] = score[j_sf *16 + k_sf] / sum;
        }
      }

      // The final Matmul
      for (int m_fl = 0; m_fl <16; ++m_fl) {
        for (int n_fl = 0; n_fl < 512; ++n_fl) {
            uint32_t sum = 0;
            for (int local_i = 0; local_i < 16; local_i++) {
                arr_a[local_i] = (uint8_t)score[i * 4096 *16 * 512 + j *16 * 512 + m * 512 + local_i];
                arr_b[local_i] = (uint8_t)V[i * 4096 *16 * 512 + j *16 * 512 + k_fl * 512 + local_i];
            }
            for (int i_src =0; i_src<NUM_32B_INT_IN_M6; i_src++){
                arr_src[i_src] = (uint32_t)0;
            }
            
            __m128i A = _mm_loadu_si128((__m128i*)&arr_a);
            __m128i B = _mm_loadu_si128((__m128i*)&arr_b);
            __m128i src = _mm_loadu_si128((__m128i*)&arr_src);
            __m128i local_result =  _mm_dpbusds_epi32(src, A, B);
            uint32_t *val = (uint32_t*) &local_result;
            for(int i = 0; i < NUM_32B_INT_IN_M128; i++){
                sum += val[1];
            }
          }
          output[i * 4096 *16 * 512 + j *16 * 512 + m_fl * 512 + n_fl] = float(sum); 
        }
      }
    }
  }
}