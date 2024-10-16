
extern "C" void multiHeadAttentionForward_kernel(
    float* Q,      //[batch, seq_len, heads, dim]
    float* K,      //[batch, seq_len, heads, dim]
    float* V,      //[batch, seq_len, heads, dim]
    float* output  //[batch, seq_len, heads, dim]
) {
  uint8_t arr_a[16];
  uint8_t arr_b[16];
  uint32_t arr_src[4];
  float score[6 * 6];
  // The dimension 1, 4096, 6, 256
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 4096; j++) {
      for (int m = 0; m < 6; m++) {
        for (int n = 0; n < 6; n++) {
          // score[m * 6 + n] = 0.0;
          uint32_t sum = 0;
          for (int local_s = 0; local_s < 16; local_s++) {
            for (int local_i = 0; local_i < 16; local_i++) {
                arr_a[local_i] = (uint8_t)Q[i * 4096 * 6 * 256 + j * 6 * 256 + m * 256 + (local_s * 16 + local_i)];
                arr_b[local_i] = (uint8_t)K[i * 4096 * 6 * 256 + j * 6 * 256 + n * 256 + local_s * 16 +local_i];
            }
            for (int i_src =0; i_src<4; i_src++){
                arr_src[i_src] = (uint32_t)0;
            }
            
            __m128i A = _mm_loadu_si128((__m128i*)&arr_a);
            __m128i B = _mm_loadu_si128((__m128i*)&arr_b);
            __m128i src = _mm_loadu_si128((__m128i*)&arr_src);
            __m128i local_result =  _mm_dpbusds_epi32(src, A, B);
            uint32_t *val = (uint32_t*) &local_result;
            for(int i = 0; i < 4; i++){
                sum += val[i];
            }
        }
        score[m * 6 + n] = float(sum); 
        }
      }

      // score
      for (int m_sc = 0; m_sc < 6; m_sc++) {
        for (int n_sc = 0; n_sc < 6; n_sc++) {
          score[m_sc * 6 + n_sc] = score[m_sc * 6 + n_sc] / sqrt(256);
        }
      }

      // The Softmax code:
      for (int j_sf = 0; j_sf < 6; ++j_sf) {
        float sum = 0;

        for (int i_ex = 0; i_ex < 6; ++i_ex) {
          score[j_sf * 6 + i_ex] = expf(score[j_sf * 6 + i_ex]);
        }
        for (int i_sf = 0; i_sf < 6; ++i_sf) {
          sum += score[j_sf * 6 + i_sf];
        }
        for (int k_sf = 0; k_sf < 6; ++k_sf) {
          score[j_sf * 6 + k_sf] = score[j_sf * 6 + k_sf] / sum;
        }
      }

      // The final Matmul
      for (int m_fl = 0; m_fl < 6; ++m_fl) {
        for (int n_fl = 0; n_fl < 256; ++n_fl) {
          output[i * 4096 * 6 * 256 + j * 6 * 256 + m_fl * 256 + n_fl] = 0.0;
          for (int k_fl = 0; k_fl < 6; ++k_fl) {
            output[i * 4096 * 6 * 256 + j * 6 * 256 + m_fl * 256 + n_fl] +=
                score[m_fl * 6 + k_fl] *
                V[i * 4096 * 6 * 256 + j * 6 * 256 + k_fl * 256 + n_fl];
          }
        }
      }
    }
  }
}