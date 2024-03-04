void lstm_kernel(float* X, 
                 float* Wi2h, 
                 float* Wh2h, 
                 float* lstm_scan, 
                 float* lstm_scan_1) {

  for (int32_t i1 = 0; i1 < 4; ++i1) {
    for (int32_t i2 = 0; i2 < 1152; ++i2) {
      ((float*)lstm_scan)[((i1 * 1152) + i2)] = 0.000000e+00f;
    }
  }
  for (int32_t i1_1 = 0; i1_1 < 4; ++i1_1) {
    for (int32_t i2_1 = 0; i2_1 < 1152; ++i2_1) {
      ((float*)lstm_scan_1)[((i1_1 * 1152) + i2_1)] = 0.000000e+00f;
    }
  }
  for (int32_t lstm_scan_idx = 0; lstm_scan_idx < 127; ++lstm_scan_idx) {
    for (int32_t x = 0; x < 4; ++x) {
      for (int32_t i = 0; i < 4; ++i) {
        for (int32_t j = 0; j < 1152; ++j) {
          ((float*)s_i2h)[(((x * 4608) + (i * 1152)) + j)] = 0.000000e+00f;
          for (int32_t ki2h = 0; ki2h < 256; ++ki2h) {
            int32_t cse_var_1 = (((x * 4608) + (i * 1152)) + j);
            ((float*)s_i2h)[cse_var_1] = (((float*)s_i2h)[cse_var_1] + (((float*)X)[(((lstm_scan_idx * 1024) + (i * 256)) + ki2h)] * ((float*)Wi2h)[(((x * 294912) + (j * 256)) + ki2h)]));
          }
        }
      }
    }
    for (int32_t x_1 = 0; x_1 < 4; ++x_1) {
      for (int32_t i_1 = 0; i_1 < 4; ++i_1) {
        for (int32_t j_1 = 0; j_1 < 1152; ++j_1) {
          ((float*)s_h2h)[(((x_1 * 4608) + (i_1 * 1152)) + j_1)] = 0.000000e+00f;
          for (int32_t ki2h_1 = 0; ki2h_1 < 1152; ++ki2h_1) {
            int32_t cse_var_3 = (i_1 * 1152);
            int32_t cse_var_2 = (((x_1 * 4608) + cse_var_3) + j_1);
            ((float*)s_h2h)[cse_var_2] = (((float*)s_h2h)[cse_var_2] + (((float*)lstm_scan)[(((lstm_scan_idx * 4608) + cse_var_3) + ki2h_1)] * ((float*)Wh2h)[(((x_1 * 1327104) + (j_1 * 1152)) + ki2h_1)]));
          }
        }
      }
    }
    for (int32_t i_2 = 0; i_2 < 4; ++i_2) {
      for (int32_t j_2 = 0; j_2 < 1152; ++j_2) {
        int32_t cse_var_7 = (i_2 * 1152);
        int32_t cse_var_6 = (cse_var_7 + j_2);
        int32_t cse_var_5 = (cse_var_6 + 9216);
        int32_t cse_var_4 = (cse_var_6 + 4608);
        ((float*)next_c)[cse_var_6] = (((1.000000e+00f / (1.000000e+00f + expf((0.000000e+00f - (((float*)s_i2h)[cse_var_5] + ((float*)s_h2h)[cse_var_5]))))) * ((float*)lstm_scan_1)[(((lstm_scan_idx * 4608) + cse_var_7) + j_2)]) + ((1.000000e+00f / (1.000000e+00f + expf((0.000000e+00f - (((float*)s_i2h)[cse_var_6] + ((float*)s_h2h)[cse_var_6]))))) * tanhf((((float*)s_i2h)[cse_var_4] + ((float*)s_h2h)[cse_var_4]))));
      }
    }
    for (int32_t i_3 = 0; i_3 < 4; ++i_3) {
      for (int32_t j_3 = 0; j_3 < 1152; ++j_3) {
        int32_t cse_var_9 = ((i_3 * 1152) + j_3);
        int32_t cse_var_8 = (cse_var_9 + 13824);
        ((float*)next_h)[cse_var_9] = ((1.000000e+00f / (1.000000e+00f + expf((0.000000e+00f - (((float*)s_i2h)[cse_var_8] + ((float*)s_h2h)[cse_var_8]))))) * tanhf(((float*)next_c)[cse_var_9]));
      }
    }
    for (int32_t i1_2 = 0; i1_2 < 4; ++i1_2) {
      for (int32_t i2_2 = 0; i2_2 < 1152; ++i2_2) {
        int32_t cse_var_10 = (i1_2 * 1152);
        ((float*)lstm_scan)[((((lstm_scan_idx * 4608) + cse_var_10) + i2_2) + 4608)] = ((float*)next_h)[(cse_var_10 + i2_2)];
      }
    }
    for (int32_t i1_3 = 0; i1_3 < 4; ++i1_3) {
      for (int32_t i2_3 = 0; i2_3 < 1152; ++i2_3) {
        int32_t cse_var_11 = (i1_3 * 1152);
        ((float*)lstm_scan_1)[((((lstm_scan_idx * 4608) + cse_var_11) + i2_3) + 4608)] = ((float*)next_c)[(cse_var_11 + i2_3)];
      }
    }
  }
}
