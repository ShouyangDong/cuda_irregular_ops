
extern "C" void deformable(float *value_, int *value_spatial_shapes_,
                                  float *sampling_locations_,
                                  float *attention_weights_, float *output_) {
  float attention_sum[8192];
  float value_offset[4];
  float height_width[8];
  float xy[8];
  float xy_grid[64];
  int xy_rounded[128];
  float corner_values[32768];
  for (int32_t j = 0; j < 200; ++j) {
    for (int32_t i_d = 0; i_d < 256; ++i_d) {
      for (int32_t i_m = 0; i_m < 8; ++i_m) {
        ((float *)attention_sum)[((i_m * 256) + i_d)] = 0.000000e+00f;
      }
    }
    for (int32_t i = 0; i < 4; ++i) {
      ((int32_t *)value_offset)[0] = 0;
      for (int32_t ii = 0; ii < i; ++ii) {
        int32_t cse_var_1 = (ii * 2);
        ((int32_t *)value_offset)[0] =
            (((int32_t *)value_offset)[0] +
             (((int32_t *)value_spatial_shapes_)[cse_var_1] *
              ((int32_t *)value_spatial_shapes_)[(cse_var_1 + 1)]));
      }
      for (int32_t k = 0; k < 4; ++k) {
        for (int32_t i_m_1 = 0; i_m_1 < 8; ++i_m_1) {
          int32_t cse_var_5 = (i_m_1 + 8);
          int32_t cse_var_4 = (i_m_1 + 16);
          int32_t cse_var_3 = (i * 2);
          int32_t cse_var_2 =
              ((((j * 256) + (i_m_1 * 32)) + (i * 8)) + (k * 2));
          ((int32_t *)height_width)[0] =
              ((int32_t *)value_spatial_shapes_)[cse_var_3];
          ((int32_t *)height_width)[1] =
              ((int32_t *)value_spatial_shapes_)[(cse_var_3 + 1)];
          ((float *)xy)[1] = ((float *)sampling_locations_)[cse_var_2];
          ((float *)xy)[0] = ((float *)sampling_locations_)[(cse_var_2 + 1)];
          ((float *)xy_grid)[i_m_1] =
              ((((float *)xy)[0] * ((float)((int32_t *)height_width)[0])) -
               5.000000e-01f);
          ((float *)xy_grid)[cse_var_5] =
              ((((float *)xy)[1] * ((float)((int32_t *)height_width)[1])) -
               5.000000e-01f);
          ((int32_t *)xy_rounded)[i_m_1] =
              ((int32_t)floorf(((float *)xy_grid)[i_m_1]));
          ((int32_t *)xy_rounded)[cse_var_5] =
              (((int32_t *)xy_rounded)[i_m_1] + 1);
          ((int32_t *)xy_rounded)[cse_var_4] =
              ((int32_t)floorf(((float *)xy_grid)[cse_var_5]));
          ((int32_t *)xy_rounded)[(i_m_1 + 24)] =
              (((int32_t *)xy_rounded)[cse_var_4] + 1);
        }
        for (int32_t i_m_2 = 0; i_m_2 < 8; ++i_m_2) {
          if ((((((int32_t *)xy_rounded)[i_m_2] < 0) ||
                (((int32_t *)height_width)[0] <=
                 ((int32_t *)xy_rounded)[i_m_2])) ||
               (((int32_t *)xy_rounded)[(i_m_2 + 16)] < 0)) ||
              (((int32_t *)height_width)[1] <=
               ((int32_t *)xy_rounded)[(i_m_2 + 16)])) {
            for (int32_t i_d_1 = 0; i_d_1 < 256; ++i_d_1) {
              ((float *)corner_values)[((i_m_2 * 256) + i_d_1)] = 0.000000e+00f;
            }
          } else {
            for (int32_t i_d_2 = 0; i_d_2 < 256; ++i_d_2) {
              int32_t cse_var_7 = (i_m_2 * 256);
              ((float *)corner_values)[(cse_var_7 + i_d_2)] =
                  ((float *)value_)[(
                      ((((((int32_t *)value_offset)[0] * 2048) +
                         ((((int32_t *)xy_rounded)[i_m_2] *
                           ((int32_t *)height_width)[1]) *
                          2048)) +
                        (((int32_t *)xy_rounded)[(i_m_2 + 16)] * 2048)) +
                       cse_var_7) +
                      i_d_2)];
            }
          }
        }
        for (int32_t i_m_3 = 0; i_m_3 < 8; ++i_m_3) {
          if ((((((int32_t *)xy_rounded)[i_m_3] < 0) ||
                (((int32_t *)height_width)[0] <=
                 ((int32_t *)xy_rounded)[i_m_3])) ||
               (((int32_t *)xy_rounded)[(i_m_3 + 24)] < 0)) ||
              (((int32_t *)height_width)[1] <=
               ((int32_t *)xy_rounded)[(i_m_3 + 24)])) {
            for (int32_t i_d_3 = 0; i_d_3 < 256; ++i_d_3) {
              ((float *)corner_values)[(((i_m_3 * 256) + i_d_3) + 2048)] =
                  0.000000e+00f;
            }
          } else {
            for (int32_t i_d_4 = 0; i_d_4 < 256; ++i_d_4) {
              int32_t cse_var_9 = (i_m_3 * 256);
              ((float *)corner_values)[((cse_var_9 + i_d_4) + 2048)] =
                  ((float *)value_)[(
                      ((((((int32_t *)value_offset)[0] * 2048) +
                         ((((int32_t *)xy_rounded)[i_m_3] *
                           ((int32_t *)height_width)[1]) *
                          2048)) +
                        (((int32_t *)xy_rounded)[(i_m_3 + 24)] * 2048)) +
                       cse_var_9) +
                      i_d_4)];
            }
          }
        }
        for (int32_t i_m_4 = 0; i_m_4 < 8; ++i_m_4) {
          if ((((((int32_t *)xy_rounded)[(i_m_4 + 8)] < 0) ||
                (((int32_t *)height_width)[0] <=
                 ((int32_t *)xy_rounded)[(i_m_4 + 8)])) ||
               (((int32_t *)xy_rounded)[(i_m_4 + 16)] < 0)) ||
              (((int32_t *)height_width)[1] <=
               ((int32_t *)xy_rounded)[(i_m_4 + 16)])) {
            for (int32_t i_d_5 = 0; i_d_5 < 256; ++i_d_5) {
              ((float *)corner_values)[(((i_m_4 * 256) + i_d_5) + 4096)] =
                  0.000000e+00f;
            }
          } else {
            for (int32_t i_d_6 = 0; i_d_6 < 256; ++i_d_6) {
              int32_t cse_var_12 = (i_m_4 * 256);
              ((float *)corner_values)[((cse_var_12 + i_d_6) + 4096)] =
                  ((float *)value_)[(
                      ((((((int32_t *)value_offset)[0] * 2048) +
                         ((((int32_t *)xy_rounded)[(i_m_4 + 8)] *
                           ((int32_t *)height_width)[1]) *
                          2048)) +
                        (((int32_t *)xy_rounded)[(i_m_4 + 16)] * 2048)) +
                       cse_var_12) +
                      i_d_6)];
            }
          }
        }
        for (int32_t i_m_5 = 0; i_m_5 < 8; ++i_m_5) {
          if ((((((int32_t *)xy_rounded)[(i_m_5 + 8)] < 0) ||
                (((int32_t *)height_width)[0] <=
                 ((int32_t *)xy_rounded)[(i_m_5 + 8)])) ||
               (((int32_t *)xy_rounded)[(i_m_5 + 24)] < 0)) ||
              (((int32_t *)height_width)[1] <=
               ((int32_t *)xy_rounded)[(i_m_5 + 24)])) {
            for (int32_t i_d_7 = 0; i_d_7 < 256; ++i_d_7) {
              ((float *)corner_values)[(((i_m_5 * 256) + i_d_7) + 6144)] =
                  0.000000e+00f;
            }
          } else {
            for (int32_t i_d_8 = 0; i_d_8 < 256; ++i_d_8) {
              int32_t cse_var_15 = (i_m_5 * 256);
              ((float *)corner_values)[((cse_var_15 + i_d_8) + 6144)] =
                  ((float *)value_)[(
                      ((((((int32_t *)value_offset)[0] * 2048) +
                         ((((int32_t *)xy_rounded)[(i_m_5 + 8)] *
                           ((int32_t *)height_width)[1]) *
                          2048)) +
                        (((int32_t *)xy_rounded)[(i_m_5 + 24)] * 2048)) +
                       cse_var_15) +
                      i_d_8)];
            }
          }
        }
        for (int32_t i_m_6 = 0; i_m_6 < 8; ++i_m_6) {
          for (int32_t i_d_9 = 0; i_d_9 < 256; ++i_d_9) {
            int32_t cse_var_19 = (i_m_6 + 8);
            int32_t cse_var_18 = (i_m_6 + 24);
            int32_t cse_var_17 = (i_m_6 + 16);
            int32_t cse_var_16 = ((i_m_6 * 256) + i_d_9);
            ((float *)attention_sum)[cse_var_16] =
                (((float *)attention_sum)[cse_var_16] +
                 ((((((((float *)corner_values)[cse_var_16] *
                       (((float)((int32_t *)xy_rounded)[cse_var_19]) -
                        ((float *)xy_grid)[i_m_6])) *
                      (((float)((int32_t *)xy_rounded)[cse_var_18]) -
                       ((float *)xy_grid)[cse_var_19])) +
                     ((((float *)corner_values)[(cse_var_16 + 4096)] *
                       (((float *)xy_grid)[i_m_6] -
                        ((float)((int32_t *)xy_rounded)[i_m_6]))) *
                      (((float)((int32_t *)xy_rounded)[cse_var_18]) -
                       ((float *)xy_grid)[cse_var_19]))) +
                    ((((float *)corner_values)[(cse_var_16 + 2048)] *
                      (((float)((int32_t *)xy_rounded)[cse_var_19]) -
                       ((float *)xy_grid)[i_m_6])) *
                     (((float *)xy_grid)[cse_var_19] -
                      ((float)((int32_t *)xy_rounded)[cse_var_17])))) +
                   ((((float *)corner_values)[(cse_var_16 + 6144)] *
                     (((float *)xy_grid)[i_m_6] -
                      ((float)((int32_t *)xy_rounded)[i_m_6]))) *
                    (((float *)xy_grid)[cse_var_19] -
                     ((float)((int32_t *)xy_rounded)[cse_var_17])))) *
                  ((float *)attention_weights_)[(
                      (((j * 128) + (i_m_6 * 16)) + (i * 4)) + k)]));
          }
        }
        for (int32_t i_m_7 = 0; i_m_7 < 8; ++i_m_7) {
          for (int32_t i_d_10 = 0; i_d_10 < 256; ++i_d_10) {
            int32_t cse_var_20 = (i_m_7 * 256);
            ((float *)output_)[(((j * 2048) + cse_var_20) + i_d_10)] =
                ((float *)attention_sum)[(cse_var_20 + i_d_10)];
          }
        }
      }
    }
  }
}