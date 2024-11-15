extern "C" void
multiHeadAttentionForward_kernel(float *Q,     //[batch, seq_len, heads, dim]
                                 float *K,     //[batch, seq_len, heads, dim]
                                 float *V,     //[batch, seq_len, heads, dim]
                                 float *output //[batch, seq_len, heads, dim]
) {
  int8_t arr_a[64];
  int8_t arr_b[64];
  int32_t arr_d[16]; // AVX-512 寄存器能同时处理 16 个 int32 元素
  float score[6 * 6];

  int8_t arr_a[16];
  int8_t arr_b[16];
  int32_t arr_d[4];

  const int batch = 1;
  const int seq_len = 4096;
  const int heads = 16;
  const int dim = 256;
  const float scale = 1.0f / sqrt(dim);

  for (int i = 0; i < batch; i++) {
    for (int j = 0; j < seq_len; j++) {
      for (int m = 0; m < heads; m++) {
        for (int n = 0; n < heads; n++) {
          int32_t sum = 0;

          for (int local_s = 0; local_s < dim / 64; local_s++) { // 每次处理 64 个元素
            for (int local_i = 0; local_i < 64; local_i++) {
              arr_a[local_i] = static_cast<int8_t>(
                  Q[i * seq_len * heads * dim + j * heads * dim + m * dim +
                    local_s * 64 + local_i] *
                  scale);
              arr_b[local_i] = static_cast<int8_t>(
                  K[i * seq_len * heads * dim + j * heads * dim + n * dim +
                    local_s * 64 + local_i] *
                  scale);
            }

            __m512i acc = _mm512_setzero_si512();

            __m512i _a = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(arr_a));
            __m512i _b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(arr_b));

            acc = _mm512_dpbusd_epi32(acc, _a, _b);

            _mm512_storeu_si512(reinterpret_cast<__m512i *>(arr_d), acc);

            for (int k = 0; k < 16; ++k) { // 处理 16 个累积结果
              sum += arr_d[k];
            }
          }

          score[m * heads + n] = static_cast<float>(sum) * scale;
        }
      }

      // Softmax
      for (int m = 0; m < heads; ++m) {
        float max_val = -INFINITY;
        for (int n = 0; n < heads; ++n) {
          max_val = std::max(max_val, score[m * heads + n]);
        }
        float sum_exp = 0.0f;
        for (int n = 0; n < heads; ++n) {
          score[m * heads + n] = expf(score[m * heads + n] - max_val);
          sum_exp += score[m * heads + n];
        }
        for (int n = 0; n < heads; ++n) {
          score[m * heads + n] /= sum_exp;
        }
      }

      for (int j_dl = 0; j_dl < 16; j_dl++) {
        for (int k_dl = 0; k_dl < 256; k_dl++) {
          int32_t sum = 0;
          // 将浮点数组A和B量化到int8类型
          for (int local_i = 0; local_i < 16; ++local_i) {
            arr_a[local_i] = static_cast<int8_t>(score[j_dl * 16 + local_i]);
            arr_b[local_i] = static_cast<int8_t>(V[i * seq_len * heads * dim + j * heads * dim + local_i * 256 + k_dl]);
          }

          // 使用VNNI指令进行乘加操作
          __m128i acc = _mm_setzero_si128(); // 初始化累加器为0

          // 加载量化后的数据到SIMD寄存器中
          __m128i _a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(arr_a));
          __m128i _b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(arr_b));

          // 使用_mm_dpbusds_epi32进行乘加操作 (VNNI)
          acc = _mm_dpbusds_epi32(acc, _a, _b); // 执行乘加操作：acc += a * b

          // 将累加结果存储到arr_d中
          _mm_storeu_si128(reinterpret_cast<__m128i *>(arr_d), acc);

          // 将arr_d中的值累加得到最终的结果
          for (int i_dl = 0; i_dl < 4; ++i_dl) {
            sum += arr_d[i_dl];
          }

          // 反量化并存储到输出矩阵result中
          result[i * seq_len * heads * dim + j * heads * dim + j_dl * 256 + k_dl] = static_cast<float>(sum);
        }
      }
    }
  }
}
