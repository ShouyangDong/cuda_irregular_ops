
float sigmoidf(float input) { return 1 / (1 + exp(-1 * input)); }
extern "C" void sigmoid_kernel(float *input, float *output) {
  for (size_t i = 0; i < 12; i++) {
    for (size_t j = 0; j < 3; j++) {
      for (size_t k = 0; k < 128; k++) {
        output[i * 3 * 128 + j * 128 + k] =
            sigmoidf(input[i * 3 * 128 + j * 128 + k]);
      }
    }
  }
}