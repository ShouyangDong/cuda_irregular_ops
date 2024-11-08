
float sigmoidf(float input) { return 1 / (1 + exp(-1 * input)); }
extern "C" void sigmoid_kernel(float *input, float *output) {
  for (size_t i = 0; i < 7; i++) {
    for (size_t j = 0; j < 1; j++) {
      for (size_t k = 0; k < 6; k++) {
        for (size_t l = 0; l < 7; l++) {
          output[i * 1 * 6 * 7 + j * 6 * 7 + k * 7 + l] =
              sigmoidf(input[i * 1 * 6 * 7 + j * 6 * 7 + k * 7 + l]);
        }
      }
    }
  }
}