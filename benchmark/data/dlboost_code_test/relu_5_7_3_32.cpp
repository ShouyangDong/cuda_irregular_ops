
float reluf(float input) { return input > 0 ? input : 0; }
extern "C" void relu_kernel(float *input, float *output) {
  for (size_t i = 0; i < 5; i++) {
    for (size_t j = 0; j < 7; j++) {
      for (size_t k = 0; k < 3; k++) {
        for (size_t l = 0; l < 32; l++) {
          output[i * 7 * 3 * 32 + j * 3 * 32 + k * 32 + l] =
              reluf(input[i * 7 * 3 * 32 + j * 3 * 32 + k * 32 + l]);
        }
      }
    }
  }
}