
extern "C" void conv1d(float *input, float *kernel, float *output) {
  for (int i = 0; i < 126; i++) {
    output[i] = 0;
    for (int j = 0; j < 3; j++) {
      output[i] += input[i + j] * kernel[j];
    }
  }
}
