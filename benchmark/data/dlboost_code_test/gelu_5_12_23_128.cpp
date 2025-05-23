
float geluf(float x) {
  return 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3))));
}
extern "C" void gelu(float *input, float *output) {
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 12; j++) {
      for (int k = 0; k < 23; k++) {
        for (int l = 0; l < 128; l++) {
          int index = i * 12 * 23 * 128 + j * 23 * 128 + k * 128 + l;
          output[index] = geluf(input[index]);
        }
      }
    }
  }
}