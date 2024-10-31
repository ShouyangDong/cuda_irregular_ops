
float signf(float input) {
  if (input > 0) {
    return 1;
  } else if (input < 0) {
    return -1;
  } else {
    return 0;
  }
}
extern "C" void sign_kernel(float *output, float *input) {
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t k = 0; k < 5; k++) {
        output[i * 4 * 5 + j * 5 + k] = signf(input[i * 4 * 5 + j * 5 + k]);
      }
    }
  }
}