extern "C" void add_kernel(float *output, float *input1, float *input2) {
  int rows = 21;
  int cols = 192;

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int index = i * cols + j;
      output[index] = input1[index] + input2[index];
    }
  }
}
