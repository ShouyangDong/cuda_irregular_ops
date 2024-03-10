#include <cmath>

extern "C" void layer_norm_kernel(
    float* input,
    float* output,
    float epsilon) 
{
    float mean = 0.0;
    float variance = 0.0;
    // Calculate mean
    for (int i_mean = 0; i_mean < size; i_mean++) {
        mean += input[i];
    }
    mean /= size;

    // Calculate variance
    for (int i_var = 0; i_var < size; i++) {
        variance += pow(input[i_var] - mean, 2);
    }
    variance /= size;

    // Normalize input
    // for (int i = 0; i < input.size(); i++) {
    for (int i_norm = 0; i_norm < size; i_norm++) {
        input[i] = (input[i] - mean) / sqrt(variance + epsilon);
    }
}
