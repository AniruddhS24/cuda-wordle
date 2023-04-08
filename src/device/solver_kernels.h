#ifndef SOLVER_KERNELS_H
#define SOLVER_KERNELS_H

void calculate_expected_information_cuda(int num_words, int word_len, int *dictionary, float *information);

#endif
