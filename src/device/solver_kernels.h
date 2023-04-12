#ifndef SOLVER_KERNELS_H
#define SOLVER_KERNELS_H

void calculate_expected_information_cuda(int num_words, int word_len, int *dictionary, float *information);
void calculate_expected_information_cuda_shmem(int num_words, int word_len, int *dictionary, float *information);
void calculate_expected_information_cuda_shmem_full(int num_words, int word_len, int *dictionary, float *information);
void calculate_expected_information_cuda_percolor(int num_words, int word_len, int *dictionary, float *information);

#endif
