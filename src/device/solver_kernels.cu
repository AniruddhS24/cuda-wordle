#include "solver_kernels.h"
#include "../wordle.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>

using namespace std;

#define BLOCK_SIZE 32
#define MAX_COLOR_PERM 243
#define MAX_VOCAB_SIZE 26

__device__ int set_base3_bit_cuda(int coloring, int pos, int value)
{
    int base = 1;
    for (int i = 0; i < pos; i++)
        base *= 3;
    coloring += value * base;
    return coloring;
}

__device__ int get_base3_bit_cuda(int coloring, int pos)
{
    for (int i = 0; i < pos; i++)
        coloring /= 3;
    return coloring % 3;
}

__device__ int generate_coloring_cuda(int *word, int *guess, int word_len)
{
    int coloring = 0;
    int letters[MAX_VOCAB_SIZE] = {};
    for (int i = 0; i < word_len; i++)
    {
        letters[word[i]]++;
    }

    for (int i = 0; i < word_len; i++)
    {
        int cur = guess[i];
        if (guess[i] == word[i])
        {
            coloring = set_base3_bit_cuda(coloring, i, GREEN);
            letters[cur]--;
        }
    }

    for (int i = 0; i < word_len; i++)
    {
        int cur = guess[i];
        if (get_base3_bit_cuda(coloring, i) == GREEN)
        {
            continue;
        }
        if (letters[cur] > 0)
        {
            coloring = set_base3_bit_cuda(coloring, i, YELLOW);
            letters[cur]--;
        }
        else
        {
            coloring = set_base3_bit_cuda(coloring, i, GRAY);
        }
    }

    return coloring;
}

__global__ void calculate_expected_information_kernel(int num_words, int word_len, int *dictionary, float *information)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_words)
    {
        int colorings[MAX_COLOR_PERM] = {};
        int word_st = tid * word_len;

        for (int i = 0; i < num_words; i++)
        {
            int cur_word_st = i * word_len;
            colorings[generate_coloring_cuda(&dictionary[word_st], &dictionary[cur_word_st], word_len)]++;
        }

        float expected_info = 0.0;
        for (int i = 0; i < MAX_COLOR_PERM; i++)
        {
            float p = (float)colorings[i] / (float)num_words;
            if (p > 0)
                expected_info += p * log2(1 / p);
        }
        information[tid] = expected_info;
    }
}

__global__ void calculate_expected_information_kernel_shmem(int num_words, int word_len, int *dictionary, float *information)
{
    __shared__ int s_colorings[BLOCK_SIZE * MAX_COLOR_PERM];
    __shared__ int s_dictionary[BLOCK_SIZE * 5];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;
    if (tid < num_words)
    {
        for (int i = 0; i < MAX_COLOR_PERM; i++)
            s_colorings[MAX_COLOR_PERM * lid + i] = 0;
        for (int i = 0; i < word_len; i++)
            s_dictionary[word_len * lid + i] = dictionary[word_len * tid + i];
    }

    __syncthreads();

    if (tid < num_words)
    {
        int word_st = lid * word_len;

        for (int i = 0; i < num_words; i++)
        {
            int cur_word_st = i * word_len;
            int c = generate_coloring_cuda(&s_dictionary[word_st], &dictionary[cur_word_st], word_len);
            s_colorings[MAX_COLOR_PERM * lid + c]++;
        }

        float expected_info = 0.0;
        for (int i = 0; i < MAX_COLOR_PERM; i++)
        {
            float p = (float)s_colorings[MAX_COLOR_PERM * lid + i] / (float)num_words;
            if (p > 0)
                expected_info += p * log2(1 / p);
        }
        information[tid] = expected_info;
    }
}

__global__ void calculate_expected_information_kernel_percolor(int num_words, int word_len, int *dictionary, float *information)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_words * MAX_COLOR_PERM)
    {
        int word_id = tid / MAX_COLOR_PERM;
        int word_st = word_id * word_len;
        int cur_color = tid % MAX_COLOR_PERM;
        int prob = 0;
        for (int i = 0; i < num_words; i++)
        {
            int cur_word_st = i * word_len;
            if (generate_coloring_cuda(&dictionary[word_st], &dictionary[cur_word_st], word_len) == cur_color)
                prob++;
        }
        float p = (float)prob / (float)num_words;
        if (p > 0)
            atomicAdd(&information[word_id], (float)p * log2(1 / p));
    }
}

void calculate_expected_information_cuda(int num_words, int word_len, int *dictionary, float *information)
{
    dim3 blockGrid((num_words + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threadBlock(BLOCK_SIZE);
    calculate_expected_information_kernel<<<blockGrid, threadBlock>>>(num_words, word_len, dictionary, information);
}

void calculate_expected_information_cuda_shmem(int num_words, int word_len, int *dictionary, float *information)
{
    dim3 blockGrid((num_words + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threadBlock(BLOCK_SIZE);
    calculate_expected_information_kernel_shmem<<<blockGrid, threadBlock>>>(num_words, word_len, dictionary, information);
}

void calculate_expected_information_cuda_percolor(int num_words, int word_len, int *dictionary, float *information)
{
    dim3 blockGrid((num_words * MAX_COLOR_PERM + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threadBlock(BLOCK_SIZE);
    calculate_expected_information_kernel_percolor<<<blockGrid, threadBlock>>>(num_words, word_len, dictionary, information);
}