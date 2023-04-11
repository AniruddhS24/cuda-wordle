#include "solver_kernels.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>

using namespace std;

#define BLOCK_SIZE 32

__global__ void calculate_expected_information_kernel(int num_words, int word_len, int *dictionary, float *information)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_words)
    {
        int colorings[243]; // num_colorings (3^word_len)
        int word_st = tid * word_len;

        for (int i = 0; i < num_words; i++)
        {
            int cur_word_st = i * word_len;

            int coloring[5];   // word_len
            int letter_ct[27]; // vocab_size

            // compute coloring
            for (int j = 0; j < word_len; j++)
                coloring[j] = 0;
            for (int j = 0; j < 27; j++)
                letter_ct[j] = 0;
            for (int j = 0; j < word_len; j++)
                letter_ct[dictionary[word_st + j]]++;
            for (int j = 0; j < word_len; j++)
            {
                int cur = dictionary[cur_word_st + j];
                if (cur == dictionary[word_st + j])
                {
                    coloring[j] = 1;
                    letter_ct[cur]--;
                }
            }

            for (int j = 0; j < word_len; j++)
            {
                int cur = dictionary[cur_word_st + j];
                if (coloring[j] == 1)
                    continue;
                if (letter_ct[cur] > 0)
                {
                    coloring[j] = 2;
                    letter_ct[cur]--;
                }
                else
                {
                    coloring[j] = 0;
                }
            }

            // convert coloring to base 3
            int base = 1;
            int c = 0;
            for (int j = 0; j < 5; j++)
            {
                c += base * coloring[j];
                base *= 3;
            }

            // if (tid == 1)
            //     printf("tid %d Coloring: %d\n", tid, c);

            // increment coloring
            colorings[c]++;
        }

        float expected_info = 0.0;
        for (int i = 0; i < 243; i++)
        {
            float p = (float)colorings[i] / (float)num_words;
            if (p > 0)
                expected_info += p * log2(1 / p);
        }
        information[tid] = expected_info;
    }
}

__global__ void calculate_expected_information_kernel_2(int num_words, int word_len, int *dictionary, float *information)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_words * 243)
    {
        int word_id = tid / 243;
        int word_st = word_id * word_len;
        int cur_color = tid % 243;
        int prob = 0;
        for (int i = 0; i < num_words; i++)
        {
            int cur_word_st = i * word_len;

            int coloring[5];   // word_len
            int letter_ct[27]; // vocab_size

            // compute coloring
            for (int j = 0; j < word_len; j++)
                coloring[j] = 0;
            for (int j = 0; j < 27; j++)
                letter_ct[j] = 0;
            for (int j = 0; j < word_len; j++)
                letter_ct[dictionary[word_st + j]]++;
            for (int j = 0; j < word_len; j++)
            {
                int cur = dictionary[cur_word_st + j];
                if (cur == dictionary[word_st + j])
                {
                    coloring[j] = 1;
                    letter_ct[cur]--;
                }
            }

            for (int j = 0; j < word_len; j++)
            {
                int cur = dictionary[cur_word_st + j];
                if (coloring[j] == 1)
                    continue;
                if (letter_ct[cur] > 0)
                {
                    coloring[j] = 2;
                    letter_ct[cur]--;
                }
                else
                {
                    coloring[j] = 0;
                }
            }

            // convert coloring to base 3
            int base = 1;
            int c = 0;
            for (int j = 0; j < 5; j++)
            {
                c += base * coloring[j];
                base *= 3;
            }

            // increment coloring
            if (c == cur_color)
                prob++;
        }
        float p = (float)prob / (float)num_words;
        if (p > 0)
            atomicAdd(&information[word_id], (float)p * log2(1 / p));
    }
}

float calculate_occupancy(void (*kernel)(int, int, int*, float*), int num_active_threads) {
    int devId = 0;
    cudaSetDevice(devId);

    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, devId);

    // max potential warps based on device properties
    int num_sm = devProp.multiProcessorCount;
    int max_block_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_block_per_sm,
        kernel,
        BLOCK_SIZE,
        0);
    int max_warps_per_sm = devProp.maxThreadsPerMultiProcessor / devProp.warpSize;
    int max_num_warps = max_warps_per_sm * max_block_per_sm * num_sm;

    // actual warps used by kernel invocation
    int num_active_warps = (num_active_threads + devProp.warpSize - 1) / devProp.warpSize;
    float occupancy = (float)num_active_warps / max_num_warps;

    std::cout << "Occupancy: " << occupancy << std::endl;
    return occupancy;
    
}

void calculate_expected_information_cuda(int num_words, int word_len, int *dictionary, float *information)
{
    dim3 blockGrid((num_words + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threadBlock(BLOCK_SIZE);

    calculate_expected_information_kernel<<<blockGrid, threadBlock>>>(num_words, word_len, dictionary, information);
    cudaDeviceSynchronize(); // wait for kernel to complete
    calculate_occupancy(calculate_expected_information_kernel, num_words);
}

void calculate_expected_information_cuda_2(int num_words, int word_len, int *dictionary, float *information)
{
    dim3 blockGrid((num_words * 243 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threadBlock(BLOCK_SIZE);
    calculate_expected_information_kernel_2<<<blockGrid, threadBlock>>>(num_words, word_len, dictionary, information);
    cudaDeviceSynchronize(); // wait for kernel to complete
    calculate_occupancy(calculate_expected_information_kernel_2, num_words * 243);
}