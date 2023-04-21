#include "solver_kernels.h"
#include "../wordle.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>

using namespace std;

#define BLOCK_SIZE 32

#define MAX_VOCAB_SIZE 26
#define OPTIM_COLOR_CACHE 64

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

__global__ void calculate_expected_information_kernel(int num_words, int word_len, int color_perm, int *dictionary, float *information, int *colorings)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_words)
    {
        int word_st = tid * word_len;

        for (int i = 0; i < num_words; i++)
        {
            int cur_word_st = i * word_len;
            int c = generate_coloring_cuda(&dictionary[word_st], &dictionary[cur_word_st], word_len);
            colorings[color_perm * tid + c]++;
        }

        float expected_info = 0.0;
        for (int i = 0; i < color_perm; i++)
        {
            float p = (float)colorings[color_perm * tid + i] / (float)num_words;
            if (p > 0)
                expected_info += p * log2(1 / p);
        }
        information[tid] = expected_info;
    }
}

__global__ void calculate_expected_information_kernel_coalesced_optim(int num_words, int word_len, int color_perm, int *dictionary, float *information, int *colorings)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_words)
    {
        int word_st = tid * word_len;

        for (int i = 0; i < num_words; i++)
        {
            int cur_word_st = i * word_len;
            int c = generate_coloring_cuda(&dictionary[word_st], &dictionary[cur_word_st], word_len);
            colorings[num_words * c + tid]++;
        }

        float expected_info = 0.0;
        for (int i = 0; i < color_perm; i++)
        {
            float p = (float)colorings[num_words * i + tid] / (float)num_words;
            if (p > 0)
                expected_info += p * log2(1 / p);
        }
        information[tid] = expected_info;
    }
}

// make a version of above that uses coalesced memory access

__global__ void calculate_expected_information_kernel_shmem(int num_words, int word_len, int color_perm, int *dictionary, float *information, int *colorings)
{
    extern __shared__ int s[];
    int *s_colorings = s;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;
    if (tid < num_words)
    {
        for (int i = 0; i < color_perm; i++)
            s_colorings[color_perm * lid + i] = 0;
    }

    __syncthreads();

    if (tid < num_words)
    {
        int word_st = lid * word_len;

        for (int i = 0; i < num_words; i++)
        {
            int cur_word_st = i * word_len;
            int c = generate_coloring_cuda(&dictionary[word_st], &dictionary[cur_word_st], word_len);
            s_colorings[color_perm * lid + c]++;
        }

        float expected_info = 0.0;
        for (int i = 0; i < color_perm; i++)
        {
            float p = (float)s_colorings[color_perm * lid + i] / (float)num_words;
            if (p > 0)
                expected_info += p * log2(1 / p);
        }
        information[tid] = expected_info;
    }
}

__global__ void calculate_expected_information_kernel_shmem_full(int num_words, int word_len, int color_perm, int *dictionary, float *information, int *colorings)
{
    extern __shared__ int s[];
    int *s_colorings = s;
    int *s_dictionary = (int *)&s_colorings[BLOCK_SIZE * color_perm];
    float *s_information = (float *)&s_dictionary[BLOCK_SIZE * word_len];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;
    if (tid < num_words)
    {
        for (int i = 0; i < color_perm; i++)
            s_colorings[color_perm * lid + i] = 0;
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
            s_colorings[color_perm * lid + c]++;
        }

        float expected_info = 0.0;
        for (int i = 0; i < color_perm; i++)
        {
            // if (tid == 0)
            //     printf("%p: %d\n", (void *)&s_colorings[color_perm * lid + i], s_colorings[color_perm * lid + i]);
            float p = (float)s_colorings[color_perm * lid + i] / (float)num_words;
            if (p > 0)
                expected_info += p * log2(1 / p);
        }
        s_information[lid] = expected_info;
    }

    __syncthreads();

    information[tid] = s_information[lid];
}

__global__ void calculate_expected_information_kernel_shmem_full_optim(int num_words, int word_len, int color_perm, int *dictionary, float *information, int *colorings)
{
    extern __shared__ int s[];
    int *s_colorings = s;
    int *s_dictionary = (int *)&s_colorings[BLOCK_SIZE * OPTIM_COLOR_CACHE];
    float *s_information = (float *)&s_dictionary[BLOCK_SIZE * word_len];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;
    if (tid < num_words)
    {
        for (int i = 0; i < OPTIM_COLOR_CACHE; i++)
            s_colorings[OPTIM_COLOR_CACHE * lid + i] = 0;
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
            if (c < OPTIM_COLOR_CACHE)
                s_colorings[OPTIM_COLOR_CACHE * lid + c]++;
            else
                colorings[color_perm * tid + c]++;
        }

        float expected_info = 0.0;
        for (int i = 0; i < color_perm; i++)
        {
            float p = 0;
            if (i < OPTIM_COLOR_CACHE)
            {
                p = (float)s_colorings[OPTIM_COLOR_CACHE * lid + i] / (float)num_words;
            }
            else
                p = (float)colorings[color_perm * tid + i] / (float)num_words;
            // if (tid == 0)
            //     printf("%d\n", s_colorings[OPTIM_COLOR_CACHE * lid + i]);
            if (p > 0)
                expected_info += p * log2(1 / p);
        }
        s_information[lid] = expected_info;
    }

    __syncthreads();

    information[tid] = s_information[lid];
}

__global__ void calculate_expected_information_kernel_percolor(int num_words, int word_len, int color_perm, int *dictionary, float *information, int *colorings)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_words * color_perm)
    {
        int word_id = tid / color_perm;
        int word_st = word_id * word_len;
        int cur_color = tid % color_perm;
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

float calculate_occupancy_shmem(void (*kernel)(int, int, int, int *, float *, int *), int num_active_threads)
{
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

float calculate_occupancy(void (*kernel)(int, int, int, int *, float *, int *), int num_active_threads)
{
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

void calculate_expected_information_cuda(int num_words, int word_len, int color_perm, int *dictionary, float *information, int *colorings)
{
    dim3 blockGrid((num_words + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threadBlock(BLOCK_SIZE);

    calculate_expected_information_kernel<<<blockGrid, threadBlock>>>(num_words, word_len, color_perm, dictionary, information, colorings);
    cudaDeviceSynchronize(); // wait for kernel to complete
    calculate_occupancy(calculate_expected_information_kernel, num_words);
}

void calculate_expected_information_cuda_coalesced_optim(int num_words, int word_len, int color_perm, int *dictionary, float *information, int *colorings)
{
    dim3 blockGrid((num_words + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threadBlock(BLOCK_SIZE);

    calculate_expected_information_kernel_coalesced_optim<<<blockGrid, threadBlock>>>(num_words, word_len, color_perm, dictionary, information, colorings);
    cudaDeviceSynchronize(); // wait for kernel to complete
    calculate_occupancy(calculate_expected_information_kernel_coalesced_optim, num_words);
}

void calculate_expected_information_cuda_shmem(int num_words, int word_len, int color_perm, int *dictionary, float *information, int *colorings)
{
    dim3 blockGrid((num_words + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threadBlock(BLOCK_SIZE);
    calculate_expected_information_kernel_shmem<<<blockGrid, threadBlock, BLOCK_SIZE * color_perm * sizeof(int)>>>(num_words, word_len, color_perm, dictionary, information, colorings);
    calculate_occupancy_shmem(calculate_expected_information_kernel_shmem, num_words);
}

void calculate_expected_information_cuda_shmem_full(int num_words, int word_len, int color_perm, int *dictionary, float *information, int *colorings)
{
    dim3 blockGrid((num_words + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threadBlock(BLOCK_SIZE);
    calculate_expected_information_kernel_shmem_full<<<blockGrid, threadBlock, BLOCK_SIZE * color_perm * sizeof(int) + BLOCK_SIZE * word_len * sizeof(int) + BLOCK_SIZE * sizeof(float)>>>(num_words, word_len, color_perm, dictionary, information, colorings);
    calculate_occupancy_shmem(calculate_expected_information_kernel_shmem_full, num_words);
}

void calculate_expected_information_cuda_shmem_full_optim(int num_words, int word_len, int color_perm, int *dictionary, float *information, int *colorings)
{
    dim3 blockGrid((num_words + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threadBlock(BLOCK_SIZE);
    calculate_expected_information_kernel_shmem_full_optim<<<blockGrid, threadBlock, BLOCK_SIZE * OPTIM_COLOR_CACHE * sizeof(int) + BLOCK_SIZE * word_len * sizeof(int) + BLOCK_SIZE * sizeof(float)>>>(num_words, word_len, color_perm, dictionary, information, colorings);
    // calculate_occupancy_shmem(calculate_expected_information_kernel_shmem_full_optim, num_words);
}

void calculate_expected_information_cuda_percolor(int num_words, int word_len, int color_perm, int *dictionary, float *information, int *colorings)
{
    dim3 blockGrid((num_words * color_perm + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threadBlock(BLOCK_SIZE);
    calculate_expected_information_kernel_percolor<<<blockGrid, threadBlock>>>(num_words, word_len, color_perm, dictionary, information, colorings);
    cudaDeviceSynchronize(); // wait for kernel to complete
    calculate_occupancy(calculate_expected_information_kernel_percolor, num_words * color_perm);
}