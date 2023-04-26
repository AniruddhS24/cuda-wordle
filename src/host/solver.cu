#include <unordered_map>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "solver.h"
#include "../device/solver_kernels.h"
#include "../wordle.h"

using namespace std;

vector<int> Solver::solve(vector<vector<int>> guesses, vector<int> colors)
{
  cout << "Starting CUDA Solver" << endl;
  if (guesses.size() > 0)
  {
    update_dictionary(guesses.back(), dictionary, colors.back());
  }
  int num_words = dictionary.size();
  int *dictionary_arr = new int[num_words * word_len];
  for (int i = 0; i < num_words * word_len; i++)
    dictionary_arr[i] = dictionary[i / word_len][i % word_len];

  int color_perm = num_color_perm(word_len);

  float *information = new float[num_words];
  int *colorings = new int[num_words * color_perm];
  for (int i = 0; i < num_words * color_perm; i++)
    colorings[i] = 0;
  int *_dictionary;
  float *_information;
  int *_colorings;
  cudaMalloc((void **)&_dictionary, num_words * word_len * sizeof(int));
  cudaMalloc((void **)&_information, num_words * sizeof(float));
  cudaMalloc((void **)&_colorings, num_words * color_perm * sizeof(int));

  cudaMemcpy(_colorings, colorings, num_words * color_perm * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(_dictionary, dictionary_arr, num_words * word_len * sizeof(int), cudaMemcpyHostToDevice);
  calculate_expected_information_cuda(num_words, word_len, color_perm, _dictionary, _information, _colorings);
  // calculate_expected_information_cuda_shmem_full(num_words, word_len, color_perm, _dictionary, _information, _colorings);

  cudaMemcpy(information, _information, num_words * sizeof(float), cudaMemcpyDeviceToHost);

  vector<int> best_guess = {};
  float highest_expected_information = -1;
  for (int i = 0; i < dictionary.size(); i++)
  {
    vector<int> current_word = dictionary[i];
    float expected_information = information[i];
    if (expected_information > highest_expected_information)
    {
      highest_expected_information = expected_information;
      best_guess = current_word;
    }
  }

  cudaFree(_dictionary);
  cudaFree(_information);
  cudaFree(_colorings);
  cout << "Expected Information: " << highest_expected_information << endl;
  return best_guess;
}