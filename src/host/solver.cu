#include <unordered_map>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "solver.h"
#include "../device/solver_kernels.h"
#include "../wordle.h"

using namespace std;

int num_color_perm(int word_len)
{
  int base = 1;
  for (int i = 0; i < word_len; i++)
    base *= 3;
  return base;
}

float calculate_expected_information(vector<int> &word, vector<vector<int>> &dictionary)
{
  unordered_map<int, int> colorings;
  for (int i = 0; i < dictionary.size(); i++)
  {
    vector<int> current_word = dictionary[i];
    int coloring = Wordle::generate_coloring(word, current_word);
    if (colorings.count(coloring))
    {
      colorings[coloring]++;
    }
    else
    {
      colorings[coloring] = 1;
    }
  }

  float expected_information = 0.0f;
  for (auto it = colorings.begin(); it != colorings.end(); it++)
  {
    int key = it->first;
    int occurances = it->second;
    float p = float(occurances) / dictionary.size();
    if (p > 0)
    {
      expected_information += p * log2(1 / p);
    }
  }
  return expected_information;
}

void update_dictionary(vector<int> &guess, vector<vector<int>> &dictionary, int color)
{
  int old_dict_size = dictionary.size();
  for (auto it = dictionary.begin(); it != dictionary.end();)
  {
    int coloring = Wordle::generate_coloring(*it, guess);
    if (coloring != color)
    {
      it = dictionary.erase(it);
    }
    else
    {
      it++;
    }
  }
  float p = float(dictionary.size()) / old_dict_size;
  cout << "Actual Information: " << log2(1 / p) << endl;
}

vector<int> Solver::serial_solver(vector<vector<int>> guesses, vector<int> colors)
{
  if (guesses.size() > 0)
  {
    update_dictionary(guesses.back(), dictionary, colors.back());
  }
  vector<int> best_guess = {};
  float highest_expected_information = -1;
  for (int i = 0; i < dictionary.size(); i++)
  {
    vector<int> current_word = dictionary[i];
    float expected_information = calculate_expected_information(current_word, dictionary);
    if (expected_information > highest_expected_information)
    {
      highest_expected_information = expected_information;
      best_guess = current_word;
    }
  }
  cout << "Expected Information: " << highest_expected_information << endl;
  return best_guess;
}

vector<int> Solver::cuda_solver(vector<vector<int>> guesses, vector<int> colors)
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