#include <unordered_map>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "solver.h"
#include "../device/solver_kernels.h"
#include "../wordle.h"

using namespace std;

float calculate_expected_information(vector<int> &word, vector<vector<int>> &dictionary)
{
  unordered_map<int, int> colorings;
  // cout << "Calculating Expected Info ...";
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
  // cout << "Finished E[I]" << endl;
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
  cout << "Updated Dictionary Old Size: " << old_dict_size << " New Size: " << dictionary.size() << endl;
}

vector<int> Solver::serial_solver(vector<vector<int>> guesses, vector<int> colors)
{
  cout << "Starting Solver" << endl;
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

  float *information = new float[num_words];
  int *_dictionary;
  float *_information;
  cudaMalloc((void **)&_dictionary, num_words * word_len * sizeof(int));
  cudaMalloc((void **)&_information, num_words * sizeof(float));

  cudaMemcpy(_dictionary, dictionary_arr, num_words * word_len * sizeof(int), cudaMemcpyHostToDevice);

  calculate_expected_information_cuda_shmem(num_words, word_len, _dictionary, _information);

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
  cout << "Expected Information: " << highest_expected_information << endl;
  return best_guess;
}