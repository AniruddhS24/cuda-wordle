#include <unordered_map>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "solver.h"
#include "../device/solver_kernels.h"
#include "../wordle.h"

using namespace std;

vector<int> calculate_expected_information(vector<int> &word, vector<vector<int>> &dictionary)
{
  vector<int> colorings(59049);
  for (int i = 0; i < dictionary.size(); i++)
  {
    vector<int> current_word = dictionary[i];
    int coloring = Wordle::generate_coloring(word, current_word);
    colorings[coloring]++;
  }

  // float expected_information = 0.0f;
  // for (int occurances : colorings)
  // {
  //   float p = float(occurances) / dictionary.size();
  //   if (p > 0)
  //   {
  //     expected_information += p * log2(1 / p);
  //   }
  // }
  return colorings;
}

void Solver::update_dictionary(vector<int> guess, int color)
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
  cout << "Old Dict Size: " << old_dict_size << " New Dict Size: " << dictionary.size() << endl;
  cout << "Actual Information: " << log2(1 / p) << endl;
}

vector<pair<float, pair<vector<int>, vector<int>>>> Solver::serial_solver(vector<vector<int>> guesses, vector<int> colors)
{
  vector<pair<float, pair<vector<int>, vector<int>>>> info = {};
  vector<int> best_guess = {};
  float highest_expected_information = -1;
  for (int i = 0; i < dictionary.size(); i++)
  {
    vector<int> current_word = dictionary[i];
    // cout << "calculating expected info" << endl;
    vector<int> colorings = calculate_expected_information(current_word, dictionary);

    float expected_information = 0.0f;
    for (int occurances : colorings)
    {
      float p = float(occurances) / dictionary.size();
      if (p > 0)
      {
        expected_information += p * log2(1 / p);
      }
    }

    if (expected_information > highest_expected_information)
    {
      highest_expected_information = expected_information;
      best_guess = current_word;
    }
    info.push_back({expected_information, {current_word, colorings}});
  }

  sort(info.begin(), info.end());
  return info;
}

vector<pair<float, vector<int>>> Solver::cuda_solver(vector<vector<int>> guesses, vector<int> colors, bool shmem, bool multi_color)
{

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

  if (shmem) {
    calculate_expected_information_cuda_shmem_full(num_words, word_len, _dictionary, _information);
  } else if (multi_color) {
    calculate_expected_information_cuda_percolor(num_words, word_len, _dictionary, _information);
  } else {
    calculate_expected_information_cuda(num_words, word_len, _dictionary, _information);
  }
  

  cudaMemcpy(information, _information, num_words * sizeof(float), cudaMemcpyDeviceToHost);

  vector<pair<float, vector<int>>> info = {};
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
    info.push_back({expected_information, current_word});
  }
  sort(info.begin(), info.end());
  return info;
}