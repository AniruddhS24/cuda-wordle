#include <unordered_map>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "solver.h"
#include "../device/solver_kernels.h"
#include "../wordle.h"

using namespace std;

float Solver::calculate_expected_information(vector<int> word)
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
  cout << "Actual Information: " << log2(1 / p) << endl;
  cout << "Updated Dictionary Old Size: " << old_dict_size << " New Size: " << dictionary.size() << endl;
}

vector<int> Solver::serial_solver(GameState state)
{
  cout << "Starting Solver" << endl;
  if (state.guesses.size() > 0)
  {
    update_dictionary(state.guesses.back(), state.colors.back());
  }
  vector<int> best_guess = {};
  float highest_expected_information = -1;
  for (int i = 0; i < dictionary.size(); i++)
  {
    vector<int> current_word = dictionary[i];
    float expected_information = calculate_expected_information(current_word);
    if (expected_information > highest_expected_information)
    {
      highest_expected_information = expected_information;
      best_guess = current_word;
    }
  }
  cout << "Expected Information: " << highest_expected_information << endl;
  return best_guess;
}

vector<int> Solver::cuda_solver(GameState state)
{
  cout << "Starting CUDA Solver" << endl;
  if (state.guesses.size() > 0)
  {
    update_dictionary(state.guesses.back(), state.colors.back());
  }

  int num_words = dictionary.size();
  int *dictionary_arr = new int[num_words * 5];
  for (int i = 0; i < num_words * 5; i++)
    dictionary_arr[i] = dictionary[i / 5][i % 5];

  float *information = new float[num_words];
  for (int i = 0; i < num_words; i++)
    information[i] = 0;
  int *_dictionary;
  float *_information;
  cudaMalloc((void **)&_dictionary, num_words * 5 * sizeof(int));
  cudaMalloc((void **)&_information, num_words * sizeof(float));

  cudaMemcpy(_dictionary, dictionary_arr, num_words * 5 * sizeof(int), cudaMemcpyHostToDevice);

  calculate_expected_information_cuda_2(num_words, 5, _dictionary, _information);
  cudaDeviceSynchronize();

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