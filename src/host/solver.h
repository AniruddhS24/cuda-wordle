#ifndef SOLVER_H
#define SOLVER_H

#include "../wordle.h"
#include <bits/stdc++.h>
#include <vector>

class Solver
{
    int vocab_size;
    int word_len;
    std::vector<std::vector<int>> dictionary;
    // prior is float[dictionary_size] where prior[i] = prior probability of word/letter i
    float *prior;

public:
    Solver(int vocab_size, int word_len, std::vector<std::vector<int>> dictionary) : vocab_size(vocab_size), word_len(word_len), dictionary(dictionary){};
    Solver(int vocab_size, int word_len, std::vector<std::vector<int>> dictionary, float *prior) : vocab_size(vocab_size), word_len(word_len), dictionary(dictionary), prior(prior){};

    // {info, {word, occurances }}
    std::vector<std::pair<float, std::vector<int>>> cuda_solver(std::vector<std::vector<int>> guesses, std::vector<int> colors, bool shmem, bool multi_color);
    std::vector<std::pair<float, std::pair<std::vector<int>, std::vector<int>>>> serial_solver(std::vector<std::vector<int>> guesses, std::vector<int> colors);
    // std::vector<std::pair<float, std::vector<int>>> info_distribution(std::vector<std::vector<int>> guesses, std::vector<int> colors);
    // float calculate_expected_information(std::vector<int> word);
    void update_dictionary(std::vector<int> guess, int color);
};

#endif