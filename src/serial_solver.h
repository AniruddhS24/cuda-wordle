#ifndef SERIAL_SOLVER_H
#define SERIAL_SOLVER_H

#include "wordle.h"
#include <vector>

class SerialSolver
{
    int vocab_size;
    int word_len;
    std::vector<std::vector<int>> dictionary;
    // prior is float[dictionary_size] where prior[i] = prior probability of word/letter i
    float *prior;

public:
    SerialSolver(int vocab_size, int word_len, std::vector<std::vector<int>> dictionary) : vocab_size(vocab_size), word_len(word_len), dictionary(dictionary){};
    SerialSolver(int vocab_size, int word_len, std::vector<std::vector<int>> dictionary, float *prior) : vocab_size(vocab_size), word_len(word_len), dictionary(dictionary), prior(prior){};

    std::vector<int> solve(std::vector<std::vector<int>> guesses, std::vector<int> colors);
    std::vector<std::pair<float, std::vector<int>>> solve_fulldist(std::vector<std::vector<int>> guesses, std::vector<int> colors);

    // float calculate_expected_information(std::vector<int> word);
    // void update_dictionary(std::vector<int> guess, int color);
};

#endif