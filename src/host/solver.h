#ifndef SOLVER_H
#define SOLVER_H

#include "../wordle.h"
#include <vector>

class Solver
{
    int vocab_size;
    std::vector<std::vector<int>> dictionary;
    float *prior;

public:
    Solver(int vocab_size, std::vector<std::vector<int>> dictionary) : vocab_size(vocab_size), dictionary(dictionary){};
    // prior is float[dictionary_size] where prior[i] = prior probability of word/letter i
    Solver(int vocab_size, std::vector<std::vector<int>> dictionary, float *prior) : vocab_size(vocab_size), dictionary(dictionary), prior(prior){};

    std::vector<int> dummy_solver(GameState state);
};

#endif