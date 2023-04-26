#include <unordered_map>
#include <stdio.h>
#include <iostream>
#include "serial_solver.h"
#include "wordle.h"

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

vector<pair<float, vector<int>>> SerialSolver::solve_fulldist(vector<vector<int>> guesses, vector<int> colors)
{
    for (int i = 0; i < guesses.size(); i++)
    {
        update_dictionary(guesses[i], dictionary, colors[i]);
    }
    cout << "Dictionary Size: " << dictionary.size() << endl;
    vector<pair<float, vector<int>>> dist = {};
    for (int i = 0; i < dictionary.size(); i++)
    {
        vector<int> current_word = dictionary[i];
        float expected_information = calculate_expected_information(current_word, dictionary);
        dist.push_back(make_pair(expected_information, current_word));
    }
    return dist;
}

vector<int> SerialSolver::solve(vector<vector<int>> guesses, vector<int> colors)
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