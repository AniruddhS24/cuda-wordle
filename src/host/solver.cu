#include <unordered_map>
#include <stdio.h>
#include <iostream>
#include "solver.h"
#include "../wordle.h"

using namespace std;

vector<int> Solver::generate_coloring(vector<int> word, vector<int> guess) {
  vector<int> coloring(word.size());
  map<int, int> letters;
  for (int c : word)
  {
      letters[c]++;
  }
  for (int i = 0; i < word.size(); i++)
  {
      int cur = guess[i];
      if (guess[i] == word[i])
      {
          coloring[i] = GREEN;
          letters[cur]--;
      }
  }

  for (int i = 0; i < word.size(); i++)
  {
      int cur = guess[i];
      if (coloring[i] == GREEN)
      {
          continue;
      }
      if (letters[cur] > 0)
      {
          coloring[i] = YELLOW;
          letters[cur]--;
      }
      else
      {
          coloring[i] = GRAY;
      }
  }
  return coloring;
}

float Solver::calculate_expected_information(vector<int> word) {
  unordered_map<string, int> colorings;
  // cout << "Calculating Expected Info ...";
  for (int i = 0; i < dictionary.size(); i++) {
    vector<int> current_word = dictionary[i];
    vector<int> coloring = generate_coloring(word, current_word);

    string coloring_str = "";
    for (int c : coloring) {
      coloring_str += ('0' + c);
    }
    if (colorings.count(coloring_str)) {
      colorings[coloring_str]++;
    } else {
      colorings[coloring_str] = 1;
    }
  }

  float expected_information = 0.0f;
  for (auto it = colorings.begin(); it != colorings.end(); it++) {
    string key = it->first;
    int occurances = it->second;
    float p = float (occurances) / dictionary.size();
    if (p > 0) {
      expected_information += p * log2(1 / p);
    }
  }
  // cout << "Finished E[I]" << endl;
  return expected_information;
}

void Solver::update_dictionary(vector<int> guess, vector<int> color) {
  int old_dict_size = dictionary.size();
  for (auto it = dictionary.begin(); it != dictionary.end();) {
    vector<int> coloring = generate_coloring(*it, guess);

    bool same = true;
    for (int i = 0; i < coloring.size(); i++) {
      if (coloring[i] != color[i]) {
        same = false;
        break;
      }
    }

    if (!same) {
      it = dictionary.erase(it);
    } else {
      it++;
    }
  }
  float p = float(dictionary.size()) / old_dict_size;
  cout << "Actual Information: " << log2(1 / p) << endl;
  cout << "Updated Dictionary Old Size: " << old_dict_size << " New Size: " << dictionary.size() << endl;
}


vector<int> Solver::dummy_solver(GameState state) {
    cout << "Starting Solver" << endl;
    if (state.guesses.size() > 0) {
      update_dictionary(state.guesses.back(), state.colors.back());
    }
    vector<int> best_guess = {};
    float highest_expected_information = -1;
    for (int i = 0; i < dictionary.size(); i++) {
      vector<int> current_word = dictionary[i];
      float expected_information = calculate_expected_information(current_word);
      if (expected_information > highest_expected_information) {
        highest_expected_information = expected_information;
        best_guess = current_word;
      }
    }
    cout << "Expected Information: " << highest_expected_information << endl;
    return best_guess;
}