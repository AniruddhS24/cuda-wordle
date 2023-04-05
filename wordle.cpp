#include <iostream>
#include <fstream>
#include <random>
#include "wordle.h"
using namespace std;



void load_dictionary() {
  ifstream guesses_file(GUESSES);
  ifstream potential_words_file(POTENTIAL_WORDS);

  if (!guesses_file.is_open()) {
    throw runtime_error("Unable to open guesses file.");
  }
  if (!potential_words_file.is_open()) {
    throw runtime_error("Unable to open poential words file.");
  }

  string word;
  while (guesses_file >> word) {
    guesses.push_back(word);
  }
  while (potential_words_file >> word) {
    potential_words.push_back(word);
  }
  
  dictionary.reserve(guesses.size() + potential_words.size());
  dictionary.insert(dictionary.end(), guesses.begin(), guesses.end());
  dictionary.insert(dictionary.end(), potential_words.begin(), potential_words.end());

  if (DEBUG) {
    cout << "Guesses: " << guesses.size() << endl;
    cout << "Potential Words: " << potential_words.size() << endl;
    cout << "Dictionary: " << dictionary.size() << endl;
  }
}

string post_guess(string guess, string word) {

  if(word.length() != guess.length()) {
    throw invalid_argument("Guesses must be of same length of word");
  }

  int letters[26] = {};
  for (char c : word) {
    letters[c - 'a']++;
  }

  string coloring (word.length(), 'x');
  for(int i = 0; i < word.length(); i++) {
    int cur = guess[i] - 'a';
    if (guess[i] == word[i]) {
      coloring[i] = GREEN;
      letters[cur]--;
    } 
  }

  for (int i = 0; i < word.length(); i++) {
    int cur = guess[i] - 'a';
    if (coloring[i] == GREEN) {
      continue;
    }
    if (letters[cur] > 0) {
      coloring[i] = YELLOW;
      letters[cur]--;
    } else {
      coloring[i] = GRAY;
    }
  } 

  return coloring;
}

int main(int argc, char *argv[]) {
  load_dictionary();

  if (DEBUG) {
    srand(123);
  }

  int index = rand() % potential_words.size();
  string word = potential_words[index];

  if (DEBUG) {
    cout << "Selected Word: " << word << endl;
  }
  
  int num_guesses = 5;
  while (num_guesses >= 0) {
    string guess;
    cout << "Guess: ";
    cin >> guess;

    string coloring = post_guess(guess, word);
    cout << "Coloring: " << coloring << endl;
  }
}