#include "wordle.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <map>
#include <algorithm>

using namespace std;

#define DEBUG false

void Wordle::load_dictionary()
{
    ifstream potential_words_file(word_file);

    if (!potential_words_file.is_open())
    {
        throw runtime_error("Unable to open poential words file.");
    }

    string word;
    while (potential_words_file >> word)
    {
        dictionary.potential_words_str.push_back(word);
        dictionary.potential_words.push_back(encode_word(word));
    }

    if (DEBUG)
    {
        cout << "Dictionary: " << dictionary.potential_words.size() << endl;
    }
}

void Wordle::load_vocabulary()
{
    ifstream vocab_file(vocabulary_file);
    if (!vocab_file.is_open())
    {
        throw runtime_error("Unable to open poential words file.");
    }

    vocab.size = 0;
    string word;
    while (vocab_file >> word)
    {
        vocab.word_to_id[word] = vocab.size;
        vocab.id_to_word[vocab.size] = word;
        vocab.size++;
    }

    if (DEBUG)
    {
        cout << "Vocabulary: " << vocab.size << endl;
    }
}

vector<int> Wordle::encode_word(string word)
{
    vector<string> parts = tokenizer(word);
    vector<int> ids;
    for (int i = 0; i < parts.size(); i++)
    {
        if (vocab.word_to_id.count(parts[i])) {
          ids.push_back(vocab.word_to_id[parts[i]]);
        }
        
    }
    return ids;
}

std::string Wordle::decode_word(vector<int> ids)
{
    string res = "";
    for (int i = 0; i < ids.size(); i++)
        if (vocab.id_to_word.count(ids[i]))
            res += vocab.id_to_word[ids[i]];
    return res;
}

void Wordle::set_target_word()
{
    int index = rand() % dictionary.potential_words.size();
    target_word = dictionary.potential_words[index];
    cout << target_word.size() << endl;
}

vector<int> Wordle::get_target_word() {
    return target_word;
}

vector<int> Wordle::post_guess(string guess_str)
{
    vector<int> guess = encode_word(guess_str);
    if (target_word.size() != guess.size())
    {
        throw invalid_argument("Guesses must be of same length of word");
    }

    auto it = find(dictionary.potential_words.begin(), dictionary.potential_words.end(), guess);
    if (it == dictionary.potential_words.end())
    {
        throw invalid_argument("Invalid guess: Word does not exist in dictionary");
    }

    map<int, int> letters;
    for (int c : target_word)
    {
        letters[c]++;
    }

    vector<int> coloring(target_word.size());
    for (int i = 0; i < target_word.size(); i++)
    {
        int cur = guess[i];
        if (guess[i] == target_word[i])
        {
            coloring[i] = GREEN;
            letters[cur]--;
        }
    }

    for (int i = 0; i < target_word.size(); i++)
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

    state.guesses.push_back(guess);
    state.colors.push_back(coloring);
    return coloring;
}

vector<string> word_tokenizer(string input) {
    vector<string> res;
    for(int i = 0; i < input.length(); i++)
        res.push_back(string(1, input[i]));
    return res;
}