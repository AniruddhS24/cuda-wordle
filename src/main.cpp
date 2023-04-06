#include "run_args.h"
#include "wordle.h"
#include "host/solver.h"
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main(int argc, char **argv)
{
    Arguments args = parse_arguments(argc, argv);
    cout << "running success!" << endl;
    Wordle wordle{args.vocab_filepath, args.dictionary_filepath, word_tokenizer};
    wordle.load_vocabulary();
    wordle.load_dictionary();
    wordle.set_target_word();

    for ( const auto &p : wordle.vocab.word_to_id)
        cout << p.first << " " << p.second << endl;
    vector<int> tmp = wordle.get_target_word();
    for(int i = 0; i < tmp.size(); i++)
        cout << tmp[i] << " ";
    cout << endl;

    string actual = wordle.decode_word(tmp);
    cout << actual << endl;

    Solver solver{wordle.vocab.size, wordle.dictionary.potential_words};

    int num_guesses = 5;
    while (num_guesses >= 0) {
        string guess;
        vector<int> solver_guess = solver.dummy_solver(wordle.state);
        cout << "Solver guessed: " << wordle.decode_word(solver_guess) << endl;
        cout << "Guess: ";
        cin >> guess;
        vector<int> colors = wordle.post_guess(guess);
        cout << "Colors: " << endl;
        for(int i = 0; i < tmp.size(); i++)
            cout << colors[i] << " ";
        cout << endl;
    }
    
}