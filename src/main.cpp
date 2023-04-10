#include "run_args.h"
#include "wordle.h"
#include "host/solver.h"
#include "util.h"
#include <iostream>
#include <time.h>
#include <vector>
#include <string>

using namespace std;

int main(int argc, char **argv)
{
    srand(time(NULL));
    Arguments args = parse_arguments(argc, argv);
    cout << "running success!" << endl;
    Wordle wordle{args.vocab_filepath, args.dictionary_filepath, word_tokenizer};
    wordle.load_vocabulary();
    wordle.load_dictionary();
    wordle.set_target_word();

    // for (const auto &p : wordle.vocab.word_to_id)
    //     cout << p.first << " " << p.second << endl;
    // vector<int> tmp = wordle.get_target_word();
    // for (int i = 0; i < tmp.size(); i++)
    //     cout << tmp[i] << " ";
    // cout << endl;

    // cout << "Actual word is: " << wordle.decode_word(wordle.get_target_word()) << endl;

    Solver solver{wordle.vocab.size, 5, wordle.dictionary.potential_words};

    int num_guesses = 1;
    while (num_guesses <= 5)
    {
        string guess;
        vector<int> solver_guess = solver.cuda_solver(wordle.state.guesses, wordle.state.colors);
        cout << "Solver guessed: " << wordle.decode_word(solver_guess) << endl;
        cout << "Guess: ";
        cin >> guess;
        int color = wordle.post_guess(guess);
        cout << "Colors: " << endl;
        for (int i = 0; i < 5; i++)
            cout << guess[i] << " ";
        cout << endl;
        bool solved = true;
        for (int i = 0; i < 5; i++)
        {
            int bit = get_base3_bit(color, i);
            if (bit == GREEN)
                cout << "G ";
            else if (bit == YELLOW)
                cout << "Y ";
            else
                cout << "  ";
            if (get_base3_bit(color, i) != GREEN)
            {
                solved = false;
            }
        }
        cout << endl;

        if (solved)
        {
            cout << "Solved in " << num_guesses << endl;
            break;
        }
        num_guesses++;
    }
}