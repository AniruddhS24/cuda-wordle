#include "run_args.h"
#include "wordle.h"
#include "host/solver.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

using namespace std;

int main(int argc, char **argv)
{

    auto start_e2e = std::chrono::high_resolution_clock::now();
    srand(std::time(nullptr));
    Arguments args = parse_arguments(argc, argv);
    Wordle wordle{args.vocab_filepath, args.dictionary_filepath, word_tokenizer};
    wordle.load_vocabulary();
    wordle.load_dictionary();
    wordle.set_target_word();

    vector<int> tmp = wordle.get_target_word();
    string actual = wordle.decode_word(tmp);
    cout << "Target Word: " << actual << endl;
    cout << endl;

    Solver solver{wordle.vocab.size, wordle.dictionary.potential_words};


    int num_guesses = 1;
    while (num_guesses <= 5)
    {
        string guess;
        vector<int> solver_guess = {};
        auto start_solver = chrono::high_resolution_clock::now();
        if (args.use_gpu) {
          solver_guess  = solver.cuda_solver(wordle.state);        
        } else {
          solver_guess = solver.serial_solver(wordle.state);
        }
        auto end_solver = chrono::high_resolution_clock::now();
        cout << "Solver Iteration Time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end_solver - start_solver).count() << endl;

        
        cout << "Solver guessed: " << wordle.decode_word(solver_guess) << endl;
        cout << "Guess: ";
        if (args.interactive) {
          cin >> guess;
        } else {
          guess = wordle.decode_word(solver_guess);
        }
        
        int color = wordle.post_guess(guess);
        cout << "Colors: " << endl;
        bool solved = true;
        for (int i = 0; i < tmp.size(); i++)
        {
            cout << Wordle::get_coloring_bit(color, i) << " ";
            if (Wordle::get_coloring_bit(color, i) != GREEN)
            {
                solved = false;
            }
        }
        cout << endl;

        if (solved)
        {
            cout << endl;
            cout << "Solved in " << num_guesses << endl;
            break;
        }
        num_guesses++;
        cout << endl; 
    }
    auto end_e2e = chrono::high_resolution_clock::now();
    cout << "End to End Time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end_e2e - start_e2e).count() << endl;
}