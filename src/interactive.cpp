#include "run_args.h"
#include "wordle.h"
#include "serial_solver.h"
#include "util.h"
#include <iostream>
#include <time.h>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <sstream>

using namespace std;

int main(int argc, char **argv)
{
    auto start_e2e = std::chrono::high_resolution_clock::now();
    srand(std::time(nullptr));
    Arguments args = parse_arguments(argc, argv);
    Wordle wordle{args.vocab_filepath, args.dictionary_filepath, word_tokenizer};
    wordle.load_vocabulary();
    wordle.load_dictionary();
    // cout << "Guesses: " << args.guesses << endl;
    // cout << "Colors: " << args.colors << endl;
    vector<vector<int>> guesses_ids;
    vector<int> color_ids;
    string sg;
    stringstream ssg(args.guesses);
    while (getline(ssg, sg, ' '))
    {
        guesses_ids.push_back(wordle.encode_word(sg));
    }

    string sc;
    stringstream ssc(args.colors);
    while (getline(ssc, sc, ' '))
    {
        int coloring = 0;
        for (int i = 0; i < sc.length(); i++)
        {
            if (sc[i] == 'G')
                coloring = set_base3_bit(coloring, i, GREEN);
            else if (sc[i] == 'Y')
                coloring = set_base3_bit(coloring, i, YELLOW);
            else
                coloring = set_base3_bit(coloring, i, GRAY);
        }
        color_ids.push_back(coloring);
    }

    SerialSolver solver{wordle.vocab.size, 5, wordle.dictionary.potential_words};
    vector<pair<float, vector<int>>> res = solver.solve_fulldist(guesses_ids, color_ids);
    sort(res.begin(), res.end());

    cout << "DISTRIBUTION:" << endl;
    for (int i = res.size() - 1; i >= 0; i--)
        cout << res[i].first << " " << wordle.decode_word(res[i].second) << endl;
}