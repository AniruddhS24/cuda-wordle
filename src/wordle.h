#ifndef WORDLE_H
#define WORDLE_H

#include <string>
#include <vector>
#include <map>

enum COLORS
{
    GREEN = 1,
    YELLOW = 2,
    GRAY = 0
};

struct Vocabulary
{
    int size;
    std::map<std::string, int> word_to_id;
    std::map<int, std::string> id_to_word;
};

struct Dictionary
{
    std::string word_file;
    std::vector<std::string> potential_words_str;
    std::vector<std::vector<int>> potential_words;
};

struct GameState
{
    std::vector<std::vector<int>> guesses;
    std::vector<int> colors;
};

class Wordle
{
public:
    std::string vocabulary_file;
    std::string word_file;
    std::vector<std::string> (*tokenizer)(std::string);
    Vocabulary vocab;
    Dictionary dictionary;
    std::vector<int> target_word;
    GameState state;

    Wordle(std::string vocabulary_file, std::string word_file, std::vector<std::string> (*tokenizer)(std::string)) : vocabulary_file(vocabulary_file), word_file(word_file), tokenizer(tokenizer){};

    void load_vocabulary();
    void load_dictionary();
    static int set_coloring_bit(int coloring, int pos, int value);
    static int get_coloring_bit(int coloring, int pos);
    static int generate_coloring(std::vector<int> word, std::vector<int> guess);
    static void print_coloring(int coloring);
    std::vector<int> encode_word(std::string word);
    std::string decode_word(std::vector<int> ids);

    void set_target_word();
    std::vector<int> get_target_word();
    int post_guess(std::string guess);
};

std::vector<std::string> word_tokenizer(std::string input);

#endif