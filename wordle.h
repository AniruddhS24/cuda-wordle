#include <string>
#include <vector>
#define DEBUG true
using namespace std;


const string GUESSES = "./basic_dictionary/guesses.txt";
const string POTENTIAL_WORDS = "./basic_dictionary/potential_words.txt";
enum COLORS {
  GREEN = 'g',
  YELLOW = 'y',
  GRAY = 'r'
};

vector<string> guesses;
vector<string> potential_words;
vector<string> dictionary;

void load_dictionary();
string post_guess(string guess, string word);