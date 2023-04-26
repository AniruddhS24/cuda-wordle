#ifndef RUN_ARGS_H
#define RUN_ARGS_H

#include <string>

struct Arguments
{
    std::string vocab_filepath;
    std::string dictionary_filepath;
    bool suppress_output;
    bool interactive;
    std::string guesses; // used for interactive mode
    std::string colors;  // used for interactive mode
};

Arguments parse_arguments(int argc, char **argv);

#endif