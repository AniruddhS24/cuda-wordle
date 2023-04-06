#ifndef RUN_ARGS_H
#define RUN_ARGS_H

#include <string>

struct Arguments
{
    std::string vocab_filepath;
    std::string dictionary_filepath;
    bool suppress_output;
    // more flags and stuff for diff implementations
};

Arguments parse_arguments(int argc, char **argv);

#endif