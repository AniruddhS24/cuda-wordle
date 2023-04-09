#include "run_args.h"
#include "getopt.h"

using namespace std;

Arguments parse_arguments(int argc, char **argv)
{
    Arguments args = {};

    int long_option_index = 0;
    static struct option long_options[] = {
        {"suppress_output", no_argument, NULL, 0}};

    int opt = 0;
    while ((opt = getopt_long(argc, argv, "d:v:ig", long_options, &long_option_index)) != -1)
    {
        switch (opt)
        {
        case 0:
            args.suppress_output = true;
            break;
        case 'd':
          args.dictionary_filepath = optarg;
          break;
        case 'v':
          args.vocab_filepath = optarg;
          break;
        case 'i':
          args.interactive = true;
          break;
        case 'g':
          args.use_gpu = true;
          break;
        }

    }

    return args;
}