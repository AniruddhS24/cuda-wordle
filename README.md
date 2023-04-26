# cuda-wordle

A CUDA-optimized Wordle implementation.

## Setup

After cloning the repo, run:

- CPU: `make build_cpu`
- GPU: `make build_gpu`

To run the command-line solver, simply run the executable like this:

- On CPU:
  `./cpu_solver -d ./basic_dictionary/potential_words.txt -v ./basic_dictionary/vocab.txt`
- On GPU:
  `./gpu_solver -d ./basic_dictionary/potential_words.txt -v ./basic_dictionary/vocab.txt`

## Interactive UI

To run the interactive Wordle UI locally, run the following commands:

```
make build_interactive
python3 app.py
```

The interface should be running on localhost.
