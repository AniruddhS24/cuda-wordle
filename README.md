# cuda-wordle

A CUDA-optimized Wordle implementation.

[Demo](https://cuda-wordle-app.herokuapp.com/) | [Paper](https://github.com/AniruddhS24/cuda-wordle/blob/main/CUDA_Wordle.pdf)

We also built a [Chrome extension](https://chrome.google.com/webstore/detail/wordle-solver/oonedgenifopijmhkkjncgmkeahoajhk) to use on the Wordle website.

## Setup

### Build

After cloning the repo, run:

- CPU: `make build_cpu`
- GPU: `make build_gpu`

### Run

To run the command-line solver, simply run the executable like this:

- CPU:
  `./cpu_solver -d ./basic_dictionary/potential_words.txt -v ./basic_dictionary/vocab.txt`
- GPU:
  `./gpu_solver -d ./basic_dictionary/potential_words.txt -v ./basic_dictionary/vocab.txt`

## Interactive UI

To run the interactive Wordle UI locally, run the following commands:

```
make build_interactive
python3 app.py
```

The interface should be running on localhost.

Credits: The Wordle interface was built using https://github.com/Morgenstern2573/wordle_clone as a template.
