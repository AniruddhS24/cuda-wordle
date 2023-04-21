# cuda-wordle

To build the program run `make build` to build the solver

To run basic test run `run_gpu` or `run_seq`

Flags:
  d: dictionary filepath
  v: vocab filepath
  i: interactive (wait for user inputto make guess)
  g: use gpu
  c: use thread per color implementation
  m: use shared memory implementation
  s: pass in random seed for target word selection
  t: pass if sentence level decoding is necessary
  o: large output for distribution information