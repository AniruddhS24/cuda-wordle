run: build
	./solver -d ./basic_dictionary/potential_words.txt -v ./basic_dictionary/vocab.txt

build: build_solver link clean

build_solver:
	nvcc src/main.cpp -o main.o -c
	nvcc src/run_args.cpp -o run_args.o -c
	nvcc src/wordle.cpp -o wordle.o -c
	nvcc src/host/solver.cu -o solver.o -c

link:
	nvcc main.o run_args.o wordle.o solver.o -o solver

clean:
	rm *.o