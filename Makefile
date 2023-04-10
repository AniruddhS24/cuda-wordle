
build: build_solver build_kernel link clean

build_solver:
	nvcc src/main.cpp -o main.o -c
	nvcc src/run_args.cpp -o run_args.o -c
	nvcc src/wordle.cpp -o wordle.o -c
	nvcc src/util.cpp -o util.o -c
	nvcc src/host/solver.cu -o solver.o -c

build_kernel:
	nvcc src/device/solver_kernels.cu -o solver_kernels.o -c

run:
	./solver -d ./basic_dictionary/potential_words.txt -v ./basic_dictionary/vocab.txt
	
link:
	nvcc main.o run_args.o wordle.o util.o solver.o solver_kernels.o -o solver

clean:
	rm *.o