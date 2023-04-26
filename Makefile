
build_cpu: build_solver_cpu link_cpu clean

build_gpu: build_solver_gpu build_kernel link_gpu clean

build_interactive: build_solver_interactive link_interactive clean

build_solver_interactive:
	g++ -std=c++11 src/interactive.cpp -o interactive.o -c
	g++ -std=c++11 src/run_args.cpp -o run_args.o -c
	g++ -std=c++11 src/wordle.cpp -o wordle.o -c
	g++ -std=c++11 src/util.cpp -o util.o -c
	g++ -std=c++11 src/serial_solver.cpp -o serial_solver.o -c

link_interactive:
	g++ interactive.o run_args.o wordle.o util.o serial_solver.o -o interactive_solver

build_solver_cpu: 
	g++ -std=c++11 src/cpu.cpp -o cpu.o -c
	g++ -std=c++11 src/run_args.cpp -o run_args.o -c
	g++ -std=c++11 src/wordle.cpp -o wordle.o -c
	g++ -std=c++11 src/util.cpp -o util.o -c
	g++ -std=c++11 src/serial_solver.cpp -o serial_solver.o -c

link_cpu: 
	g++ cpu.o run_args.o wordle.o util.o serial_solver.o -o cpu_solver

build_solver_gpu:
	nvcc src/gpu.cpp -o gpu.o -c
	nvcc src/run_args.cpp -o run_args.o -c
	nvcc src/wordle.cpp -o wordle.o -c
	nvcc src/util.cpp -o util.o -c
	nvcc src/host/solver.cu -o solver.o -c

build_kernel:
	nvcc src/device/solver_kernels.cu -o solver_kernels.o -c

link_gpu:
	nvcc gpu.o run_args.o wordle.o util.o solver.o solver_kernels.o -o gpu_solver

run_gpu:
	./gpu_solver -d ./basic_dictionary/potential_words.txt -v ./basic_dictionary/vocab.txt

run_cpu:
	./cpu_solver -d ./basic_dictionary/potential_words.txt -v ./basic_dictionary/vocab.txt
	
clean:
	rm *.o 