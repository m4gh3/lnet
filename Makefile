CFLAGS=-g -O0
CUDA_CFLAGS=-lcuda -lcudart -L/usr/local/cuda/lib -fpermissive
OUTDIR=.

example_training_cuda: examples/example_training_cuda.cu build/mmatrix_cuda.o build/arrays_cuda.o
	nvcc $(CFLAGS) -G -c `projcfg -i` examples/example_training_cuda.cu -o build/example_training_cuda.o
	g++ $(CFLAGS) $(CUDA_CFLAGS) `projcfg -l` -lm -lpng -lpngdraw build/example_training_cuda.o build/mmatrix_cuda.o build/arrays_cuda.o -o $(OUTDIR)/example_training_cuda

example_training: examples/example_training.c build/mmatrix.o build/arrays.o build/processing.o build/lnetlayer.o
	gcc $(CFLAGS) `projcfg -li` -lm -lpng -lpngdraw examples/example_training.c build/mmatrix.o build/arrays.o build/processing.o build/lnetlayer.o -o $(OUTDIR)/example_training

example_small_training: examples/example_small_training.c build/mmatrix.o build/arrays.o
	gcc $(CFLAGS) -lm examples/example_small_training.c build/mmatrix.o build/arrays.o -o $(OUTDIR)/example_small_training

example_xor: examples/example_xor.c build/mmatrix.o build/arrays.o
	gcc $(CLFAGS) -lm examples/example_xor.c build/mmatrix.o build/arrays.o -o $(OUTDIR)/example_xor

build/mmatrix.o: src/mmatrix.c include/mmatrix.h
	gcc $(CFLAGS) -c src/mmatrix.c -o build/mmatrix.o

build/arrays.o: src/arrays.c include/arrays.h
	gcc $(CFLAGS) -c src/arrays.c -o build/arrays.o

build/processing.o: src/processing.c include/processing.h
	gcc $(CFLAGS) -c src/processing.c -o build/processing.o

build/lnetlayer.o: src/lnetlayer.c include/lnetlayer.h
	gcc $(CFLAGS) -c src/lnetlayer.c -o build/lnetlayer.o

build/mmatrix_cuda.o: src/mmatrix.cu include/mmatrix.h
	nvcc $(CFLAGS) -G -c src/mmatrix.cu -o build/mmatrix_cuda.o

build/arrays_cuda.o: src/arrays.cu include/arrays.h
	nvcc $(CFLAGS) -G -c src/arrays.cu -o build/arrays_cuda.o
