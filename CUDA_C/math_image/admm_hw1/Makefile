main: admm.o admm_tools.o
	nvcc -o main -lcuda -lcudart -L/opt/cuda/lib64 admm.o admm_tools.o `pkg-config --cflags --libs opencv`
admm.o: admm.cu admm_tools.cuh
	nvcc -g -G admm.cu -c admm.o -lcuda -lcudart
admm_tools.o: admm_tools.cu admm_tools.cuh
	nvcc -g -G admm_tools.cu -c admm_tools.o -lcuda -lcudart
clean:
	rm -f *.o
