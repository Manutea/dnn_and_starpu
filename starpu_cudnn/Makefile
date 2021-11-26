CFLAGS  += $(shell pkg-config --cflags starpu-1.3) -I${CUDNN_INSTALL_PATH}/include
LDLIBS  += $(shell pkg-config --libs starpu-1.3)
CC      = gcc

cudnn_test: main.o
	gcc main.o -L${STARPU_LIB} -L${HWLOC_HOME}/lib -lstdc++ -lstarpu-1.3 -lpthread -lhwloc -lcudart -lcublas -lcudnn -lm -o cudnn_test

main.o: src/main.cpp src/starpu_dnn.hpp src/model_load.hpp include/cifar-10/cifar10_reader.hpp include/mnist-fashion/mnist_reader.hpp
	nvcc $(CFLAGS) $< -c $@  

# gcc main.o -L${STARPU_LIB} -L${HWLOC_HOME}/lib -lstdc++ -lstarpu-1.3 -lpthread -lhwloc -lcudart -lcublas -lcudnn -lm -o cudnn_test
