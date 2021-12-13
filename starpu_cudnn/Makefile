CFLAGS  += $(shell pkg-config --cflags starpu-1.3) -I${CUDNN_INSTALL_PATH}/include -I${MKLROOT}/include
LDLIBS  += $(shell pkg-config --libs starpu-1.3)
CC      = gcc

cudnn_test: main.o starpu_dnn.o
	gcc main.o starpu_dnn.o -L${STARPU_LIB} -L${HWLOC_HOME}/lib -lstdc++ -lstarpu-1.3 -lpthread -lhwloc -lcudart -lcublas -lcudnn -lm -o cudnn_test

starpu_dnn.o: src/starpu_dnn.cpp include/starpu_dnn/starpu_dnn.hpp
	nvcc $(CFLAGS) $< -c $@

main.o: src/main.cpp include/starpu_dnn/starpu_dnn.hpp include/starpu_dnn/model_load.hpp include/cifar-10/cifar10_reader.hpp include/mnist-fashion/mnist_reader.hpp
	nvcc $(CFLAGS) $< -c $@
