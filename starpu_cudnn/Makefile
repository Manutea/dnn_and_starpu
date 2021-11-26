CFLAGS  += $(shell pkg-config --cflags starpu-1.3) -I/cm/shared/dev/modules/generic/apps/dnn/cudnn/11.2-v8.1.1.33/cuda/include                                                                             
LDLIBS  += $(shell pkg-config --libs starpu-1.3)                                                                                                                                                           
                                                                                                                                                                                                           
LIBS    = -lstdc++ -lstarpu-1.3 -lpthread -lhwloc -lcudart -lcublas -lcudnn -lm                                                                                                                            
CC      = gcc                                                                                                                                                                                              
                                                                                                                                                                                                           
main.o: src/main.cpp src/starpu_dnn.hpp include/cifar-10/cifar10_reader.hpp include/mnist-fashion/mnist_reader.hpp                                                                                         
        nvcc $(CFLAGS) $< -c $@  

# gcc main.o -L/cm/shared/dev/modules/generic/apps/runtime/starpu/1.3.8/gcc@9.2.0-hwloc@2.4.0-openmpi@4.0.3-cuda@11.2-fxt@0.3.13/lib -L/cm/shared/dev/modules/generic/apps/hardware/hwloc/hwloc-2.5.0/install/lib -lstdc++ -lstarpu-1.3 -lpthread -lhwloc -lcudart -lcublas -lcudnn -lm -o cudnn_test