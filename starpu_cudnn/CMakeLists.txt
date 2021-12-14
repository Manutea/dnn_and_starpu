project("HelloWorld")
cmake_minimum_required(VERSION "3.15.1")

#===========================================================================
# Project Declaration
#===========================================================================
list( APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/morse_cmake/modules")
include(MorseInit)

# Local header files here ONLY
SET(TARGET_H
    "include/starpu_dnn/model_load.hpp"
    "include/starpu_dnn/starpu_dnn.hpp"
    "include/mnist-fashion/mnist_reader.hpp"
    "include/mnist-fashion/mnist_reader_common.hpp"
    "include/cifar-10/cifar10_reader.hpp"
)

# Local source files here
SET(TARGET_SRC
    "src/main.cpp"
    "src/starpu_dnn.cpp"
)


add_executable(${PROJECT_NAME} ${TARGET_H} ${TARGET_SRC})

#===========================================================================
# StarPu
#===========================================================================
set(STARPU_VERSION "1.3" CACHE STRING "oldest STARPU version desired")
set(STARPU_COMPONENT_LIST "CUDA")

find_package(STARPU ${STARPU_VERSION} REQUIRED COMPONENTS ${STARPU_COMPONENT_LIST})

if(STARPU_FOUND)
	include_directories(${STARPU_INCLUDE_DIRS})
	target_link_libraries(${PROJECT_NAME} ${STARPU_LIBRARIES})
	message(STATUS "------ ${STARPU_LIBRARIES}")
endif(STARPU_FOUND)

#===========================================================================
# CUDNN
#===========================================================================
FIND_PACKAGE(CUDNN REQUIRED)

if(CUDNN_FOUND)
        include_directories(${CUDNN_INCLUDE_DIRS})
        target_link_libraries(${PROJECT_NAME} ${CUDNN_LIBRARIES})
	target_link_libraries(${PROJECT_NAME} -lcudart -lcublas)
endif(CUDNN_FOUND)
