#include "starpu_dnn.hpp"
#include "model_load.hpp"
#include "../include/mnist-fashion/mnist_reader.hpp"

void show_result(const tensor *);

int main(int argc, char **argv)
{
  const int ret = starpu_init(NULL);
  if (ret == -ENODEV)
  {
    return 77;
  }
  starpu_fxt_autostart_profiling(0);

  //auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("data/mnist/");
  //std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
  //std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
  //std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
  //std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;
  //std::cout << "IDK = " << dataset.training_images.at(0).size() << std::endl; //28*28 

  //weight & bias linear ReLu 0
  float *wlinrelu0_data, *blinrelu0_data;
  tensor *wlinrelu0_tensor = load_tensor("data/model/0", 1, 1, 512, 784, wlinrelu0_data);
  tensor *blinrelu0_tensor = load_tensor("data/model/1", 1, 1, 512, 1, blinrelu0_data);

  //weight & bias linear ReLu 1
  float *wlinrelu1_data, *blinrelu1_data;                                                                                                                                                                   
  tensor *wlinrelu1_tensor = load_tensor("data/model/2", 1, 1, 512, 512, wlinrelu1_data);
  tensor *blinrelu1_tensor = load_tensor("data/model/3", 1, 1, 512, 1, blinrelu1_data);

  //weight & bias linear ReLu 2
  float *wlinrelu2_data, *blinrelu2_data;
  tensor *wlinrelu2_tensor = load_tensor("data/model/4", 1, 1, 10, 512, wlinrelu2_data);
  tensor *blinrelu2_tensor = load_tensor("data/model/5", 1, 1, 10, 1, blinrelu2_data);

  /* Enable profiling */
  starpu_profiling_status_set(STARPU_PROFILING_ENABLE);

  starpu_execute_on_each_worker(cudnn_init, cudnn, STARPU_CUDA);

  // Linear(28*28, 512),
  //ReLU(),
  //Linear(512, 512),
  //ReLU(),
  //Linear(512, 10)

  free_tensor(wlinrelu0_tensor);
  free_tensor(blinrelu0_tensor);
  free_tensor(wlinrelu1_tensor);
  free_tensor(blinrelu1_tensor);
  free_tensor(wlinrelu2_tensor);
  free_tensor(blinrelu2_tensor);

  cudnn_shutdown();
  starpu_cublas_shutdown();
  starpu_shutdown();

  return 0;
}

void show_result(const tensor *tensor)
{
  starpu_data_acquire(tensor->handle, STARPU_R);
  const float *data= (const float *)starpu_data_get_local_ptr(tensor->handle);
  for(int i=0; i<tensor->n * tensor->c * tensor->h * tensor->w; i++) 
  {
    printf("%f \n", data[i]);
  }
  starpu_data_release(tensor->handle);
}


//	// output = weights^T * input (without biases)
//	checkCublasErrors(
//		cublasSgemm(cuda_->cublas(),
//			CUBLAS_OP_T, CUBLAS_OP_N, 
//			output_size_, batch_size_, input_size_,
//			&cuda_->one,  
//			weights_->cuda(), input_size_, 
//			input_->cuda(), input_size_,
//			&cuda_->zero, 
//			output_->cuda(),  output_size_));
//
//	// output += biases * d_one_vec^T
//	checkCublasErrors(cublasSgemm(cuda_->cublas(),
//					CUBLAS_OP_N, CUBLAS_OP_N, 
//					output_size_, batch_size_, 1,
//					&cuda_->one, 
//					biases_->cuda(), output_size_, 
//					d_one_vec, 1, 
//					&cuda_->one, 
//					output_->cuda(), output_size_));
