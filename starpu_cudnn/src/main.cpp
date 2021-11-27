#include "starpu_dnn.hpp"
#include "model_load.hpp"
#include "../include/mnist-fashion/mnist_reader.hpp"

void show_result(const tensor *);

int main(int argc, char **argv)
{
  const int ret = starpu_dnn_init();
  if (ret == -ENODEV)
  {
    return 77;
  }
  starpu_profiling_status_set(STARPU_PROFILING_ENABLE);

  auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("data/mnist/");
  std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
  std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
  std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
  std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;
  std::cout << "Pixel 0 = " << unsigned(dataset.training_images.at(0).at(0)) << std::endl;

  //Image inference
  std::vector<float> image(dataset.training_images.at(0).begin(), dataset.training_images.at(0).end()); //Maybe not necessary ? doesn't need an explicit cast for an uint8_t
  float *whoaim_data = image.data();
  starpu_malloc((void **)&whoaim_data, 28*28*sizeof(float)); //TODO : Move starpu_malloc inside tensor_init()
  //for(int i=0; i<28*28; i++)
  //  whoaim_data[i] = (float)unsigned(dataset.training_images.at(0).at(i));
  tensor *whoaim = init_tensor(whoaim_data, 1, 1, 28, 28);

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

  //Linear(28*28, 512),
  //ReLU(),
  //Linear(512, 512),
  //ReLU(),
  //Linear(512, 10)

  free_tensor(whoaim);
  free_tensor(wlinrelu0_tensor);
  free_tensor(blinrelu0_tensor);
  free_tensor(wlinrelu1_tensor);
  free_tensor(blinrelu1_tensor);
  free_tensor(wlinrelu2_tensor);
  free_tensor(blinrelu2_tensor);

  starpu_dnn_shutdown();

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
