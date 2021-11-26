#include "starpu_dnn.hpp"
#include "../include/cifar-10/cifar10_reader.hpp"
#include "../include/mnist-fashion/mnist_reader.hpp"

void show_result(const tensor *);

int main(int argc, char **argv)
{
  if(argc != 5)
  {
    printf("\nshow_result repeat in_h in_w\n");
    return -1;
  }

  const int show = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  const int in_h = atoi(argv[3]);
  const int in_w = atoi(argv[4]);

  const int ret = starpu_init(NULL);
  if (ret == -ENODEV)
  {
    return 77;
  }

  const int in_n = 1, in_c = 1;
  const int in_size = in_n * in_c * in_h * in_w;          
  float *in_data;
  starpu_malloc((void**)&in_data, in_size * sizeof(*in_data));
  for(int i=0; i<in_size; i++) 
  {
    in_data[i] = i;
  }

  const int filt_k = 1, filt_c = 1, filt_h = 2, filt_w = 2;
  const int filt_size = filt_k * filt_c * filt_h * filt_w;  
  float *filt_data;
  starpu_malloc((void**)&filt_data, filt_size * sizeof(*filt_data));
  for(int i=0; i<filt_size; i++) 
  {
    filt_data[i] = 1.0f;
  }

  float *conv1_bias_data;
  starpu_malloc((void**)&conv1_bias_data, 1 * sizeof(*conv1_bias_data));
  conv1_bias_data[0] = 0.0f;

  float *conv2_bias_data;
  starpu_malloc((void**)&conv2_bias_data, 1 * sizeof(*conv2_bias_data));
  conv2_bias_data[0] = 0.0f;

  starpu_fxt_autostart_profiling(0);

  auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("data/mnist/");
  std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
  std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
  std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
  std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

  std::cout << "IDK = " << dataset.training_images.at(0).size() << std::endl; //28*28

  /* Enable profiling */
  starpu_profiling_status_set(STARPU_PROFILING_ENABLE);

  starpu_execute_on_each_worker(cudnn_init, cudnn, STARPU_CUDA);

  tensor *filter = init_tensor(filt_data, filt_k, filt_c, filt_h, filt_w);
  tensor *conv1_bias = init_tensor(conv1_bias_data, 1, 1, 1, 1);
  tensor *conv2_bias = init_tensor(conv2_bias_data, 1, 1, 1, 1);
  
  for(int i=0; i<repeat; i++)
  {
    tensor *in = init_tensor(in_data, in_n, in_c, in_h, in_w);

    tensor *out = submit_convolution2D_forward(1, 1, 1, 1, 1, 1, 1.0, 0.0, in, filter, conv1_bias);
    free_tensor(in, in_data);
    tensor *out2 = submit_convolution2D_forward(1, 1, 1, 1, 1, 1, 1.0, 0.0, out, filter, conv2_bias);
    free_tensor(out);
    tensor *out3 = submit_max_pooling2D_forward(3, 3, 0, 0, 1, 1, 1.0, 0.0, out2);
    free_tensor(out2);
    tensor *out4 = submit_relu_forward(1.0, 1.0, out3);
    free_tensor(out3);
    tensor *out5 = submit_softmax_forward(1.0, 1.0, out4);
    free_tensor(out4);

    if(show && i == repeat - 1)
    {
      show_result(out5);
    }

    free_tensor(out5);
  }

  free_tensor(filter, filt_data);
  free_tensor(conv1_bias, conv1_bias_data);
  free_tensor(conv2_bias, conv2_bias_data);

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
