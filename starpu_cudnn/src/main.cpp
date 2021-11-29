#include "starpu_dnn.hpp"
#include "model_load.hpp"
#include "../include/mnist-fashion/mnist_reader.hpp"

void show(const tensor *);

int main(int argc, char **argv)
{
  const int itest = atoi(argv[1]);;
  auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("data/fashion/");
  
  const int ret = starpu_dnn_init();
  if (ret == -ENODEV)
  {
    return 77;
  }
  starpu_profiling_status_set(STARPU_PROFILING_ENABLE);

  //Image inference
  float *whoaim_data;
  starpu_malloc((void **)&whoaim_data, 28*28*sizeof(float)); //TODO : Move starpu_malloc inside tensor_init()
  for(int i=0; i<28*28; i++)
  {
    float f =  (float)unsigned(dataset.test_images.at(itest).at(i));
    //Because Pytorch Fashion Model scale the pixels values [0, 255] to [0.0, 1.0]
    whoaim_data[i] = (1.0/255.0)*f;
  }
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

  //------ Dnn model -------
  //Linear(28*28, 512)
  tensor *out0 = submit_linear_forward(whoaim, wlinrelu0_tensor, blinrelu0_tensor);
  free_tensor(wlinrelu0_tensor);
  free_tensor(blinrelu0_tensor);
  free_tensor(whoaim);  
  tensor *out1 = submit_relu_forward(1.0f, 1.0f, out0);
  free_tensor(out0);
  
  //Linear(512, 512)
  tensor *out2 = submit_linear_forward(out1, wlinrelu1_tensor, blinrelu1_tensor);  
  free_tensor(wlinrelu1_tensor);
  free_tensor(blinrelu1_tensor);
  free_tensor(out1);
  tensor *out3 = submit_relu_forward(1.0f, 1.0f, out2);
  free_tensor(out2);                                                                                                                                                                                           
  
  //Linear(512, 10)
  tensor *out4 = submit_linear_forward(out3, wlinrelu2_tensor, blinrelu2_tensor); 
  free_tensor(wlinrelu2_tensor);
  free_tensor(blinrelu2_tensor);
  free_tensor(out3);
  tensor *out5 = submit_relu_forward(1.0f, 1.0f, out4);
  free_tensor(out4);

  starpu_data_acquire(out5->handle, STARPU_R);
  const float *scores= (const float *)starpu_data_get_local_ptr(out5->handle);

  // Determine classification according to maximal response
  int chosen = 0;
  std::cout<<"\nlabel["<<chosen<<"] rating is : "<<scores[chosen]<<"\n";
  for (int id = 1; id < 10; ++id)
  {
    std::cout<<"label["<<id<<"] rating is : "<<scores[id]<<"\n";
    if(scores[chosen] < scores[id]) 
      chosen = id;
  }

  std::cout << "\nImage label to guess is : " << unsigned(dataset.test_labels.at(itest)) << std::endl;
  std::cout << "The found label is : " << chosen << "\n" << std::endl;

  free_tensor(out5);
  starpu_dnn_shutdown();
  return 0;
}

void show(const tensor *tensor)
{
  starpu_data_acquire(tensor->handle, STARPU_R);
  const float *data= (const float *)starpu_data_get_local_ptr(tensor->handle);
  for(int i=0; i<tensor->n * tensor->c * tensor->h * tensor->w; i++) 
  {
    printf("%f \n", data[i]);
  }
  starpu_data_release(tensor->handle);
}
