#include "../include/starpu_dnn/starpu_dnn.hpp"
#include "../include/starpu_dnn/model_load.hpp"
#include "../include/mnist-fashion/mnist_reader.hpp"

#define HEIGHT 28
#define WIDTH 28

int main(int argc, char **argv)
{
  if(argc < 2)
  {
    std::cout << "[batch size]";
    return 0;
  }
  const int batch = atoi(argv[1]);
  const auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("data/fashion/");
  
  const int ret = starpu_dnn_init();
  if (ret == -ENODEV)
  {
    return 77;
  }
  starpu_profiling_status_set(STARPU_PROFILING_ENABLE);

  //Image inference
  std::vector<float> whoaim_vec(HEIGHT * WIDTH * batch);
  for(int idimage = 0; idimage < batch; idimage++)
  {
    for(int i=0; i<WIDTH*HEIGHT; i++)
    {
      float f =  (float)unsigned(dataset.test_images.at(idimage).at(i));
      //Because Pytorch Fashion Model scale the pixels values [0, 255] to [0.0, 1.0]
      whoaim_vec[i + WIDTH*HEIGHT*idimage] = (f/255.0);
    }
  }
  tensor *whoaim = init_tensor(whoaim_vec.data(), batch, 1, HEIGHT, WIDTH);

  //weight & bias linear ReLu 0
  float *wlinrelu0_data, *blinrelu0_data;
  tensor *wlinrelu0_tensor = load_tensor("data/model/0", 1, 1, 512, HEIGHT*WIDTH, wlinrelu0_data);
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

  for(int ibatch=0; ibatch<batch; ibatch++)
  {
    int chosen = ibatch*10;
    //std::cout << "----"<< ibatch <<"-----\n" << std::endl; 
    //std::cout << "[" << 0 << "] = " << scores[ibatch*10] << "\n";
    for(int i=ibatch*10+1; i<10+(ibatch*10); i++)
    {
      std::cout << "[" << i - (ibatch * 10) << "] = " << scores[i] << "\n";
      if(scores[chosen] < scores[i])
      {
	chosen = i;
      }
    }
    ///std::cout << "\nImage label to guess is : " << unsigned(dataset.test_labels.at(ibatch)) << std::endl;  
    // std::cout << "The found label is : " << chosen - (ibatch * 10)<< "\n" << std::endl;
  }

  free_tensor(out5);

  starpu_dnn_shutdown();
  return 0;
}
