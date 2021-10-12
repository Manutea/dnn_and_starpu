#include <starpu.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define INSZ  25
#define FILTSZ  4

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

extern void init_conv_cudnn_func(void *buffers[], void *_args);

static struct starpu_perfmodel vector_scal_model =
{
  .type = STARPU_HISTORY_BASED,
  .symbol = "vector_scal"
};

static struct starpu_codelet cl =
{
  .cuda_funcs = {init_conv_cudnn_func},
  .cuda_flags = {STARPU_CUDA_ASYNC},
  .nbuffers = 3,
  .modes = {STARPU_R, STARPU_R, STARPU_RW},
  .model = &vector_scal_model,
};

struct cudnn_convolution_params
{
  // input
  int in_n, in_c, in_h, in_w;
  // filter
  int filt_k, filt_c, filt_h, filt_w;
  // convolution
  int pad_h, pad_w, str_h, str_w, dil_h, dil_w;
  // out
  int out_n, out_c, out_h, out_w;
  // workspace size
  size_t ws_size;
};

int main(void)
{                                                 
  struct cudnn_convolution_params conv_params = {1, 1, 5, 5,       // input
                                                 1, 1, 2, 2,       // filter
                                                 1, 1, 1, 1, 1, 1, // convolution
                                                 0, 0, 0, 0,       // out
                                                 0};               // workspace size
  int out_n, out_c, out_h, out_w;

  float in_data[INSZ];
  for(int i=0; i<INSZ; i++) {
    in_data[i] = i;
  }

  float filter_data[FILTSZ];
  for(int i=0; i<FILTSZ; i++) {
    filter_data[i] = 1.0f;
  }

  float out_data[1*1*6*6];

  int ret = starpu_init(NULL);
  if (ret == -ENODEV)
  {
    return 77;
  }

  //starpu data register
  starpu_data_handle_t in_data_handle;
  starpu_memory_pin(in_data, sizeof(in_data));
  starpu_vector_data_register(&in_data_handle, STARPU_MAIN_RAM, (uintptr_t)in_data, INSZ, sizeof(in_data[0]));

  starpu_data_handle_t filter_handle;
  starpu_memory_pin(filter_data, sizeof(filter_data));
  starpu_vector_data_register(&filter_handle, STARPU_MAIN_RAM, (uintptr_t)filter_data, FILTSZ, sizeof(filter_data[0]));

  starpu_data_handle_t out_data_handle;
  starpu_memory_pin(out_data, sizeof(out_data));
  starpu_vector_data_register(&out_data_handle, STARPU_MAIN_RAM, (uintptr_t)out_data, 1*1*6*6, sizeof(out_data[0]));

  //
  struct starpu_task *task = starpu_task_create();
  task->synchronous = 1;
  task->cl = &cl;
  task->handles[0] = in_data_handle;
  task->handles[1] = filter_handle;
  task->handles[2] = out_data_handle;
  task->cl_arg = &conv_params;
  task->cl_arg_size = sizeof(conv_params);

  ret = starpu_task_submit(task);
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

  //
  starpu_data_unregister(in_data_handle);
  starpu_data_unregister(filter_handle);
  starpu_data_unregister(out_data_handle);

  starpu_memory_unpin(in_data, sizeof(in_data));
  starpu_memory_unpin(filter_data, sizeof(filter_data));
  starpu_memory_unpin(out_data, sizeof(out_data));

  starpu_shutdown();

  printf("\n");
  printf("out_n : %d\n", conv_params.out_n);
  printf("out_c : %d\n", conv_params.out_c);
  printf("out_h : %d\n", conv_params.out_h);
  printf("out_w : %d\n", conv_params.out_w);
  printf("\n");

  printf("\n");
  for(int i=0; i<1*1*6*6; i++) 
  {
    printf("%f \n", out_data[i]);
  }

  return 0;
}



  // starpu_cuda_get_local_stream(), cudaStreamSynchronize(starpu_cuda_get_local_stream());
  // multiformat data register ?
