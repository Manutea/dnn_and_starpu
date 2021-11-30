#ifndef STARPU_DNN_HPP
#define STARPU_DNN_HPP

#include <starpu.h>

struct _tensor 
{
  float *data;
  starpu_data_handle_t handle;
  int n, c, h, w;
};
typedef struct _tensor tensor;

int starpu_dnn_init();
void starpu_dnn_shutdown();

tensor *init_tensor(float *, int, int, int, int);
void free_tensor(tensor *);

void convolution2D_forward(void **, void *);
tensor *submit_convolution2D_forward(int, int, int, int, int, int, float, float, const tensor *, const tensor *, const tensor *);

void pooling2D_forward(void **, void *);
tensor *submit_max_pooling2D_forward(int, int, int, int, int, int, float, float, const tensor *);

void relu_forward(void **, void *);
tensor *submit_relu_forward(float, float, const tensor *);

void softmax_forward(void **, void *);
tensor *submit_softmax_forward(float, float, const tensor *);

void fullyco_forward(void **, void *);
tensor *submit_fullyco_forward(float, float, const tensor *, const tensor *);

void linear_forward(void **, void *);
tensor *submit_linear_forward(const tensor *, const tensor *, const tensor *, float alpha_w=1.0f, float beta_w=0.0f, float alpha_b=1.0f, float beta_b=1.0f);

#endif