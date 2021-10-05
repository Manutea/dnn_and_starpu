#include <starpu.h>
#include <math.h>

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

extern void dev_const(void *descr[], void *_args);
extern void dev_iota(void *descr[], void *_args);

typedef void (*device_func)(void **, void *);

int execute_on(uint32_t where, device_func func, float *matrix_data, int pnx, int pny)
{
  struct starpu_codelet cl;
  starpu_data_handle_t matrix_data_handle;

  starpu_matrix_data_register(&matrix_data_handle, STARPU_MAIN_RAM, (uintptr_t)matrix_data, pnx, pnx, pny, sizeof(float));

  starpu_codelet_init(&cl);
  cl.where = where;
  cl.cuda_funcs[0] = func;
  cl.nbuffers = 1;
  cl.modes[0] = STARPU_RW,
  cl.model = NULL;
  cl.name = "dev";

  struct starpu_task *task = starpu_task_create();
  task->cl = &cl;
  task->callback_func = NULL;
  task->handles[0] = matrix_data_handle;

  int ret = starpu_task_submit(task);
  if (STARPU_UNLIKELY(ret == -ENODEV))
  {
    FPRINTF(stderr, "No worker may execute this task\n");
    task->destroy = 0;
    starpu_task_destroy(task);
    return 1;
  }

  starpu_task_wait_for_all();
  starpu_data_unregister(matrix_data_handle);

  for(int i=0; i<pnx*pny; i++)
  {
    FPRINTF(stderr, "%f ", matrix_data[i]);
  }
  FPRINTF(stderr, "\n\n");


  return 0;
}


int main(void)
{
  const int filt_k = 1, filt_c = 1, filt_h = 2, filt_w = 2;
  const int in_n = 1, in_c = 1, in_h = 5, in_w = 5;

  float *filt_data, *in_data;
  int ret;

  int cosnt_nx=filt_w * filt_h;
  int const_ny=filt_k * filt_c;

  int iota_nx = in_w * in_h;
  int iota_ny = in_n * in_c;

  //Init StarPu
  ret = starpu_init(NULL);
  if (ret == -ENODEV)
    return 77;
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

  filt_data = (float*)malloc(cosnt_nx*const_ny*sizeof(float));
  assert(filt_data);
  execute_on(STARPU_CUDA, dev_const, filt_data, cosnt_nx, const_ny);

  in_data = (float*)malloc(iota_nx * iota_ny * sizeof(float));
  assert(in_data);
  execute_on(STARPU_CUDA, dev_iota, in_data, iota_nx, iota_ny);

  free(in_data);
  free(filt_data);
  starpu_shutdown();

  return 0;
}