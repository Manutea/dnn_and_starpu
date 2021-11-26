#ifndef MODEL_LOAD_HPP
#define MODEL_LOAD_HPP

#include <iostream>
#include <fstream>
#include <string>
#include "starpu_dnn.hpp"

tensor *load_tensor(const std::string path, int n, int c, int h, int w, float *data)
{
  float f;
  std::ifstream file(path, std::ios::binary);

  if(!file)
  {
    std::cout << "Error opening file : " << path << std::endl;
    return nullptr;
  }

  file.seekg (0, file.end);
  const int size = file.tellg()/sizeof(float);
  file.seekg (0, file.beg);

  starpu_malloc((void**)&data, size * sizeof(*data));

  int i=0;
  while (file.read(reinterpret_cast<char*>(&f), sizeof(float)))
    data[i++] = f;
   
  return init_tensor(data, n, c, h, w);
}

#endif
