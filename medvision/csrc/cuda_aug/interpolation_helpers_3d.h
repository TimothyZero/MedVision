#include <float.h>

#include "cuda_helpers.h"

template <typename T>
__device__ T linear_3d_interpolate(
  const T* input,
  const int depth, const int height, const int width,
  T z, T y, T x,
  const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width || z < -1.0 || z > depth) return 0;

  if (z <= 0) z = 0;
  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int z0 = (int)z;
  int y0 = (int)y;
  int x0 = (int)x;
  int z1;
  int y1;
  int x1;

  if (z0 >= depth - 1) {
    z1 = z0 = depth - 1;
    z = (T)z0;
  } else {
    z1 = z0 + 1;
  }

  if (y0 >= height - 1) {
    y1 = y0 = height - 1;
    y = (T)y0;
  } else {
    y1 = y0 + 1;
  }

  if (x0 >= width - 1) {
    x1 = x0 = width - 1;
    x = (T)x0;
  } else {
    x1 = x0 + 1;
  }

  T lz = z - z0;
  T ly = y - y0;
  T lx = x - x0;
  T hz = 1. - lz;
  T hy = 1. - ly;
  T hx = 1. - lx;
  // do linear 3d interpolation
  T v1 = input[z0 * height * width + y0 * width + x0];
  T v2 = input[z0 * height * width + y0 * width + x1];
  T v3 = input[z0 * height * width + y1 * width + x0];
  T v4 = input[z0 * height * width + y1 * width + x1];
  T v5 = input[z1 * height * width + y0 * width + x0];
  T v6 = input[z1 * height * width + y0 * width + x1];
  T v7 = input[z1 * height * width + y1 * width + x0];
  T v8 = input[z1 * height * width + y1 * width + x1];

  T w1 = hz * hy * hx, w2 = hz * hy * lx, w3 = hz * ly * hx, w4 = hz * ly * lx;
  T w5 = lz * hy * hx, w6 = lz * hy * lx, w7 = lz * ly * hx, w8 = lz * ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4 + w5 * v5 + w6 * v6 + w7 * v7 + w8 * v8);

  return val;
}

template <typename T>
__device__ void linear_3d_interpolate_gradient(
  const int depth, const int height, const int width,
  T z, T y, T x,
  T& w1, T& w2, T& w3, T& w4, T& w5, T& w6, T& w7, T& w8,
  int& x0, int& x1,
  int& y0, int& y1,
  int& z0, int& z1,
  const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width || z < -1.0 || z > depth) {
    // empty
    w1 = w2 = w3 = w4 = w5 = w6 = w7 = w8 = 0.;
    x0 = x1 = y0 = y1 = z0 = z1 = -1;
    return;
  }

  if (z <= 0) z = 0;
  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  z0 = (int)z;
  y0 = (int)y;
  x0 = (int)x;

  if (z0 >= depth - 1) {
    z1 = z0 = depth - 1;
    z = (T)z0;
  } else {
    z1 = z0 + 1;
  }

  if (y0 >= height - 1) {
    y1 = y0 = height - 1;
    y = (T)y0;
  } else {
    y1 = y0 + 1;
  }

  if (x0 >= width - 1) {
    x1 = x0 = width - 1;
    x = (T)x0;
  } else {
    x1 = x0 + 1;
  }

  T lz = z - z0;
  T ly = y - y0;
  T lx = x - x0;
  T hz = 1. - lz;
  T hy = 1. - ly;
  T hx = 1. - lx;

  w1 = hz * hy * hx, w2 = hz * hy * lx, w3 = hz * ly * hx, w4 = hz * ly * lx;
  w5 = lz * hy * hx, w6 = lz * hy * lx, w7 = lz * ly * hx, w8 = lz * ly * lx;

  return;
}

template <typename T>
__device__ T nearest_3d_interpolate(
  const T* input,
  const int depth, const int height, const int width,
  T z, T y, T x,
  const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width || z < -1.0 || z > depth) return 0;

  if (z <= 0) z = 0;
  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int z0 = (int)z;
  int y0 = (int)y;
  int x0 = (int)x;
  int z1;
  int y1;
  int x1;

  if (z0 >= depth - 1) {
    z1 = z0 = depth - 1;
    z = (T)z0;
  } else {
    z1 = z0 + 1;
  }

  if (y0 >= height - 1) {
    y1 = y0 = height - 1;
    y = (T)y0;
  } else {
    y1 = y0 + 1;
  }

  if (x0 >= width - 1) {
    x1 = x0 = width - 1;
    x = (T)x0;
  } else {
    x1 = x0 + 1;
  }

  T lz = z - z0;
  T ly = y - y0;
  T lx = x - x0;
  T hz = 1. - lz;
  T hy = 1. - ly;
  T hx = 1. - lx;

  T v1 = input[z0 * height * width + y0 * width + x0];
  T v2 = input[z0 * height * width + y0 * width + x1];
  T v3 = input[z0 * height * width + y1 * width + x0];
  T v4 = input[z0 * height * width + y1 * width + x1];
  T v5 = input[z1 * height * width + y0 * width + x0];
  T v6 = input[z1 * height * width + y0 * width + x1];
  T v7 = input[z1 * height * width + y1 * width + x0];
  T v8 = input[z1 * height * width + y1 * width + x1];

  T w1 = 0, w2 = 0, w3 = 0, w4 = 0, w5 =0, w6 =0, w7 = 0, w8 = 0;
  if      (hz >= 0.5 && hy >= 0.5 && hx >= 0.5) w1 = 1.0;
  else if (hz >= 0.5 && hy >= 0.5 && lx >= 0.5) w2 = 1.0;
  else if (hz >= 0.5 && ly >= 0.5 && hx >= 0.5) w3 = 1.0;
  else if (hz >= 0.5 && ly >= 0.5 && lx >= 0.5) w4 = 1.0;
  else if (lz >= 0.5 && hy >= 0.5 && hx >= 0.5) w5 = 1.0;
  else if (lz >= 0.5 && hy >= 0.5 && lx >= 0.5) w6 = 1.0;
  else if (lz >= 0.5 && ly >= 0.5 && hx >= 0.5) w7 = 1.0;
  else if (lz >= 0.5 && ly >= 0.5 && lx >= 0.5) w8 = 1.0;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4 + w5 * v5 + w6 * v6 + w7 * v7 + w8 * v8);

  return val;
}

template <typename T>
__device__ void nearest_3d_interpolate_gradient(
  const int depth, const int height, const int width,
  T z, T y, T x,
  T& w1, T& w2, T& w3, T& w4, T& w5, T& w6, T& w7, T& w8,
  int& x0, int& x1,
  int& y0, int& y1,
  int& z0, int& z1,
  const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width || z < -1.0 || z > depth) {
    // empty
    w1 = w2 = w3 = w4 = w5 = w6 = w7 = w8 = 0.;
    x0 = x1 = y0 = y1 = z0 = z1 = -1;
    return;
  }

  if (z <= 0) z = 0;
  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  z0 = (int)z;
  y0 = (int)y;
  x0 = (int)x;

  if (z0 >= depth - 1) {
    z1 = z0 = depth - 1;
    z = (T)z0;
  } else {
    z1 = z0 + 1;
  }

  if (y0 >= height - 1) {
    y1 = y0 = height - 1;
    y = (T)y0;
  } else {
    y1 = y0 + 1;
  }

  if (x0 >= width - 1) {
    x1 = x0 = width - 1;
    x = (T)x0;
  } else {
    x1 = x0 + 1;
  }

  T lz = z - z0;
  T ly = y - y0;
  T lx = x - x0;
  T hz = 1. - lz;
  T hy = 1. - ly;
  T hx = 1. - lx;

  w1 = 0, w2 = 0, w3 = 0, w4 = 0, w5 =0, w6 =0, w7 = 0, w8 = 0;
  if      (hz >= 0.5 && hy >= 0.5 && hx >= 0.5) w1 = 1.0;
  else if (hz >= 0.5 && hy >= 0.5 && lx >= 0.5) w2 = 1.0;
  else if (hz >= 0.5 && ly >= 0.5 && hx >= 0.5) w3 = 1.0;
  else if (hz >= 0.5 && ly >= 0.5 && lx >= 0.5) w4 = 1.0;
  else if (lz >= 0.5 && hy >= 0.5 && hx >= 0.5) w5 = 1.0;
  else if (lz >= 0.5 && hy >= 0.5 && lx >= 0.5) w6 = 1.0;
  else if (lz >= 0.5 && ly >= 0.5 && hx >= 0.5) w7 = 1.0;
  else if (lz >= 0.5 && ly >= 0.5 && lx >= 0.5) w8 = 1.0;

  return;
}


//__device__ double matrix_a[64][64] = {
//  {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {-3, 3, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {2, -2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {-3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {9, -9, -9, 9, 0, 0, 0, 0, 6, 3, -6, -3, 0, 0, 0, 0, 6, -6, 3, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {-6, 6, 6, -6, 0, 0, 0, 0, -3, -3, 3, 3, 0, 0, 0, 0, -4, 4, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -2, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {-6, 6, 6, -6, 0, 0, 0, 0, -4, -2, 4, 2, 0, 0, 0, 0, -3, 3, -3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {4, -4, -4, 4, 0, 0, 0, 0, 2, 2, -2, -2, 0, 0, 0, 0, 2, -2, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, -9, -9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 3, -6, -3, 0, 0, 0, 0, 6, -6, 3, -3, 0, 0, 0, 0, 4, 2, 2, 1, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 6, 6, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -3, 3, 3, 0, 0, 0, 0, -4, 4, -2, 2, 0, 0, 0, 0, -2, -2, -1, -1, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 6, 6, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -2, 4, 2, 0, 0, 0, 0, -3, 3, -3, 3, 0, 0, 0, 0, -2, -1, -2, -1, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, -4, -4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, -2, -2, 0, 0, 0, 0, 2, -2, 2, -2, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0},
//  {-3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {9, -9, 0, 0, -9, 9, 0, 0, 6, 3, 0, 0, -6, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, -6, 0, 0, 3, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {-6, 6, 0, 0, 6, -6, 0, 0, -3, -3, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 4, 0, 0, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -2, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, -9, 0, 0, -9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 3, 0, 0, -6, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, -6, 0, 0, 3, -3, 0, 0, 4, 2, 0, 0, 2, 1, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 6, 0, 0, 6, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -3, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 4, 0, 0, -2, 2, 0, 0, -2, -2, 0, 0, -1, -1, 0, 0},
//  {9, 0, -9, 0, -9, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 3, 0, -6, 0, -3, 0, 6, 0, -6, 0, 3, 0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 9, 0, -9, 0, -9, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 3, 0, -6, 0, -3, 0, 6, 0, -6, 0, 3, 0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 0, 2, 0, 1, 0},
//  {-27, 27, 27, -27, 27, -27, -27, 27, -18, -9, 18, 9, 18, 9, -18, -9, -18, 18, -9, 9, 18, -18, 9, -9, -18, 18, 18, -18, -9, 9, 9, -9, -12, -6, -6, -3, 12, 6, 6, 3, -12, -6, 12, 6, -6, -3, 6, 3, -12, 12, -6, 6, -6, 6, -3, 3, -8, -4, -4, -2, -4, -2, -2, -1},
//  {18, -18, -18, 18, -18, 18, 18, -18, 9, 9, -9, -9, -9, -9, 9, 9, 12, -12, 6, -6, -12, 12, -6, 6, 12, -12, -12, 12, 6, -6, -6, 6, 6, 6, 3, 3, -6, -6, -3, -3, 6, 6, -6, -6, 3, 3, -3, -3, 8, -8, 4, -4, 4, -4, 2, -2, 4, 4, 2, 2, 2, 2, 1, 1},
//  {-6, 0, 6, 0, 6, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, -3, 0, 3, 0, 3, 0, -4, 0, 4, 0, -2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -2, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 6, 0, 6, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, -3, 0, 3, 0, 3, 0, -4, 0, 4, 0, -2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -2, 0, -1, 0, -1, 0},
//  {18, -18, -18, 18, -18, 18, 18, -18, 12, 6, -12, -6, -12, -6, 12, 6, 9, -9, 9, -9, -9, 9, -9, 9, 12, -12, -12, 12, 6, -6, -6, 6, 6, 3, 6, 3, -6, -3, -6, -3, 8, 4, -8, -4, 4, 2, -4, -2, 6, -6, 6, -6, 3, -3, 3, -3, 4, 2, 4, 2, 2, 1, 2, 1},
//  {-12, 12, 12, -12, 12, -12, -12, 12, -6, -6, 6, 6, 6, 6, -6, -6, -6, 6, -6, 6, 6, -6, 6, -6, -8, 8, 8, -8, -4, 4, 4, -4, -3, -3, -3, -3, 3, 3, 3, 3, -4, -4, 4, 4, -2, -2, 2, 2, -4, 4, -4, 4, -2, 2, -2, 2, -2, -2, -2, -2, -1, -1, -1, -1},
//  {2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {-6, 6, 0, 0, 6, -6, 0, 0, -4, -2, 0, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {4, -4, 0, 0, -4, 4, 0, 0, 2, 2, 0, 0, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 6, 0, 0, 6, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -2, 0, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0, -2, -1, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, -4, 0, 0, -4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0},
//  {-6, 0, 6, 0, 6, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, -2, 0, 4, 0, 2, 0, -3, 0, 3, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 6, 0, 6, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, -2, 0, 4, 0, 2, 0, -3, 0, 3, 0, -3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, -2, 0, -1, 0},
//  {18, -18, -18, 18, -18, 18, 18, -18, 12, 6, -12, -6, -12, -6, 12, 6, 12, -12, 6, -6, -12, 12, -6, 6, 9, -9, -9, 9, 9, -9, -9, 9, 8, 4, 4, 2, -8, -4, -4, -2, 6, 3, -6, -3, 6, 3, -6, -3, 6, -6, 3, -3, 6, -6, 3, -3, 4, 2, 2, 1, 4, 2, 2, 1},
//  {-12, 12, 12, -12, 12, -12, -12, 12, -6, -6, 6, 6, 6, 6, -6, -6, -8, 8, -4, 4, 8, -8, 4, -4, -6, 6, 6, -6, -6, 6, 6, -6, -4, -4, -2, -2, 4, 4, 2, 2, -3, -3, 3, 3, -3, -3, 3, 3, -4, 4, -2, 2, -4, 4, -2, 2, -2, -2, -1, -1, -2, -2, -1, -1},
//  {4, 0, -4, 0, -4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, -2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 4, 0, -4, 0, -4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, -2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0},
//  {-12, 12, 12, -12, 12, -12, -12, 12, -8, -4, 8, 4, 8, 4, -8, -4, -6, 6, -6, 6, 6, -6, 6, -6, -6, 6, 6, -6, -6, 6, 6, -6, -4, -2, -4, -2, 4, 2, 4, 2, -4, -2, 4, 2, -4, -2, 4, 2, -3, 3, -3, 3, -3, 3, -3, 3, -2, -1, -2, -1, -2, -1, -2, -1},
//  {8, -8, -8, 8, -8, 8, 8, -8, 4, 4, -4, -4, -4, -4, 4, 4, 4, -4, 4, -4, -4, 4, -4, 4, 4, -4, -4, 4, 4, -4, -4, 4, 2, 2, 2, 2, -2, -2, -2, -2, 2, 2, -2, -2, 2, 2, -2, -2, 2, -2, 2, -2, 2, -2, 2, -2, 1, 1, 1, 1, 1, 1, 1, 1}
//};
// // naive
//template <typename T>
//__device__ T hermite_3d_interpolate(
//  const T* input,
//  const int depth, const int height, const int width,
//  T z, T y, T x,
//  const int index /* index for debug only*/) {
//  // deal with cases that inverse elements are out of feature map boundary
//  if (y < -1.0 || y > height || x < -1.0 || x > width || z < -1.0 || z > depth) return 0;
//
//  if (z <= 0) z = 0;
//  if (y <= 0) y = 0;
//  if (x <= 0) x = 0;
//
//  int z1 = (int)z;
//  int y1 = (int)y;
//  int x1 = (int)x;
//  int z0 = z1 - 1, z2 = z1 + 1, z3 = z1 + 2;
//  int y0 = y1 - 1, y2 = y1 + 1, y3 = y1 + 2;
//  int x0 = x1 - 1, x2 = x1 + 1, x3 = x1 + 2;
//  float uz = z - z1;
//  float uy = y - y1;
//  float ux = x - x1;
//
//  while (z0 < 0){z0 ++;}
//  while (z1 < 0){z1 ++;}
//  while (z2 < 0){z2 ++;}
//  while (z3 < 0){z3 ++;}
//  while (z0 > depth - 1) {z0 --;}
//  while (z1 > depth - 1) {z1 --;}
//  while (z2 > depth - 1) {z2 --;}
//  while (z3 > depth - 1) {z3 --;}
//
//  while (y0 < 0){y0 ++;}
//  while (y1 < 0){y1 ++;}
//  while (y2 < 0){y2 ++;}
//  while (y3 < 0){y3 ++;}
//  while (y0 > height - 1) {y0 --;}
//  while (y1 > height - 1) {y1 --;}
//  while (y2 > height - 1) {y2 --;}
//  while (y3 > height - 1) {y3 --;}
//
//  while (x0 < 0){x0 ++;}
//  while (x1 < 0){x1 ++;}
//  while (x2 < 0){x2 ++;}
//  while (x3 < 0){x3 ++;}
//  while (x0 > width - 1) {x0 --;}
//  while (x1 > width - 1) {x1 --;}
//  while (x2 > width - 1) {x2 --;}
//  while (x3 > width - 1) {x3 --;}
//
//  //
//  T v000 = input[z0 * height * width + y0 * width + x0];
//  T v100 = input[z0 * height * width + y0 * width + x1];
//  T v200 = input[z0 * height * width + y0 * width + x2];
//  T v300 = input[z0 * height * width + y0 * width + x3];
//
//  T v010 = input[z0 * height * width + y1 * width + x0];
//  T v110 = input[z0 * height * width + y1 * width + x1];
//  T v210 = input[z0 * height * width + y1 * width + x2];
//  T v310 = input[z0 * height * width + y1 * width + x3];
//
//  T v020 = input[z0 * height * width + y2 * width + x0];
//  T v120 = input[z0 * height * width + y2 * width + x1];
//  T v220 = input[z0 * height * width + y2 * width + x2];
//  T v320 = input[z0 * height * width + y2 * width + x3];
//
//  T v030 = input[z0 * height * width + y3 * width + x0];
//  T v130 = input[z0 * height * width + y3 * width + x1];
//  T v230 = input[z0 * height * width + y3 * width + x2];
//  T v330 = input[z0 * height * width + y3 * width + x3];
//  //
//  T v001 = input[z1 * height * width + y0 * width + x0];
//  T v101 = input[z1 * height * width + y0 * width + x1];
//  T v201 = input[z1 * height * width + y0 * width + x2];
//  T v301 = input[z1 * height * width + y0 * width + x3];
//
//  T v011 = input[z1 * height * width + y1 * width + x0];
//  T v111 = input[z1 * height * width + y1 * width + x1];
//  T v211 = input[z1 * height * width + y1 * width + x2];
//  T v311 = input[z1 * height * width + y1 * width + x3];
//
//  T v021 = input[z1 * height * width + y2 * width + x0];
//  T v121 = input[z1 * height * width + y2 * width + x1];
//  T v221 = input[z1 * height * width + y2 * width + x2];
//  T v321 = input[z1 * height * width + y2 * width + x3];
//
//  T v031 = input[z1 * height * width + y3 * width + x0];
//  T v131 = input[z1 * height * width + y3 * width + x1];
//  T v231 = input[z1 * height * width + y3 * width + x2];
//  T v331 = input[z1 * height * width + y3 * width + x3];
//  //
//  T v002 = input[z2 * height * width + y0 * width + x0];
//  T v102 = input[z2 * height * width + y0 * width + x1];
//  T v202 = input[z2 * height * width + y0 * width + x2];
//  T v302 = input[z2 * height * width + y0 * width + x3];
//
//  T v012 = input[z2 * height * width + y1 * width + x0];
//  T v112 = input[z2 * height * width + y1 * width + x1];
//  T v212 = input[z2 * height * width + y1 * width + x2];
//  T v312 = input[z2 * height * width + y1 * width + x3];
//
//  T v022 = input[z2 * height * width + y2 * width + x0];
//  T v122 = input[z2 * height * width + y2 * width + x1];
//  T v222 = input[z2 * height * width + y2 * width + x2];
//  T v322 = input[z2 * height * width + y2 * width + x3];
//
//  T v032 = input[z2 * height * width + y3 * width + x0];
//  T v132 = input[z2 * height * width + y3 * width + x1];
//  T v232 = input[z2 * height * width + y3 * width + x2];
//  T v332 = input[z2 * height * width + y3 * width + x3];
//  //
//  T v003 = input[z3 * height * width + y0 * width + x0];
//  T v103 = input[z3 * height * width + y0 * width + x1];
//  T v203 = input[z3 * height * width + y0 * width + x2];
//  T v303 = input[z3 * height * width + y0 * width + x3];
//
//  T v013 = input[z3 * height * width + y1 * width + x0];
//  T v113 = input[z3 * height * width + y1 * width + x1];
//  T v213 = input[z3 * height * width + y1 * width + x2];
//  T v313 = input[z3 * height * width + y1 * width + x3];
//
//  T v023 = input[z3 * height * width + y2 * width + x0];
//  T v123 = input[z3 * height * width + y2 * width + x1];
//  T v223 = input[z3 * height * width + y2 * width + x2];
//  T v323 = input[z3 * height * width + y2 * width + x3];
//
//  T v033 = input[z3 * height * width + y3 * width + x0];
//  T v133 = input[z3 * height * width + y3 * width + x1];
//  T v233 = input[z3 * height * width + y3 * width + x2];
//  T v333 = input[z3 * height * width + y3 * width + x3];
//
//  T vector_x[64] = {
//    v111,
//    v211,
//    v121,
//    v221,
//    v112,
//    v212,
//    v122,
//    v222,
//    // values of df/dx at each corner.
//    0.5 * (v211 - v011),
//    0.5 * (v311 - v111),
//    0.5 * (v221 - v021),
//    0.5 * (v321 - v121),
//    0.5 * (v212 - v012),
//    0.5 * (v312 - v112),
//    0.5 * (v222 - v022),
//    0.5 * (v322 - v122),
//    // values of df/dy at each corner.
//    0.5 * (v121 - v101),
//    0.5 * (v221 - v201),
//    0.5 * (v131 - v111),
//    0.5 * (v231 - v211),
//    0.5 * (v122 - v102),
//    0.5 * (v222 - v202),
//    0.5 * (v132 - v112),
//    0.5 * (v232 - v212),
//    // values of df/dz at each corner.
//    0.5 * (v112 - v110),
//    0.5 * (v212 - v210),
//    0.5 * (v122 - v120),
//    0.5 * (v222 - v220),
//    0.5 * (v113 - v111),
//    0.5 * (v213 - v211),
//    0.5 * (v123 - v121),
//    0.5 * (v223 - v221),
//    // values of d2f/dxdy at each corner.
//    0.25 * (v221 - v021 - v201 + v001),
//    0.25 * (v321 - v121 - v301 + v101),
//    0.25 * (v231 - v031 - v211 + v011),
//    0.25 * (v331 - v131 - v311 + v111),
//    0.25 * (v222 - v022 - v202 + v002),
//    0.25 * (v322 - v122 - v302 + v102),
//    0.25 * (v232 - v032 - v212 + v012),
//    0.25 * (v332 - v132 - v312 + v112),
//    // values of d2f/dxdz at each corner.
//    0.25 * (v212 - v012 - v210 + v010),
//    0.25 * (v312 - v112 - v310 + v110),
//    0.25 * (v222 - v022 - v220 + v020),
//    0.25 * (v322 - v122 - v320 + v120),
//    0.25 * (v213 - v013 - v211 + v011),
//    0.25 * (v313 - v113 - v311 + v111),
//    0.25 * (v223 - v023 - v221 + v021),
//    0.25 * (v323 - v123 - v321 + v121),
//    // values of d2f/dydz at each corner.
//    0.25 * (v122 - v102 - v120 + v100),
//    0.25 * (v222 - v202 - v220 + v200),
//    0.25 * (v132 - v112 - v130 + v110),
//    0.25 * (v232 - v212 - v230 + v210),
//    0.25 * (v123 - v103 - v121 + v101),
//    0.25 * (v223 - v203 - v221 + v201),
//    0.25 * (v133 - v113 - v131 + v111),
//    0.25 * (v233 - v213 - v231 + v211),
//    // values of d3f/dxdydz at each corner.
//    0.125 * (v222 - v022 - v202 + v002 - v220 + v020 + v200 - v000),
//    0.125 * (v322 - v122 - v302 + v102 - v320 + v120 + v300 - v100),
//    0.125 * (v232 - v032 - v212 + v012 - v230 + v030 + v210 - v010),
//    0.125 * (v332 - v132 - v312 + v112 - v330 + v130 + v310 - v110),
//    0.125 * (v223 - v023 - v203 + v003 - v221 + v021 + v201 - v001),
//    0.125 * (v323 - v123 - v303 + v103 - v321 + v121 + v301 - v101),
//    0.125 * (v233 - v033 - v213 + v013 - v231 + v031 + v211 - v011),
//    0.125 * (v333 - v133 - v313 + v113 - v331 + v131 + v311 - v111)
//  };
//
//  T alpha[64];
//  for ( int i = 0; i < 64 ; i++)
//  {
//    alpha[i] = 0;  // ! important
//    for ( int j = 0; j < 64; j++)
//    {
//      alpha[i] += matrix_a[i][j] * vector_x[j];
//    }
//  }
//
//  T val = 0;
//  T uz_pow = 1;
//  for ( int k = 0; k < 4 ; k++){
//    T uy_pow = 1;
//    for ( int j = 0; j< 4 ; j++){
//      T ux_pow = 1;
//      for ( int i = 0; i< 4 ; i++){
//        val += uz_pow * uy_pow * ux_pow * alpha[i + 4*j +4*4*k];
//        ux_pow *= ux;
//      }
//      uy_pow *= uy;
//    }
//    uz_pow *= uz;
//  }
//
////  T val1 = linear_3d_interpolate(input, depth, height, width, z, y, x, index);
////  printf("Hello from %f, %f\n", val1, val);
//  return val;
//}


template <typename T>
__device__ T hermite_3d_interpolate(
  const T* input,
  const int depth, const int height, const int width,
  T z, T y, T x,
  const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width || z < -1.0 || z > depth) return 0;

  if (z <= 0) z = 0;
  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int z1 = (int)z;
  int y1 = (int)y;
  int x1 = (int)x;
  int z0 = z1 - 1, z2 = z1 + 1, z3 = z1 + 2;
  int y0 = y1 - 1, y2 = y1 + 1, y3 = y1 + 2;
  int x0 = x1 - 1, x2 = x1 + 1, x3 = x1 + 2;
  float uz = z - z1;
  float uy = y - y1;
  float ux = x - x1;

  while (z0 < 0){z0 ++;}
  while (z1 < 0){z1 ++;}
  while (z2 < 0){z2 ++;}
  while (z3 < 0){z3 ++;}
  while (z0 > depth - 1) {z0 --;}
  while (z1 > depth - 1) {z1 --;}
  while (z2 > depth - 1) {z2 --;}
  while (z3 > depth - 1) {z3 --;}

  while (y0 < 0){y0 ++;}
  while (y1 < 0){y1 ++;}
  while (y2 < 0){y2 ++;}
  while (y3 < 0){y3 ++;}
  while (y0 > height - 1) {y0 --;}
  while (y1 > height - 1) {y1 --;}
  while (y2 > height - 1) {y2 --;}
  while (y3 > height - 1) {y3 --;}

  while (x0 < 0){x0 ++;}
  while (x1 < 0){x1 ++;}
  while (x2 < 0){x2 ++;}
  while (x3 < 0){x3 ++;}
  while (x0 > width - 1) {x0 --;}
  while (x1 > width - 1) {x1 --;}
  while (x2 > width - 1) {x2 --;}
  while (x3 > width - 1) {x3 --;}

  //
  T v000 = input[z0 * height * width + y0 * width + x0];
  T v100 = input[z0 * height * width + y0 * width + x1];
  T v200 = input[z0 * height * width + y0 * width + x2];
  T v300 = input[z0 * height * width + y0 * width + x3];

  T v010 = input[z0 * height * width + y1 * width + x0];
  T v110 = input[z0 * height * width + y1 * width + x1];
  T v210 = input[z0 * height * width + y1 * width + x2];
  T v310 = input[z0 * height * width + y1 * width + x3];

  T v020 = input[z0 * height * width + y2 * width + x0];
  T v120 = input[z0 * height * width + y2 * width + x1];
  T v220 = input[z0 * height * width + y2 * width + x2];
  T v320 = input[z0 * height * width + y2 * width + x3];

  T v030 = input[z0 * height * width + y3 * width + x0];
  T v130 = input[z0 * height * width + y3 * width + x1];
  T v230 = input[z0 * height * width + y3 * width + x2];
  T v330 = input[z0 * height * width + y3 * width + x3];
  //
  T v001 = input[z1 * height * width + y0 * width + x0];
  T v101 = input[z1 * height * width + y0 * width + x1];
  T v201 = input[z1 * height * width + y0 * width + x2];
  T v301 = input[z1 * height * width + y0 * width + x3];

  T v011 = input[z1 * height * width + y1 * width + x0];
  T v111 = input[z1 * height * width + y1 * width + x1];
  T v211 = input[z1 * height * width + y1 * width + x2];
  T v311 = input[z1 * height * width + y1 * width + x3];

  T v021 = input[z1 * height * width + y2 * width + x0];
  T v121 = input[z1 * height * width + y2 * width + x1];
  T v221 = input[z1 * height * width + y2 * width + x2];
  T v321 = input[z1 * height * width + y2 * width + x3];

  T v031 = input[z1 * height * width + y3 * width + x0];
  T v131 = input[z1 * height * width + y3 * width + x1];
  T v231 = input[z1 * height * width + y3 * width + x2];
  T v331 = input[z1 * height * width + y3 * width + x3];
  //
  T v002 = input[z2 * height * width + y0 * width + x0];
  T v102 = input[z2 * height * width + y0 * width + x1];
  T v202 = input[z2 * height * width + y0 * width + x2];
  T v302 = input[z2 * height * width + y0 * width + x3];

  T v012 = input[z2 * height * width + y1 * width + x0];
  T v112 = input[z2 * height * width + y1 * width + x1];
  T v212 = input[z2 * height * width + y1 * width + x2];
  T v312 = input[z2 * height * width + y1 * width + x3];

  T v022 = input[z2 * height * width + y2 * width + x0];
  T v122 = input[z2 * height * width + y2 * width + x1];
  T v222 = input[z2 * height * width + y2 * width + x2];
  T v322 = input[z2 * height * width + y2 * width + x3];

  T v032 = input[z2 * height * width + y3 * width + x0];
  T v132 = input[z2 * height * width + y3 * width + x1];
  T v232 = input[z2 * height * width + y3 * width + x2];
  T v332 = input[z2 * height * width + y3 * width + x3];
  //
  T v003 = input[z3 * height * width + y0 * width + x0];
  T v103 = input[z3 * height * width + y0 * width + x1];
  T v203 = input[z3 * height * width + y0 * width + x2];
  T v303 = input[z3 * height * width + y0 * width + x3];

  T v013 = input[z3 * height * width + y1 * width + x0];
  T v113 = input[z3 * height * width + y1 * width + x1];
  T v213 = input[z3 * height * width + y1 * width + x2];
  T v313 = input[z3 * height * width + y1 * width + x3];

  T v023 = input[z3 * height * width + y2 * width + x0];
  T v123 = input[z3 * height * width + y2 * width + x1];
  T v223 = input[z3 * height * width + y2 * width + x2];
  T v323 = input[z3 * height * width + y2 * width + x3];

  T v033 = input[z3 * height * width + y3 * width + x0];
  T v133 = input[z3 * height * width + y3 * width + x1];
  T v233 = input[z3 * height * width + y3 * width + x2];
  T v333 = input[z3 * height * width + y3 * width + x3];

  T ux2 = ux * ux;
  T ux3 = ux * ux * ux;
  T uy2 = uy * uy;
  T uy3 = uy * uy * uy;
  T uz2 = uz * uz;
  T uz3 = uz * uz * uz;

  T w000 = -0.125*ux3*uy3*uz3 + 0.25*ux3*uy3*uz2 - 0.125*ux3*uy3*uz + 0.25*ux3*uy2*uz3 - 0.5*ux3*uy2*uz2 + 0.25*ux3*uy2*uz - 0.125*ux3*uy*uz3 + 0.25*ux3*uy*uz2 - 0.125*ux3*uy*uz + 0.25*ux2*uy3*uz3 - 0.5*ux2*uy3*uz2 + 0.25*ux2*uy3*uz - 0.5*ux2*uy2*uz3 + ux2*uy2*uz2 - 0.5*ux2*uy2*uz + 0.25*ux2*uy*uz3 - 0.5*ux2*uy*uz2 + 0.25*ux2*uy*uz - 0.125*ux*uy3*uz3 + 0.25*ux*uy3*uz2 - 0.125*ux*uy3*uz + 0.25*ux*uy2*uz3 - 0.5*ux*uy2*uz2 + 0.25*ux*uy2*uz - 0.125*ux*uy*uz3 + 0.25*ux*uy*uz2 - 0.125*ux*uy*uz;
  T w100 = 0.375*ux3*uy3*uz3 - 0.75*ux3*uy3*uz2 + 0.375*ux3*uy3*uz - 0.75*ux3*uy2*uz3 + 1.5*ux3*uy2*uz2 - 0.75*ux3*uy2*uz + 0.375*ux3*uy*uz3 - 0.75*ux3*uy*uz2 + 0.375*ux3*uy*uz - 0.625*ux2*uy3*uz3 + 1.25*ux2*uy3*uz2 - 0.625*ux2*uy3*uz + 1.25*ux2*uy2*uz3 - 2.5*ux2*uy2*uz2 + 1.25*ux2*uy2*uz - 0.625*ux2*uy*uz3 + 1.25*ux2*uy*uz2 - 0.625*ux2*uy*uz + 0.25*uy3*uz3 - 0.5*uy3*uz2 + 0.25*uy3*uz - 0.5*uy2*uz3 + uy2*uz2 - 0.5*uy2*uz + 0.25*uy*uz3 - 0.5*uy*uz2 + 0.25*uy*uz;
  T w200 = -0.375*ux3*uy3*uz3 + 0.75*ux3*uy3*uz2 - 0.375*ux3*uy3*uz + 0.75*ux3*uy2*uz3 - 1.5*ux3*uy2*uz2 + 0.75*ux3*uy2*uz - 0.375*ux3*uy*uz3 + 0.75*ux3*uy*uz2 - 0.375*ux3*uy*uz + 0.5*ux2*uy3*uz3 - 1.0*ux2*uy3*uz2 + 0.5*ux2*uy3*uz - 1.0*ux2*uy2*uz3 + 2.0*ux2*uy2*uz2 - 1.0*ux2*uy2*uz + 0.5*ux2*uy*uz3 - 1.0*ux2*uy*uz2 + 0.5*ux2*uy*uz + 0.125*ux*uy3*uz3 - 0.25*ux*uy3*uz2 + 0.125*ux*uy3*uz - 0.25*ux*uy2*uz3 + 0.5*ux*uy2*uz2 - 0.25*ux*uy2*uz + 0.125*ux*uy*uz3 - 0.25*ux*uy*uz2 + 0.125*ux*uy*uz;
  T w300 = 0.125*ux3*uy3*uz3 - 0.25*ux3*uy3*uz2 + 0.125*ux3*uy3*uz - 0.25*ux3*uy2*uz3 + 0.5*ux3*uy2*uz2 - 0.25*ux3*uy2*uz + 0.125*ux3*uy*uz3 - 0.25*ux3*uy*uz2 + 0.125*ux3*uy*uz - 0.125*ux2*uy3*uz3 + 0.25*ux2*uy3*uz2 - 0.125*ux2*uy3*uz + 0.25*ux2*uy2*uz3 - 0.5*ux2*uy2*uz2 + 0.25*ux2*uy2*uz - 0.125*ux2*uy*uz3 + 0.25*ux2*uy*uz2 - 0.125*ux2*uy*uz;
  T w010 = 0.375*ux3*uy3*uz3 - 0.75*ux3*uy3*uz2 + 0.375*ux3*uy3*uz - 0.625*ux3*uy2*uz3 + 1.25*ux3*uy2*uz2 - 0.625*ux3*uy2*uz + 0.25*ux3*uz3 - 0.5*ux3*uz2 + 0.25*ux3*uz - 0.75*ux2*uy3*uz3 + 1.5*ux2*uy3*uz2 - 0.75*ux2*uy3*uz + 1.25*ux2*uy2*uz3 - 2.5*ux2*uy2*uz2 + 1.25*ux2*uy2*uz - 0.5*ux2*uz3 + ux2*uz2 - 0.5*ux2*uz + 0.375*ux*uy3*uz3 - 0.75*ux*uy3*uz2 + 0.375*ux*uy3*uz - 0.625*ux*uy2*uz3 + 1.25*ux*uy2*uz2 - 0.625*ux*uy2*uz + 0.25*ux*uz3 - 0.5*ux*uz2 + 0.25*ux*uz;
  T w110 = -1.125*ux3*uy3*uz3 + 2.25*ux3*uy3*uz2 - 1.125*ux3*uy3*uz + 1.875*ux3*uy2*uz3 - 3.75*ux3*uy2*uz2 + 1.875*ux3*uy2*uz - 0.75*ux3*uz3 + 1.5*ux3*uz2 - 0.75*ux3*uz + 1.875*ux2*uy3*uz3 - 3.75*ux2*uy3*uz2 + 1.875*ux2*uy3*uz - 3.125*ux2*uy2*uz3 + 6.25*ux2*uy2*uz2 - 3.125*ux2*uy2*uz + 1.25*ux2*uz3 - 2.5*ux2*uz2 + 1.25*ux2*uz - 0.75*uy3*uz3 + 1.5*uy3*uz2 - 0.75*uy3*uz + 1.25*uy2*uz3 - 2.5*uy2*uz2 + 1.25*uy2*uz - 0.5*uz3 + uz2 - 0.5*uz;
  T w210 = 1.125*ux3*uy3*uz3 - 2.25*ux3*uy3*uz2 + 1.125*ux3*uy3*uz - 1.875*ux3*uy2*uz3 + 3.75*ux3*uy2*uz2 - 1.875*ux3*uy2*uz + 0.75*ux3*uz3 - 1.5*ux3*uz2 + 0.75*ux3*uz - 1.5*ux2*uy3*uz3 + 3.0*ux2*uy3*uz2 - 1.5*ux2*uy3*uz + 2.5*ux2*uy2*uz3 - 5.0*ux2*uy2*uz2 + 2.5*ux2*uy2*uz - 1.0*ux2*uz3 + 2.0*ux2*uz2 - 1.0*ux2*uz - 0.375*ux*uy3*uz3 + 0.75*ux*uy3*uz2 - 0.375*ux*uy3*uz + 0.625*ux*uy2*uz3 - 1.25*ux*uy2*uz2 + 0.625*ux*uy2*uz - 0.25*ux*uz3 + 0.5*ux*uz2 - 0.25*ux*uz;
  T w310 = -0.375*ux3*uy3*uz3 + 0.75*ux3*uy3*uz2 - 0.375*ux3*uy3*uz + 0.625*ux3*uy2*uz3 - 1.25*ux3*uy2*uz2 + 0.625*ux3*uy2*uz - 0.25*ux3*uz3 + 0.5*ux3*uz2 - 0.25*ux3*uz + 0.375*ux2*uy3*uz3 - 0.75*ux2*uy3*uz2 + 0.375*ux2*uy3*uz - 0.625*ux2*uy2*uz3 + 1.25*ux2*uy2*uz2 - 0.625*ux2*uy2*uz + 0.25*ux2*uz3 - 0.5*ux2*uz2 + 0.25*ux2*uz;
  T w020 = -0.375*ux3*uy3*uz3 + 0.75*ux3*uy3*uz2 - 0.375*ux3*uy3*uz + 0.5*ux3*uy2*uz3 - 1.0*ux3*uy2*uz2 + 0.5*ux3*uy2*uz + 0.125*ux3*uy*uz3 - 0.25*ux3*uy*uz2 + 0.125*ux3*uy*uz + 0.75*ux2*uy3*uz3 - 1.5*ux2*uy3*uz2 + 0.75*ux2*uy3*uz - 1.0*ux2*uy2*uz3 + 2.0*ux2*uy2*uz2 - 1.0*ux2*uy2*uz - 0.25*ux2*uy*uz3 + 0.5*ux2*uy*uz2 - 0.25*ux2*uy*uz - 0.375*ux*uy3*uz3 + 0.75*ux*uy3*uz2 - 0.375*ux*uy3*uz + 0.5*ux*uy2*uz3 - 1.0*ux*uy2*uz2 + 0.5*ux*uy2*uz + 0.125*ux*uy*uz3 - 0.25*ux*uy*uz2 + 0.125*ux*uy*uz;
  T w120 = 1.125*ux3*uy3*uz3 - 2.25*ux3*uy3*uz2 + 1.125*ux3*uy3*uz - 1.5*ux3*uy2*uz3 + 3.0*ux3*uy2*uz2 - 1.5*ux3*uy2*uz - 0.375*ux3*uy*uz3 + 0.75*ux3*uy*uz2 - 0.375*ux3*uy*uz - 1.875*ux2*uy3*uz3 + 3.75*ux2*uy3*uz2 - 1.875*ux2*uy3*uz + 2.5*ux2*uy2*uz3 - 5.0*ux2*uy2*uz2 + 2.5*ux2*uy2*uz + 0.625*ux2*uy*uz3 - 1.25*ux2*uy*uz2 + 0.625*ux2*uy*uz + 0.75*uy3*uz3 - 1.5*uy3*uz2 + 0.75*uy3*uz - 1.0*uy2*uz3 + 2.0*uy2*uz2 - 1.0*uy2*uz - 0.25*uy*uz3 + 0.5*uy*uz2 - 0.25*uy*uz;
  T w220 = -1.125*ux3*uy3*uz3 + 2.25*ux3*uy3*uz2 - 1.125*ux3*uy3*uz + 1.5*ux3*uy2*uz3 - 3.0*ux3*uy2*uz2 + 1.5*ux3*uy2*uz + 0.375*ux3*uy*uz3 - 0.75*ux3*uy*uz2 + 0.375*ux3*uy*uz + 1.5*ux2*uy3*uz3 - 3.0*ux2*uy3*uz2 + 1.5*ux2*uy3*uz - 2.0*ux2*uy2*uz3 + 4.0*ux2*uy2*uz2 - 2.0*ux2*uy2*uz - 0.5*ux2*uy*uz3 + 1.0*ux2*uy*uz2 - 0.5*ux2*uy*uz + 0.375*ux*uy3*uz3 - 0.75*ux*uy3*uz2 + 0.375*ux*uy3*uz - 0.5*ux*uy2*uz3 + 1.0*ux*uy2*uz2 - 0.5*ux*uy2*uz - 0.125*ux*uy*uz3 + 0.25*ux*uy*uz2 - 0.125*ux*uy*uz;
  T w320 = 0.375*ux3*uy3*uz3 - 0.75*ux3*uy3*uz2 + 0.375*ux3*uy3*uz - 0.5*ux3*uy2*uz3 + 1.0*ux3*uy2*uz2 - 0.5*ux3*uy2*uz - 0.125*ux3*uy*uz3 + 0.25*ux3*uy*uz2 - 0.125*ux3*uy*uz - 0.375*ux2*uy3*uz3 + 0.75*ux2*uy3*uz2 - 0.375*ux2*uy3*uz + 0.5*ux2*uy2*uz3 - 1.0*ux2*uy2*uz2 + 0.5*ux2*uy2*uz + 0.125*ux2*uy*uz3 - 0.25*ux2*uy*uz2 + 0.125*ux2*uy*uz;
  T w030 = 0.125*ux3*uy3*uz3 - 0.25*ux3*uy3*uz2 + 0.125*ux3*uy3*uz - 0.125*ux3*uy2*uz3 + 0.25*ux3*uy2*uz2 - 0.125*ux3*uy2*uz - 0.25*ux2*uy3*uz3 + 0.5*ux2*uy3*uz2 - 0.25*ux2*uy3*uz + 0.25*ux2*uy2*uz3 - 0.5*ux2*uy2*uz2 + 0.25*ux2*uy2*uz + 0.125*ux*uy3*uz3 - 0.25*ux*uy3*uz2 + 0.125*ux*uy3*uz - 0.125*ux*uy2*uz3 + 0.25*ux*uy2*uz2 - 0.125*ux*uy2*uz;
  T w130 = -0.375*ux3*uy3*uz3 + 0.75*ux3*uy3*uz2 - 0.375*ux3*uy3*uz + 0.375*ux3*uy2*uz3 - 0.75*ux3*uy2*uz2 + 0.375*ux3*uy2*uz + 0.625*ux2*uy3*uz3 - 1.25*ux2*uy3*uz2 + 0.625*ux2*uy3*uz - 0.625*ux2*uy2*uz3 + 1.25*ux2*uy2*uz2 - 0.625*ux2*uy2*uz - 0.25*uy3*uz3 + 0.5*uy3*uz2 - 0.25*uy3*uz + 0.25*uy2*uz3 - 0.5*uy2*uz2 + 0.25*uy2*uz;
  T w230 = 0.375*ux3*uy3*uz3 - 0.75*ux3*uy3*uz2 + 0.375*ux3*uy3*uz - 0.375*ux3*uy2*uz3 + 0.75*ux3*uy2*uz2 - 0.375*ux3*uy2*uz - 0.5*ux2*uy3*uz3 + 1.0*ux2*uy3*uz2 - 0.5*ux2*uy3*uz + 0.5*ux2*uy2*uz3 - 1.0*ux2*uy2*uz2 + 0.5*ux2*uy2*uz - 0.125*ux*uy3*uz3 + 0.25*ux*uy3*uz2 - 0.125*ux*uy3*uz + 0.125*ux*uy2*uz3 - 0.25*ux*uy2*uz2 + 0.125*ux*uy2*uz;
  T w330 = -0.125*ux3*uy3*uz3 + 0.25*ux3*uy3*uz2 - 0.125*ux3*uy3*uz + 0.125*ux3*uy2*uz3 - 0.25*ux3*uy2*uz2 + 0.125*ux3*uy2*uz + 0.125*ux2*uy3*uz3 - 0.25*ux2*uy3*uz2 + 0.125*ux2*uy3*uz - 0.125*ux2*uy2*uz3 + 0.25*ux2*uy2*uz2 - 0.125*ux2*uy2*uz;
  T w001 = 0.375*ux3*uy3*uz3 - 0.625*ux3*uy3*uz2 + 0.25*ux3*uy3 - 0.75*ux3*uy2*uz3 + 1.25*ux3*uy2*uz2 - 0.5*ux3*uy2 + 0.375*ux3*uy*uz3 - 0.625*ux3*uy*uz2 + 0.25*ux3*uy - 0.75*ux2*uy3*uz3 + 1.25*ux2*uy3*uz2 - 0.5*ux2*uy3 + 1.5*ux2*uy2*uz3 - 2.5*ux2*uy2*uz2 + ux2*uy2 - 0.75*ux2*uy*uz3 + 1.25*ux2*uy*uz2 - 0.5*ux2*uy + 0.375*ux*uy3*uz3 - 0.625*ux*uy3*uz2 + 0.25*ux*uy3 - 0.75*ux*uy2*uz3 + 1.25*ux*uy2*uz2 - 0.5*ux*uy2 + 0.375*ux*uy*uz3 - 0.625*ux*uy*uz2 + 0.25*ux*uy;
  T w101 = -1.125*ux3*uy3*uz3 + 1.875*ux3*uy3*uz2 - 0.75*ux3*uy3 + 2.25*ux3*uy2*uz3 - 3.75*ux3*uy2*uz2 + 1.5*ux3*uy2 - 1.125*ux3*uy*uz3 + 1.875*ux3*uy*uz2 - 0.75*ux3*uy + 1.875*ux2*uy3*uz3 - 3.125*ux2*uy3*uz2 + 1.25*ux2*uy3 - 3.75*ux2*uy2*uz3 + 6.25*ux2*uy2*uz2 - 2.5*ux2*uy2 + 1.875*ux2*uy*uz3 - 3.125*ux2*uy*uz2 + 1.25*ux2*uy - 0.75*uy3*uz3 + 1.25*uy3*uz2 - 0.5*uy3 + 1.5*uy2*uz3 - 2.5*uy2*uz2 + uy2 - 0.75*uy*uz3 + 1.25*uy*uz2 - 0.5*uy;
  T w201 = 1.125*ux3*uy3*uz3 - 1.875*ux3*uy3*uz2 + 0.75*ux3*uy3 - 2.25*ux3*uy2*uz3 + 3.75*ux3*uy2*uz2 - 1.5*ux3*uy2 + 1.125*ux3*uy*uz3 - 1.875*ux3*uy*uz2 + 0.75*ux3*uy - 1.5*ux2*uy3*uz3 + 2.5*ux2*uy3*uz2 - 1.0*ux2*uy3 + 3.0*ux2*uy2*uz3 - 5.0*ux2*uy2*uz2 + 2.0*ux2*uy2 - 1.5*ux2*uy*uz3 + 2.5*ux2*uy*uz2 - 1.0*ux2*uy - 0.375*ux*uy3*uz3 + 0.625*ux*uy3*uz2 - 0.25*ux*uy3 + 0.75*ux*uy2*uz3 - 1.25*ux*uy2*uz2 + 0.5*ux*uy2 - 0.375*ux*uy*uz3 + 0.625*ux*uy*uz2 - 0.25*ux*uy;
  T w301 = -0.375*ux3*uy3*uz3 + 0.625*ux3*uy3*uz2 - 0.25*ux3*uy3 + 0.75*ux3*uy2*uz3 - 1.25*ux3*uy2*uz2 + 0.5*ux3*uy2 - 0.375*ux3*uy*uz3 + 0.625*ux3*uy*uz2 - 0.25*ux3*uy + 0.375*ux2*uy3*uz3 - 0.625*ux2*uy3*uz2 + 0.25*ux2*uy3 - 0.75*ux2*uy2*uz3 + 1.25*ux2*uy2*uz2 - 0.5*ux2*uy2 + 0.375*ux2*uy*uz3 - 0.625*ux2*uy*uz2 + 0.25*ux2*uy;
  T w011 = -1.125*ux3*uy3*uz3 + 1.875*ux3*uy3*uz2 - 0.75*ux3*uy3 + 1.875*ux3*uy2*uz3 - 3.125*ux3*uy2*uz2 + 1.25*ux3*uy2 - 0.75*ux3*uz3 + 1.25*ux3*uz2 - 0.5*ux3 + 2.25*ux2*uy3*uz3 - 3.75*ux2*uy3*uz2 + 1.5*ux2*uy3 - 3.75*ux2*uy2*uz3 + 6.25*ux2*uy2*uz2 - 2.5*ux2*uy2 + 1.5*ux2*uz3 - 2.5*ux2*uz2 + ux2 - 1.125*ux*uy3*uz3 + 1.875*ux*uy3*uz2 - 0.75*ux*uy3 + 1.875*ux*uy2*uz3 - 3.125*ux*uy2*uz2 + 1.25*ux*uy2 - 0.75*ux*uz3 + 1.25*ux*uz2 - 0.5*ux;
  T w111 = 3.375*ux3*uy3*uz3 - 5.625*ux3*uy3*uz2 + 2.25*ux3*uy3 - 5.625*ux3*uy2*uz3 + 9.375*ux3*uy2*uz2 - 3.75*ux3*uy2 + 2.25*ux3*uz3 - 3.75*ux3*uz2 + 1.5*ux3 - 5.625*ux2*uy3*uz3 + 9.375*ux2*uy3*uz2 - 3.75*ux2*uy3 + 9.375*ux2*uy2*uz3 - 15.625*ux2*uy2*uz2 + 6.25*ux2*uy2 - 3.75*ux2*uz3 + 6.25*ux2*uz2 - 2.5*ux2 + 2.25*uy3*uz3 - 3.75*uy3*uz2 + 1.5*uy3 - 3.75*uy2*uz3 + 6.25*uy2*uz2 - 2.5*uy2 + 1.5*uz3 - 2.5*uz2 + 1;
  T w211 = -3.375*ux3*uy3*uz3 + 5.625*ux3*uy3*uz2 - 2.25*ux3*uy3 + 5.625*ux3*uy2*uz3 - 9.375*ux3*uy2*uz2 + 3.75*ux3*uy2 - 2.25*ux3*uz3 + 3.75*ux3*uz2 - 1.5*ux3 + 4.5*ux2*uy3*uz3 - 7.5*ux2*uy3*uz2 + 3.0*ux2*uy3 - 7.5*ux2*uy2*uz3 + 12.5*ux2*uy2*uz2 - 5.0*ux2*uy2 + 3.0*ux2*uz3 - 5.0*ux2*uz2 + 2.0*ux2 + 1.125*ux*uy3*uz3 - 1.875*ux*uy3*uz2 + 0.75*ux*uy3 - 1.875*ux*uy2*uz3 + 3.125*ux*uy2*uz2 - 1.25*ux*uy2 + 0.75*ux*uz3 - 1.25*ux*uz2 + 0.5*ux;
  T w311 = 1.125*ux3*uy3*uz3 - 1.875*ux3*uy3*uz2 + 0.75*ux3*uy3 - 1.875*ux3*uy2*uz3 + 3.125*ux3*uy2*uz2 - 1.25*ux3*uy2 + 0.75*ux3*uz3 - 1.25*ux3*uz2 + 0.5*ux3 - 1.125*ux2*uy3*uz3 + 1.875*ux2*uy3*uz2 - 0.75*ux2*uy3 + 1.875*ux2*uy2*uz3 - 3.125*ux2*uy2*uz2 + 1.25*ux2*uy2 - 0.75*ux2*uz3 + 1.25*ux2*uz2 - 0.5*ux2;
  T w021 = 1.125*ux3*uy3*uz3 - 1.875*ux3*uy3*uz2 + 0.75*ux3*uy3 - 1.5*ux3*uy2*uz3 + 2.5*ux3*uy2*uz2 - 1.0*ux3*uy2 - 0.375*ux3*uy*uz3 + 0.625*ux3*uy*uz2 - 0.25*ux3*uy - 2.25*ux2*uy3*uz3 + 3.75*ux2*uy3*uz2 - 1.5*ux2*uy3 + 3.0*ux2*uy2*uz3 - 5.0*ux2*uy2*uz2 + 2.0*ux2*uy2 + 0.75*ux2*uy*uz3 - 1.25*ux2*uy*uz2 + 0.5*ux2*uy + 1.125*ux*uy3*uz3 - 1.875*ux*uy3*uz2 + 0.75*ux*uy3 - 1.5*ux*uy2*uz3 + 2.5*ux*uy2*uz2 - 1.0*ux*uy2 - 0.375*ux*uy*uz3 + 0.625*ux*uy*uz2 - 0.25*ux*uy;
  T w121 = -3.375*ux3*uy3*uz3 + 5.625*ux3*uy3*uz2 - 2.25*ux3*uy3 + 4.5*ux3*uy2*uz3 - 7.5*ux3*uy2*uz2 + 3.0*ux3*uy2 + 1.125*ux3*uy*uz3 - 1.875*ux3*uy*uz2 + 0.75*ux3*uy + 5.625*ux2*uy3*uz3 - 9.375*ux2*uy3*uz2 + 3.75*ux2*uy3 - 7.5*ux2*uy2*uz3 + 12.5*ux2*uy2*uz2 - 5.0*ux2*uy2 - 1.875*ux2*uy*uz3 + 3.125*ux2*uy*uz2 - 1.25*ux2*uy - 2.25*uy3*uz3 + 3.75*uy3*uz2 - 1.5*uy3 + 3.0*uy2*uz3 - 5.0*uy2*uz2 + 2.0*uy2 + 0.75*uy*uz3 - 1.25*uy*uz2 + 0.5*uy;
  T w221 = 3.375*ux3*uy3*uz3 - 5.625*ux3*uy3*uz2 + 2.25*ux3*uy3 - 4.5*ux3*uy2*uz3 + 7.5*ux3*uy2*uz2 - 3.0*ux3*uy2 - 1.125*ux3*uy*uz3 + 1.875*ux3*uy*uz2 - 0.75*ux3*uy - 4.5*ux2*uy3*uz3 + 7.5*ux2*uy3*uz2 - 3.0*ux2*uy3 + 6.0*ux2*uy2*uz3 - 10.0*ux2*uy2*uz2 + 4.0*ux2*uy2 + 1.5*ux2*uy*uz3 - 2.5*ux2*uy*uz2 + 1.0*ux2*uy - 1.125*ux*uy3*uz3 + 1.875*ux*uy3*uz2 - 0.75*ux*uy3 + 1.5*ux*uy2*uz3 - 2.5*ux*uy2*uz2 + 1.0*ux*uy2 + 0.375*ux*uy*uz3 - 0.625*ux*uy*uz2 + 0.25*ux*uy;
  T w321 = -1.125*ux3*uy3*uz3 + 1.875*ux3*uy3*uz2 - 0.75*ux3*uy3 + 1.5*ux3*uy2*uz3 - 2.5*ux3*uy2*uz2 + 1.0*ux3*uy2 + 0.375*ux3*uy*uz3 - 0.625*ux3*uy*uz2 + 0.25*ux3*uy + 1.125*ux2*uy3*uz3 - 1.875*ux2*uy3*uz2 + 0.75*ux2*uy3 - 1.5*ux2*uy2*uz3 + 2.5*ux2*uy2*uz2 - 1.0*ux2*uy2 - 0.375*ux2*uy*uz3 + 0.625*ux2*uy*uz2 - 0.25*ux2*uy;
  T w031 = -0.375*ux3*uy3*uz3 + 0.625*ux3*uy3*uz2 - 0.25*ux3*uy3 + 0.375*ux3*uy2*uz3 - 0.625*ux3*uy2*uz2 + 0.25*ux3*uy2 + 0.75*ux2*uy3*uz3 - 1.25*ux2*uy3*uz2 + 0.5*ux2*uy3 - 0.75*ux2*uy2*uz3 + 1.25*ux2*uy2*uz2 - 0.5*ux2*uy2 - 0.375*ux*uy3*uz3 + 0.625*ux*uy3*uz2 - 0.25*ux*uy3 + 0.375*ux*uy2*uz3 - 0.625*ux*uy2*uz2 + 0.25*ux*uy2;
  T w131 = 1.125*ux3*uy3*uz3 - 1.875*ux3*uy3*uz2 + 0.75*ux3*uy3 - 1.125*ux3*uy2*uz3 + 1.875*ux3*uy2*uz2 - 0.75*ux3*uy2 - 1.875*ux2*uy3*uz3 + 3.125*ux2*uy3*uz2 - 1.25*ux2*uy3 + 1.875*ux2*uy2*uz3 - 3.125*ux2*uy2*uz2 + 1.25*ux2*uy2 + 0.75*uy3*uz3 - 1.25*uy3*uz2 + 0.5*uy3 - 0.75*uy2*uz3 + 1.25*uy2*uz2 - 0.5*uy2;
  T w231 = -1.125*ux3*uy3*uz3 + 1.875*ux3*uy3*uz2 - 0.75*ux3*uy3 + 1.125*ux3*uy2*uz3 - 1.875*ux3*uy2*uz2 + 0.75*ux3*uy2 + 1.5*ux2*uy3*uz3 - 2.5*ux2*uy3*uz2 + 1.0*ux2*uy3 - 1.5*ux2*uy2*uz3 + 2.5*ux2*uy2*uz2 - 1.0*ux2*uy2 + 0.375*ux*uy3*uz3 - 0.625*ux*uy3*uz2 + 0.25*ux*uy3 - 0.375*ux*uy2*uz3 + 0.625*ux*uy2*uz2 - 0.25*ux*uy2;
  T w331 = 0.375*ux3*uy3*uz3 - 0.625*ux3*uy3*uz2 + 0.25*ux3*uy3 - 0.375*ux3*uy2*uz3 + 0.625*ux3*uy2*uz2 - 0.25*ux3*uy2 - 0.375*ux2*uy3*uz3 + 0.625*ux2*uy3*uz2 - 0.25*ux2*uy3 + 0.375*ux2*uy2*uz3 - 0.625*ux2*uy2*uz2 + 0.25*ux2*uy2;
  T w002 = -0.375*ux3*uy3*uz3 + 0.5*ux3*uy3*uz2 + 0.125*ux3*uy3*uz + 0.75*ux3*uy2*uz3 - 1.0*ux3*uy2*uz2 - 0.25*ux3*uy2*uz - 0.375*ux3*uy*uz3 + 0.5*ux3*uy*uz2 + 0.125*ux3*uy*uz + 0.75*ux2*uy3*uz3 - 1.0*ux2*uy3*uz2 - 0.25*ux2*uy3*uz - 1.5*ux2*uy2*uz3 + 2.0*ux2*uy2*uz2 + 0.5*ux2*uy2*uz + 0.75*ux2*uy*uz3 - 1.0*ux2*uy*uz2 - 0.25*ux2*uy*uz - 0.375*ux*uy3*uz3 + 0.5*ux*uy3*uz2 + 0.125*ux*uy3*uz + 0.75*ux*uy2*uz3 - 1.0*ux*uy2*uz2 - 0.25*ux*uy2*uz - 0.375*ux*uy*uz3 + 0.5*ux*uy*uz2 + 0.125*ux*uy*uz;
  T w102 = 1.125*ux3*uy3*uz3 - 1.5*ux3*uy3*uz2 - 0.375*ux3*uy3*uz - 2.25*ux3*uy2*uz3 + 3.0*ux3*uy2*uz2 + 0.75*ux3*uy2*uz + 1.125*ux3*uy*uz3 - 1.5*ux3*uy*uz2 - 0.375*ux3*uy*uz - 1.875*ux2*uy3*uz3 + 2.5*ux2*uy3*uz2 + 0.625*ux2*uy3*uz + 3.75*ux2*uy2*uz3 - 5.0*ux2*uy2*uz2 - 1.25*ux2*uy2*uz - 1.875*ux2*uy*uz3 + 2.5*ux2*uy*uz2 + 0.625*ux2*uy*uz + 0.75*uy3*uz3 - 1.0*uy3*uz2 - 0.25*uy3*uz - 1.5*uy2*uz3 + 2.0*uy2*uz2 + 0.5*uy2*uz + 0.75*uy*uz3 - 1.0*uy*uz2 - 0.25*uy*uz;
  T w202 = -1.125*ux3*uy3*uz3 + 1.5*ux3*uy3*uz2 + 0.375*ux3*uy3*uz + 2.25*ux3*uy2*uz3 - 3.0*ux3*uy2*uz2 - 0.75*ux3*uy2*uz - 1.125*ux3*uy*uz3 + 1.5*ux3*uy*uz2 + 0.375*ux3*uy*uz + 1.5*ux2*uy3*uz3 - 2.0*ux2*uy3*uz2 - 0.5*ux2*uy3*uz - 3.0*ux2*uy2*uz3 + 4.0*ux2*uy2*uz2 + 1.0*ux2*uy2*uz + 1.5*ux2*uy*uz3 - 2.0*ux2*uy*uz2 - 0.5*ux2*uy*uz + 0.375*ux*uy3*uz3 - 0.5*ux*uy3*uz2 - 0.125*ux*uy3*uz - 0.75*ux*uy2*uz3 + 1.0*ux*uy2*uz2 + 0.25*ux*uy2*uz + 0.375*ux*uy*uz3 - 0.5*ux*uy*uz2 - 0.125*ux*uy*uz;
  T w302 = 0.375*ux3*uy3*uz3 - 0.5*ux3*uy3*uz2 - 0.125*ux3*uy3*uz - 0.75*ux3*uy2*uz3 + 1.0*ux3*uy2*uz2 + 0.25*ux3*uy2*uz + 0.375*ux3*uy*uz3 - 0.5*ux3*uy*uz2 - 0.125*ux3*uy*uz - 0.375*ux2*uy3*uz3 + 0.5*ux2*uy3*uz2 + 0.125*ux2*uy3*uz + 0.75*ux2*uy2*uz3 - 1.0*ux2*uy2*uz2 - 0.25*ux2*uy2*uz - 0.375*ux2*uy*uz3 + 0.5*ux2*uy*uz2 + 0.125*ux2*uy*uz;
  T w012 = 1.125*ux3*uy3*uz3 - 1.5*ux3*uy3*uz2 - 0.375*ux3*uy3*uz - 1.875*ux3*uy2*uz3 + 2.5*ux3*uy2*uz2 + 0.625*ux3*uy2*uz + 0.75*ux3*uz3 - 1.0*ux3*uz2 - 0.25*ux3*uz - 2.25*ux2*uy3*uz3 + 3.0*ux2*uy3*uz2 + 0.75*ux2*uy3*uz + 3.75*ux2*uy2*uz3 - 5.0*ux2*uy2*uz2 - 1.25*ux2*uy2*uz - 1.5*ux2*uz3 + 2.0*ux2*uz2 + 0.5*ux2*uz + 1.125*ux*uy3*uz3 - 1.5*ux*uy3*uz2 - 0.375*ux*uy3*uz - 1.875*ux*uy2*uz3 + 2.5*ux*uy2*uz2 + 0.625*ux*uy2*uz + 0.75*ux*uz3 - 1.0*ux*uz2 - 0.25*ux*uz;
  T w112 = -3.375*ux3*uy3*uz3 + 4.5*ux3*uy3*uz2 + 1.125*ux3*uy3*uz + 5.625*ux3*uy2*uz3 - 7.5*ux3*uy2*uz2 - 1.875*ux3*uy2*uz - 2.25*ux3*uz3 + 3.0*ux3*uz2 + 0.75*ux3*uz + 5.625*ux2*uy3*uz3 - 7.5*ux2*uy3*uz2 - 1.875*ux2*uy3*uz - 9.375*ux2*uy2*uz3 + 12.5*ux2*uy2*uz2 + 3.125*ux2*uy2*uz + 3.75*ux2*uz3 - 5.0*ux2*uz2 - 1.25*ux2*uz - 2.25*uy3*uz3 + 3.0*uy3*uz2 + 0.75*uy3*uz + 3.75*uy2*uz3 - 5.0*uy2*uz2 - 1.25*uy2*uz - 1.5*uz3 + 2.0*uz2 + 0.5*uz;
  T w212 = 3.375*ux3*uy3*uz3 - 4.5*ux3*uy3*uz2 - 1.125*ux3*uy3*uz - 5.625*ux3*uy2*uz3 + 7.5*ux3*uy2*uz2 + 1.875*ux3*uy2*uz + 2.25*ux3*uz3 - 3.0*ux3*uz2 - 0.75*ux3*uz - 4.5*ux2*uy3*uz3 + 6.0*ux2*uy3*uz2 + 1.5*ux2*uy3*uz + 7.5*ux2*uy2*uz3 - 10.0*ux2*uy2*uz2 - 2.5*ux2*uy2*uz - 3.0*ux2*uz3 + 4.0*ux2*uz2 + 1.0*ux2*uz - 1.125*ux*uy3*uz3 + 1.5*ux*uy3*uz2 + 0.375*ux*uy3*uz + 1.875*ux*uy2*uz3 - 2.5*ux*uy2*uz2 - 0.625*ux*uy2*uz - 0.75*ux*uz3 + 1.0*ux*uz2 + 0.25*ux*uz;
  T w312 = -1.125*ux3*uy3*uz3 + 1.5*ux3*uy3*uz2 + 0.375*ux3*uy3*uz + 1.875*ux3*uy2*uz3 - 2.5*ux3*uy2*uz2 - 0.625*ux3*uy2*uz - 0.75*ux3*uz3 + 1.0*ux3*uz2 + 0.25*ux3*uz + 1.125*ux2*uy3*uz3 - 1.5*ux2*uy3*uz2 - 0.375*ux2*uy3*uz - 1.875*ux2*uy2*uz3 + 2.5*ux2*uy2*uz2 + 0.625*ux2*uy2*uz + 0.75*ux2*uz3 - 1.0*ux2*uz2 - 0.25*ux2*uz;
  T w022 = -1.125*ux3*uy3*uz3 + 1.5*ux3*uy3*uz2 + 0.375*ux3*uy3*uz + 1.5*ux3*uy2*uz3 - 2.0*ux3*uy2*uz2 - 0.5*ux3*uy2*uz + 0.375*ux3*uy*uz3 - 0.5*ux3*uy*uz2 - 0.125*ux3*uy*uz + 2.25*ux2*uy3*uz3 - 3.0*ux2*uy3*uz2 - 0.75*ux2*uy3*uz - 3.0*ux2*uy2*uz3 + 4.0*ux2*uy2*uz2 + 1.0*ux2*uy2*uz - 0.75*ux2*uy*uz3 + 1.0*ux2*uy*uz2 + 0.25*ux2*uy*uz - 1.125*ux*uy3*uz3 + 1.5*ux*uy3*uz2 + 0.375*ux*uy3*uz + 1.5*ux*uy2*uz3 - 2.0*ux*uy2*uz2 - 0.5*ux*uy2*uz + 0.375*ux*uy*uz3 - 0.5*ux*uy*uz2 - 0.125*ux*uy*uz;
  T w122 = 3.375*ux3*uy3*uz3 - 4.5*ux3*uy3*uz2 - 1.125*ux3*uy3*uz - 4.5*ux3*uy2*uz3 + 6.0*ux3*uy2*uz2 + 1.5*ux3*uy2*uz - 1.125*ux3*uy*uz3 + 1.5*ux3*uy*uz2 + 0.375*ux3*uy*uz - 5.625*ux2*uy3*uz3 + 7.5*ux2*uy3*uz2 + 1.875*ux2*uy3*uz + 7.5*ux2*uy2*uz3 - 10.0*ux2*uy2*uz2 - 2.5*ux2*uy2*uz + 1.875*ux2*uy*uz3 - 2.5*ux2*uy*uz2 - 0.625*ux2*uy*uz + 2.25*uy3*uz3 - 3.0*uy3*uz2 - 0.75*uy3*uz - 3.0*uy2*uz3 + 4.0*uy2*uz2 + 1.0*uy2*uz - 0.75*uy*uz3 + 1.0*uy*uz2 + 0.25*uy*uz;
  T w222 = -3.375*ux3*uy3*uz3 + 4.5*ux3*uy3*uz2 + 1.125*ux3*uy3*uz + 4.5*ux3*uy2*uz3 - 6.0*ux3*uy2*uz2 - 1.5*ux3*uy2*uz + 1.125*ux3*uy*uz3 - 1.5*ux3*uy*uz2 - 0.375*ux3*uy*uz + 4.5*ux2*uy3*uz3 - 6.0*ux2*uy3*uz2 - 1.5*ux2*uy3*uz - 6.0*ux2*uy2*uz3 + 8.0*ux2*uy2*uz2 + 2.0*ux2*uy2*uz - 1.5*ux2*uy*uz3 + 2.0*ux2*uy*uz2 + 0.5*ux2*uy*uz + 1.125*ux*uy3*uz3 - 1.5*ux*uy3*uz2 - 0.375*ux*uy3*uz - 1.5*ux*uy2*uz3 + 2.0*ux*uy2*uz2 + 0.5*ux*uy2*uz - 0.375*ux*uy*uz3 + 0.5*ux*uy*uz2 + 0.125*ux*uy*uz;
  T w322 = 1.125*ux3*uy3*uz3 - 1.5*ux3*uy3*uz2 - 0.375*ux3*uy3*uz - 1.5*ux3*uy2*uz3 + 2.0*ux3*uy2*uz2 + 0.5*ux3*uy2*uz - 0.375*ux3*uy*uz3 + 0.5*ux3*uy*uz2 + 0.125*ux3*uy*uz - 1.125*ux2*uy3*uz3 + 1.5*ux2*uy3*uz2 + 0.375*ux2*uy3*uz + 1.5*ux2*uy2*uz3 - 2.0*ux2*uy2*uz2 - 0.5*ux2*uy2*uz + 0.375*ux2*uy*uz3 - 0.5*ux2*uy*uz2 - 0.125*ux2*uy*uz;
  T w032 = 0.375*ux3*uy3*uz3 - 0.5*ux3*uy3*uz2 - 0.125*ux3*uy3*uz - 0.375*ux3*uy2*uz3 + 0.5*ux3*uy2*uz2 + 0.125*ux3*uy2*uz - 0.75*ux2*uy3*uz3 + 1.0*ux2*uy3*uz2 + 0.25*ux2*uy3*uz + 0.75*ux2*uy2*uz3 - 1.0*ux2*uy2*uz2 - 0.25*ux2*uy2*uz + 0.375*ux*uy3*uz3 - 0.5*ux*uy3*uz2 - 0.125*ux*uy3*uz - 0.375*ux*uy2*uz3 + 0.5*ux*uy2*uz2 + 0.125*ux*uy2*uz;
  T w132 = -1.125*ux3*uy3*uz3 + 1.5*ux3*uy3*uz2 + 0.375*ux3*uy3*uz + 1.125*ux3*uy2*uz3 - 1.5*ux3*uy2*uz2 - 0.375*ux3*uy2*uz + 1.875*ux2*uy3*uz3 - 2.5*ux2*uy3*uz2 - 0.625*ux2*uy3*uz - 1.875*ux2*uy2*uz3 + 2.5*ux2*uy2*uz2 + 0.625*ux2*uy2*uz - 0.75*uy3*uz3 + 1.0*uy3*uz2 + 0.25*uy3*uz + 0.75*uy2*uz3 - 1.0*uy2*uz2 - 0.25*uy2*uz;
  T w232 = 1.125*ux3*uy3*uz3 - 1.5*ux3*uy3*uz2 - 0.375*ux3*uy3*uz - 1.125*ux3*uy2*uz3 + 1.5*ux3*uy2*uz2 + 0.375*ux3*uy2*uz - 1.5*ux2*uy3*uz3 + 2.0*ux2*uy3*uz2 + 0.5*ux2*uy3*uz + 1.5*ux2*uy2*uz3 - 2.0*ux2*uy2*uz2 - 0.5*ux2*uy2*uz - 0.375*ux*uy3*uz3 + 0.5*ux*uy3*uz2 + 0.125*ux*uy3*uz + 0.375*ux*uy2*uz3 - 0.5*ux*uy2*uz2 - 0.125*ux*uy2*uz;
  T w332 = -0.375*ux3*uy3*uz3 + 0.5*ux3*uy3*uz2 + 0.125*ux3*uy3*uz + 0.375*ux3*uy2*uz3 - 0.5*ux3*uy2*uz2 - 0.125*ux3*uy2*uz + 0.375*ux2*uy3*uz3 - 0.5*ux2*uy3*uz2 - 0.125*ux2*uy3*uz - 0.375*ux2*uy2*uz3 + 0.5*ux2*uy2*uz2 + 0.125*ux2*uy2*uz;
  T w003 = 0.125*ux3*uy3*uz3 - 0.125*ux3*uy3*uz2 - 0.25*ux3*uy2*uz3 + 0.25*ux3*uy2*uz2 + 0.125*ux3*uy*uz3 - 0.125*ux3*uy*uz2 - 0.25*ux2*uy3*uz3 + 0.25*ux2*uy3*uz2 + 0.5*ux2*uy2*uz3 - 0.5*ux2*uy2*uz2 - 0.25*ux2*uy*uz3 + 0.25*ux2*uy*uz2 + 0.125*ux*uy3*uz3 - 0.125*ux*uy3*uz2 - 0.25*ux*uy2*uz3 + 0.25*ux*uy2*uz2 + 0.125*ux*uy*uz3 - 0.125*ux*uy*uz2;
  T w103 = -0.375*ux3*uy3*uz3 + 0.375*ux3*uy3*uz2 + 0.75*ux3*uy2*uz3 - 0.75*ux3*uy2*uz2 - 0.375*ux3*uy*uz3 + 0.375*ux3*uy*uz2 + 0.625*ux2*uy3*uz3 - 0.625*ux2*uy3*uz2 - 1.25*ux2*uy2*uz3 + 1.25*ux2*uy2*uz2 + 0.625*ux2*uy*uz3 - 0.625*ux2*uy*uz2 - 0.25*uy3*uz3 + 0.25*uy3*uz2 + 0.5*uy2*uz3 - 0.5*uy2*uz2 - 0.25*uy*uz3 + 0.25*uy*uz2;
  T w203 = 0.375*ux3*uy3*uz3 - 0.375*ux3*uy3*uz2 - 0.75*ux3*uy2*uz3 + 0.75*ux3*uy2*uz2 + 0.375*ux3*uy*uz3 - 0.375*ux3*uy*uz2 - 0.5*ux2*uy3*uz3 + 0.5*ux2*uy3*uz2 + 1.0*ux2*uy2*uz3 - 1.0*ux2*uy2*uz2 - 0.5*ux2*uy*uz3 + 0.5*ux2*uy*uz2 - 0.125*ux*uy3*uz3 + 0.125*ux*uy3*uz2 + 0.25*ux*uy2*uz3 - 0.25*ux*uy2*uz2 - 0.125*ux*uy*uz3 + 0.125*ux*uy*uz2;
  T w303 = -0.125*ux3*uy3*uz3 + 0.125*ux3*uy3*uz2 + 0.25*ux3*uy2*uz3 - 0.25*ux3*uy2*uz2 - 0.125*ux3*uy*uz3 + 0.125*ux3*uy*uz2 + 0.125*ux2*uy3*uz3 - 0.125*ux2*uy3*uz2 - 0.25*ux2*uy2*uz3 + 0.25*ux2*uy2*uz2 + 0.125*ux2*uy*uz3 - 0.125*ux2*uy*uz2;
  T w013 = -0.375*ux3*uy3*uz3 + 0.375*ux3*uy3*uz2 + 0.625*ux3*uy2*uz3 - 0.625*ux3*uy2*uz2 - 0.25*ux3*uz3 + 0.25*ux3*uz2 + 0.75*ux2*uy3*uz3 - 0.75*ux2*uy3*uz2 - 1.25*ux2*uy2*uz3 + 1.25*ux2*uy2*uz2 + 0.5*ux2*uz3 - 0.5*ux2*uz2 - 0.375*ux*uy3*uz3 + 0.375*ux*uy3*uz2 + 0.625*ux*uy2*uz3 - 0.625*ux*uy2*uz2 - 0.25*ux*uz3 + 0.25*ux*uz2;
  T w113 = 1.125*ux3*uy3*uz3 - 1.125*ux3*uy3*uz2 - 1.875*ux3*uy2*uz3 + 1.875*ux3*uy2*uz2 + 0.75*ux3*uz3 - 0.75*ux3*uz2 - 1.875*ux2*uy3*uz3 + 1.875*ux2*uy3*uz2 + 3.125*ux2*uy2*uz3 - 3.125*ux2*uy2*uz2 - 1.25*ux2*uz3 + 1.25*ux2*uz2 + 0.75*uy3*uz3 - 0.75*uy3*uz2 - 1.25*uy2*uz3 + 1.25*uy2*uz2 + 0.5*uz3 - 0.5*uz2;
  T w213 = -1.125*ux3*uy3*uz3 + 1.125*ux3*uy3*uz2 + 1.875*ux3*uy2*uz3 - 1.875*ux3*uy2*uz2 - 0.75*ux3*uz3 + 0.75*ux3*uz2 + 1.5*ux2*uy3*uz3 - 1.5*ux2*uy3*uz2 - 2.5*ux2*uy2*uz3 + 2.5*ux2*uy2*uz2 + 1.0*ux2*uz3 - 1.0*ux2*uz2 + 0.375*ux*uy3*uz3 - 0.375*ux*uy3*uz2 - 0.625*ux*uy2*uz3 + 0.625*ux*uy2*uz2 + 0.25*ux*uz3 - 0.25*ux*uz2;
  T w313 = 0.375*ux3*uy3*uz3 - 0.375*ux3*uy3*uz2 - 0.625*ux3*uy2*uz3 + 0.625*ux3*uy2*uz2 + 0.25*ux3*uz3 - 0.25*ux3*uz2 - 0.375*ux2*uy3*uz3 + 0.375*ux2*uy3*uz2 + 0.625*ux2*uy2*uz3 - 0.625*ux2*uy2*uz2 - 0.25*ux2*uz3 + 0.25*ux2*uz2;
  T w023 = 0.375*ux3*uy3*uz3 - 0.375*ux3*uy3*uz2 - 0.5*ux3*uy2*uz3 + 0.5*ux3*uy2*uz2 - 0.125*ux3*uy*uz3 + 0.125*ux3*uy*uz2 - 0.75*ux2*uy3*uz3 + 0.75*ux2*uy3*uz2 + 1.0*ux2*uy2*uz3 - 1.0*ux2*uy2*uz2 + 0.25*ux2*uy*uz3 - 0.25*ux2*uy*uz2 + 0.375*ux*uy3*uz3 - 0.375*ux*uy3*uz2 - 0.5*ux*uy2*uz3 + 0.5*ux*uy2*uz2 - 0.125*ux*uy*uz3 + 0.125*ux*uy*uz2;
  T w123 = -1.125*ux3*uy3*uz3 + 1.125*ux3*uy3*uz2 + 1.5*ux3*uy2*uz3 - 1.5*ux3*uy2*uz2 + 0.375*ux3*uy*uz3 - 0.375*ux3*uy*uz2 + 1.875*ux2*uy3*uz3 - 1.875*ux2*uy3*uz2 - 2.5*ux2*uy2*uz3 + 2.5*ux2*uy2*uz2 - 0.625*ux2*uy*uz3 + 0.625*ux2*uy*uz2 - 0.75*uy3*uz3 + 0.75*uy3*uz2 + 1.0*uy2*uz3 - 1.0*uy2*uz2 + 0.25*uy*uz3 - 0.25*uy*uz2;
  T w223 = 1.125*ux3*uy3*uz3 - 1.125*ux3*uy3*uz2 - 1.5*ux3*uy2*uz3 + 1.5*ux3*uy2*uz2 - 0.375*ux3*uy*uz3 + 0.375*ux3*uy*uz2 - 1.5*ux2*uy3*uz3 + 1.5*ux2*uy3*uz2 + 2.0*ux2*uy2*uz3 - 2.0*ux2*uy2*uz2 + 0.5*ux2*uy*uz3 - 0.5*ux2*uy*uz2 - 0.375*ux*uy3*uz3 + 0.375*ux*uy3*uz2 + 0.5*ux*uy2*uz3 - 0.5*ux*uy2*uz2 + 0.125*ux*uy*uz3 - 0.125*ux*uy*uz2;
  T w323 = -0.375*ux3*uy3*uz3 + 0.375*ux3*uy3*uz2 + 0.5*ux3*uy2*uz3 - 0.5*ux3*uy2*uz2 + 0.125*ux3*uy*uz3 - 0.125*ux3*uy*uz2 + 0.375*ux2*uy3*uz3 - 0.375*ux2*uy3*uz2 - 0.5*ux2*uy2*uz3 + 0.5*ux2*uy2*uz2 - 0.125*ux2*uy*uz3 + 0.125*ux2*uy*uz2;
  T w033 = -0.125*ux3*uy3*uz3 + 0.125*ux3*uy3*uz2 + 0.125*ux3*uy2*uz3 - 0.125*ux3*uy2*uz2 + 0.25*ux2*uy3*uz3 - 0.25*ux2*uy3*uz2 - 0.25*ux2*uy2*uz3 + 0.25*ux2*uy2*uz2 - 0.125*ux*uy3*uz3 + 0.125*ux*uy3*uz2 + 0.125*ux*uy2*uz3 - 0.125*ux*uy2*uz2;
  T w133 = 0.375*ux3*uy3*uz3 - 0.375*ux3*uy3*uz2 - 0.375*ux3*uy2*uz3 + 0.375*ux3*uy2*uz2 - 0.625*ux2*uy3*uz3 + 0.625*ux2*uy3*uz2 + 0.625*ux2*uy2*uz3 - 0.625*ux2*uy2*uz2 + 0.25*uy3*uz3 - 0.25*uy3*uz2 - 0.25*uy2*uz3 + 0.25*uy2*uz2;
  T w233 = -0.375*ux3*uy3*uz3 + 0.375*ux3*uy3*uz2 + 0.375*ux3*uy2*uz3 - 0.375*ux3*uy2*uz2 + 0.5*ux2*uy3*uz3 - 0.5*ux2*uy3*uz2 - 0.5*ux2*uy2*uz3 + 0.5*ux2*uy2*uz2 + 0.125*ux*uy3*uz3 - 0.125*ux*uy3*uz2 - 0.125*ux*uy2*uz3 + 0.125*ux*uy2*uz2;
  T w333 = 0.125*ux3*uy3*uz3 - 0.125*ux3*uy3*uz2 - 0.125*ux3*uy2*uz3 + 0.125*ux3*uy2*uz2 - 0.125*ux2*uy3*uz3 + 0.125*ux2*uy3*uz2 + 0.125*ux2*uy2*uz3 - 0.125*ux2*uy2*uz2;

  T val = 0;
  val += w000 * v000 + w100 * v100 + w200 * v200 + w300 * v300;
  val += w010 * v010 + w110 * v110 + w210 * v210 + w310 * v310;
  val += w020 * v020 + w120 * v120 + w220 * v220 + w320 * v320;
  val += w030 * v030 + w130 * v130 + w230 * v230 + w330 * v330;
  val += w001 * v001 + w101 * v101 + w201 * v201 + w301 * v301;
  val += w011 * v011 + w111 * v111 + w211 * v211 + w311 * v311;
  val += w021 * v021 + w121 * v121 + w221 * v221 + w321 * v321;
  val += w031 * v031 + w131 * v131 + w231 * v231 + w331 * v331;
  val += w002 * v002 + w102 * v102 + w202 * v202 + w302 * v302;
  val += w012 * v012 + w112 * v112 + w212 * v212 + w312 * v312;
  val += w022 * v022 + w122 * v122 + w222 * v222 + w322 * v322;
  val += w032 * v032 + w132 * v132 + w232 * v232 + w332 * v332;
  val += w003 * v003 + w103 * v103 + w203 * v203 + w303 * v303;
  val += w013 * v013 + w113 * v113 + w213 * v213 + w313 * v313;
  val += w023 * v023 + w123 * v123 + w223 * v223 + w323 * v323;
  val += w033 * v033 + w133 * v133 + w233 * v233 + w333 * v333;

  return val;
}