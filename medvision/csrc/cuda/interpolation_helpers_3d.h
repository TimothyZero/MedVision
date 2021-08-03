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