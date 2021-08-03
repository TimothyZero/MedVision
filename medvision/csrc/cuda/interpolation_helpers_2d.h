#include <float.h>

#include "cuda_helpers.h"

template <typename T>
__device__ T linear_2d_interpolate(
  const T* input,
  const int height, const int width,
  T y, T x,
  const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) return 0;

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y0 = (int)y;
  int x0 = (int)x;
  int y1;
  int x1;

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

  T ly = y - y0;
  T lx = x - x0;
  T hy = 1. - ly;
  T hx = 1. - lx;
  // do linear 2d interpolation
  T v1 = input[y0 * width + x0];
  T v2 = input[y0 * width + x1];
  T v3 = input[y1 * width + x0];
  T v4 = input[y1 * width + x1];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T>
__device__ void linear_2d_interpolate_gradient(
  const int height, const int width,
  T y, T x,
  T& w1, T& w2, T& w3, T& w4,
  int& x0, int& x1,
  int& y0, int& y1,
  const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x0 = x1 = y0 = y1 = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y0 = (int)y;
  x0 = (int)x;

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

  T ly = y - y0;
  T lx = x - x0;
  T hy = 1. - ly;
  T hx = 1. - lx;

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <typename T>
__device__ T nearest_2d_interpolate(
  const T* input,
  const int height, const int width,
  T y, T x,
  const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) return 0;

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y0 = (int)y;
  int x0 = (int)x;
  int y1;
  int x1;

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

  T ly = y - y0;
  T lx = x - x0;
  T hy = 1. - ly;
  T hx = 1. - lx;

  T v1 = input[y0 * width + x0];
  T v2 = input[y0 * width + x1];
  T v3 = input[y1 * width + x0];
  T v4 = input[y1 * width + x1];

  T w1 = 0, w2 = 0, w3 = 0, w4 = 0;
  if      (hy >= 0.5 && hx >= 0.5) w1 = 1;
  else if (hy >= 0.5 && lx >= 0.5) w2 = 1;
  else if (ly >= 0.5 && hx >= 0.5) w3 = 1;
  else if (ly >= 0.5 && lx >= 0.5) w4 = 1;

  T val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;

  return val;
}

template <typename T>
__device__ void nearest_2d_interpolate_gradient(
  const int height, const int width,
  T y, T x,
  T& w1, T& w2, T& w3, T& w4,
  int& x0, int& x1,
  int& y0, int& y1,
  const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x0 = x1 = y0 = y1 = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y0 = (int)y;
  x0 = (int)x;

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

  T ly = y - y0;
  T lx = x - x0;
  T hy = 1. - ly;
  T hx = 1. - lx;

  w1 = 0, w2 = 0, w3 = 0, w4 = 0;
  if      (hy >= 0.5 && hx >= 0.5) w1 = 1;
  else if (hy >= 0.5 && lx >= 0.5) w2 = 1;
  else if (ly >= 0.5 && hx >= 0.5) w3 = 1;
  else if (ly >= 0.5 && lx >= 0.5) w4 = 1;

  return;
}