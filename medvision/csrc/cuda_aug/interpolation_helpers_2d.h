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


//__device__ double matrix_a[16][16] = {
//  {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0},
//  {0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0},
//  {-3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0},
//  {9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1},
//  {-6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1},
//  {2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0},
//  {0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0},
//  {-6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1},
//  {4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1},
//};
//
//// naive
//template <typename T>
//__device__ T hermite_2d_interpolate(
//  const T* input,
//  const int height, const int width,
//  T y, T x,
//  const int index /* index for debug only*/) {
//  // deal with cases that inverse elements are out of feature map boundary
//  if (y < -1.0 || y > height || x < -1.0 || x > width) return 0;
//
//  if (y <= 0) y = 0;
//  if (x <= 0) x = 0;
//
//  int y1 = (int)y;
//  int x1 = (int)x;
//  int y0 = y1 - 1, y2 = y1 + 1, y3 = y1 + 2;
//  int x0 = x1 - 1, x2 = x1 + 1, x3 = x1 + 2;
//  float uy = y - y1;
//  float ux = x - x1;
//
//  // eq to padding
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
//  T v00 = input[y0 * width + x0];
//  T v10 = input[y0 * width + x1];
//  T v20 = input[y0 * width + x2];
//  T v30 = input[y0 * width + x3];
//
//  T v01 = input[y1 * width + x0];
//  T v11 = input[y1 * width + x1];
//  T v21 = input[y1 * width + x2];
//  T v31 = input[y1 * width + x3];
//
//  T v02 = input[y2 * width + x0];
//  T v12 = input[y2 * width + x1];
//  T v22 = input[y2 * width + x2];
//  T v32 = input[y2 * width + x3];
//
//  T v03 = input[y3 * width + x0];
//  T v13 = input[y3 * width + x1];
//  T v23 = input[y3 * width + x2];
//  T v33 = input[y3 * width + x3];
//
//  //
//  //  00    10    20    30
//  //  01    11    21    31
//  //           ?
//  //  02    12    22    32
//  //  03    13    23    33
//  //
//
//  T vector_x[16] = {
//    v11,
//    v21,
//    v12,
//    v22,
//    0.50 * (v21 - v01),
//    0.50 * (v31 - v11),
//    0.50 * (v22 - v02),
//    0.50 * (v32 - v12),
//    0.50 * (v12 - v10),
//    0.50 * (v22 - v20),
//    0.50 * (v13 - v11),
//    0.50 * (v23 - v21),
//    0.25 * (v22 + v00 - v02 - v20),
//    0.25 * (v32 + v10 - v12 - v30),
//    0.25 * (v23 + v01 - v03 - v21),
//    0.25 * (v33 + v11 - v13 - v31)
//  };
//
//  T alpha[16];
//  // a00 a10 a20 a30 a01 a11 a21 a31 a02 a12 a22 a32 a03 a13 a23 a33
//  // aij = > ux**i uy**j
//  for ( int i = 0; i < 16 ; i++)
//  {
//    alpha[i] = 0;  // ! important
//    for ( int j = 0; j < 16; j++)
//    {
//      alpha[i] += matrix_a[i][j] * vector_x[j];
//    }
//  }
//
//  T val = 0;
//  T uy_pow = 1;
//  for ( int j = 0; j < 4 ; j++){
//    T ux_pow = 1;
//    for ( int i = 0; i< 4 ; i++){
//      val += uy_pow * ux_pow * alpha[i + 4*j];
//      ux_pow *= ux;
//    }
//    uy_pow *= uy;
//  }
//
////  T val1 = linear_2d_interpolate(input, height, width, y, x, index);
////  printf("Hello from %f, %f\n", val1, val);
//  return val;
//}

template <typename T>
__device__ T hermite_2d_interpolate(
  const T* input,
  const int height, const int width,
  T y, T x,
  const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) return 0;

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y1 = (int)y;
  int x1 = (int)x;
  int y0 = y1 - 1, y2 = y1 + 1, y3 = y1 + 2;
  int x0 = x1 - 1, x2 = x1 + 1, x3 = x1 + 2;
  float uy = y - y1;
  float ux = x - x1;

  // eq to padding
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

  T v00 = input[y0 * width + x0];
  T v10 = input[y0 * width + x1];
  T v20 = input[y0 * width + x2];
  T v30 = input[y0 * width + x3];

  T v01 = input[y1 * width + x0];
  T v11 = input[y1 * width + x1];
  T v21 = input[y1 * width + x2];
  T v31 = input[y1 * width + x3];

  T v02 = input[y2 * width + x0];
  T v12 = input[y2 * width + x1];
  T v22 = input[y2 * width + x2];
  T v32 = input[y2 * width + x3];

  T v03 = input[y3 * width + x0];
  T v13 = input[y3 * width + x1];
  T v23 = input[y3 * width + x2];
  T v33 = input[y3 * width + x3];

  //
  //  00    10    20    30
  //  01    11    21    31
  //           ?
  //  02    12    22    32
  //  03    13    23    33
  //
  float ux2 = ux * ux;
  float ux3 = ux * ux * ux;
  float uy2 = uy * uy;
  float uy3 = uy * uy * uy;

  T w00 = 0.25*ux3*uy3 - 0.5*ux3*uy2 + 0.25*ux3*uy - 0.5*ux2*uy3 + ux2*uy2 - 0.5*ux2*uy + 0.25*ux*uy3 - 0.5*ux*uy2 + 0.25*ux*uy;
  T w10 = -0.75*ux3*uy3 + 1.5*ux3*uy2 - 0.75*ux3*uy + 1.25*ux2*uy3 - 2.5*ux2*uy2 + 1.25*ux2*uy - 0.5*uy3 + uy2 - 0.5*uy;
  T w20 = 0.75*ux3*uy3 - 1.5*ux3*uy2 + 0.75*ux3*uy - 1.0*ux2*uy3 + 2.0*ux2*uy2 - 1.0*ux2*uy - 0.25*ux*uy3 + 0.5*ux*uy2 - 0.25*ux*uy;
  T w30 = -0.25*ux3*uy3 + 0.5*ux3*uy2 - 0.25*ux3*uy + 0.25*ux2*uy3 - 0.5*ux2*uy2 + 0.25*ux2*uy;
  T w01 = -0.75*ux3*uy3 + 1.25*ux3*uy2 - 0.5*ux3 + 1.5*ux2*uy3 - 2.5*ux2*uy2 + ux2 - 0.75*ux*uy3 + 1.25*ux*uy2 - 0.5*ux;
  T w11 = 2.25*ux3*uy3 - 3.75*ux3*uy2 + 1.5*ux3 - 3.75*ux2*uy3 + 6.25*ux2*uy2 - 2.5*ux2 + 1.5*uy3 - 2.5*uy2 + 1;
  T w21 = -2.25*ux3*uy3 + 3.75*ux3*uy2 - 1.5*ux3 + 3.0*ux2*uy3 - 5.0*ux2*uy2 + 2.0*ux2 + 0.75*ux*uy3 - 1.25*ux*uy2 + 0.5*ux;
  T w31 = 0.75*ux3*uy3 - 1.25*ux3*uy2 + 0.5*ux3 - 0.75*ux2*uy3 + 1.25*ux2*uy2 - 0.5*ux2;
  T w02 = 0.75*ux3*uy3 - 1.0*ux3*uy2 - 0.25*ux3*uy - 1.5*ux2*uy3 + 2.0*ux2*uy2 + 0.5*ux2*uy + 0.75*ux*uy3 - 1.0*ux*uy2 - 0.25*ux*uy;
  T w12 = -2.25*ux3*uy3 + 3.0*ux3*uy2 + 0.75*ux3*uy + 3.75*ux2*uy3 - 5.0*ux2*uy2 - 1.25*ux2*uy - 1.5*uy3 + 2.0*uy2 + 0.5*uy;
  T w22 = 2.25*ux3*uy3 - 3.0*ux3*uy2 - 0.75*ux3*uy - 3.0*ux2*uy3 + 4.0*ux2*uy2 + 1.0*ux2*uy - 0.75*ux*uy3 + 1.0*ux*uy2 + 0.25*ux*uy;
  T w32 = -0.75*ux3*uy3 + 1.0*ux3*uy2 + 0.25*ux3*uy + 0.75*ux2*uy3 - 1.0*ux2*uy2 - 0.25*ux2*uy;
  T w03 = -0.25*ux3*uy3 + 0.25*ux3*uy2 + 0.5*ux2*uy3 - 0.5*ux2*uy2 - 0.25*ux*uy3 + 0.25*ux*uy2;
  T w13 = 0.75*ux3*uy3 - 0.75*ux3*uy2 - 1.25*ux2*uy3 + 1.25*ux2*uy2 + 0.5*uy3 - 0.5*uy2;
  T w23 = -0.75*ux3*uy3 + 0.75*ux3*uy2 + 1.0*ux2*uy3 - 1.0*ux2*uy2 + 0.25*ux*uy3 - 0.25*ux*uy2;
  T w33 = 0.25*ux3*uy3 - 0.25*ux3*uy2 - 0.25*ux2*uy3 + 0.25*ux2*uy2;

  T val = w00 * v00 + w10 * v10 + w20 * v20 + w30 * v30 + w01 * v01 + w11 * v11 + w21 * v21 + w31 * v31 + w02 * v02 + w12 * v12 + w22 * v22 + w32 * v32 + w03 * v03 + w13 * v13 + w23 * v23 + w33 * v33;

  return val;
}