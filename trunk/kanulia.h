#ifndef _MANDELBROT_KERNAL_h_
#define _MANDELBROT_KERNAL_h_

#include <vector_types.h>

// 4D rotations angles for Julia 4D
__constant__ float aanglexw;
__constant__ float aangleyw;
__constant__ float aanglexy;
__constant__ float aanglexz;

// 4D position of Julia4D seed point
__constant__ double xJS;
__constant__ double yJS;
__constant__ double zJS;
__constant__ double wJS;

__constant__ int crn; // Max itération de la Julia
__constant__ int numSM; // KO number of multiprocessors on device
__constant__ int imgW; // KO image Width
__constant__ int imgH; // KO image Height

extern "C" void RunJulia4Drepart(uchar4 *dst, const int imageW, const int imageH,
					const double xOff, const double yOff, const double scale,
					const double xJOff, const double yJOff, const double scaleJ,
					const float xblur, const float yblur, // blur coeff for julia 4D
					const uchar4 colors, const int frame, const int animationFrame, const int mode, const int numSMs, const int julia, const int julia4D);
//extern "C" void RunJulia4D1_sm13(uchar4 *dst, const int imageW, const int imageH, const double xOff, const double yOff, const double scale, const double xJOff, const double yJOff, const double scaleJ, const uchar4 colors, const int frame, const int animationFrame, const int mode, const int numSMs, const int julia, const int julia4D);

extern "C" int inEmulationMode();

template<class T> __device__ inline void rotate4(T *px, T *py, T *pz, T *pw);

#endif
